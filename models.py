import torch
import wandb
from collections import Counter
import torch.nn.functional as F


class DialoGPTModel:

    def __init__(self, model, device, parallel=False):
        self.model = model
        self.device = device
        self.parallel = parallel
        if self.parallel:
            self.model = self.model.to(device)
        else:
            self.model.to(device)

    def __call__(self, input_ids):

        self.model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            output = self.model(input_ids=input_ids)
        return output['logits']

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                tokens = batch.to(self.device)
                labels = batch.to(self.device)

                output = self.model(input_ids=tokens, labels=labels)
                loss = output['loss']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))

        return avg_val_loss, val_ppl

    def train(self, train_dataloader, val_dataloader, optimizer, n_epoch=3, checkpoint_step=100):

        for epoch in range(n_epoch):

            self.model.train()

            train_loss = 0
            for step_num, batch in enumerate(train_dataloader):
                ids = batch.to(self.device)
                labels = batch.to(self.device)

                output = self.model(input_ids=ids, labels=labels)
                loss = output['loss']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"batch train loss": loss.item()})

                if step_num != 0 and step_num % checkpoint_step == 0:
                    step_loss = train_loss / step_num
                    step_ppl = torch.exp(torch.tensor(step_loss))
                    wandb.log({"step train loss": step_loss, "step train ppl": step_ppl, "step": step_num})
                    step_val_loss, step_val_ppl = self.validate(val_dataloader)
                    wandb.log({"step val loss": step_val_loss, "step val ppl": step_val_ppl})

            avg_val_loss, val_ppl = self.validate(val_dataloader)
            avg_train_loss = train_loss / len(train_dataloader)
            train_ppl = torch.exp(torch.tensor(avg_train_loss))

            wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
                       "train ppl": train_ppl, "val ppl": val_ppl,
                       "epoch": epoch})
        return self.model


class DialoGPTUnlikelihoodModel:

    def __init__(self, model, device, parallel=False):
        self.model = model
        self.device = device
        self.parallel = parallel
        if self.parallel:
            self.model = self.model.to(device)
        else:
            self.model.to(device)

    def __call__(self, input_ids):

        self.model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            output = self.model(input_ids=input_ids)
        return output['logits']

    def get_penalize_weights(self, neg_inputs, target_inputs):
        negative_history = []
        target_history = []
        running_negative = Counter()
        running_target = Counter()

        neg_counter = Counter(neg_inputs[0].view(-1).tolist())
        for neg_toks in neg_inputs[1:]:
            neg_counter += Counter(neg_toks.view(-1).tolist())
        negative_history.append(neg_counter)
        target_history.append(Counter(target_inputs.view(-1).tolist()))
        running_negative += negative_history[-1]
        running_target += target_history[-1]

        threshold = 1e-3
        neg_sum = sum(running_negative.values())
        target_sum = sum(running_target.values())
        to_penalize = {
            w: (v / neg_sum) - (running_target.get(w, 0) / target_sum)
            for w, v in running_negative.items()
        }
        to_penalize = {
            w: 1 for w, v in to_penalize.items() if v >= threshold
        }
        return to_penalize

    def get_loss_from_penalties(self, penalties):
        log_penalties = []
        beta = 1
        for pack in penalties:
            for p in pack:
                log_p = beta * torch.log(1 - p)
                log_penalties.append(log_p.reshape(1))
        if log_penalties:
            ul_loss = - torch.sum(torch.cat(log_penalties), dim=0)
        else:
            ul_loss = 0
        return ul_loss

    def get_loss_from_penalties_torch(self, penalties):
        log_penalties = torch.log(1 - penalties)
        ul_loss = - torch.sum(log_penalties)
        return ul_loss

    def get_ul_loss(self, batch):
        neg_inputs = batch['negatives'].to(self.device)
        target_inputs = batch['input_ids']
        context_tokens = batch['context'].to(self.device)
        batch_num = target_inputs.shape[0]
        all_losses = []
        for batch_ind in range(batch_num):  # по батчам
            target_inputs_no_context = target_inputs[batch_ind][len(context_tokens[batch_ind]):]
            if (neg_inputs[batch_ind] == 0).all():
                continue
            to_penalize = self.get_penalize_weights(neg_inputs[batch_ind], target_inputs_no_context)

            if len(context_tokens[batch_ind]) > 0:
                neg_cands_num = neg_inputs[batch_ind].shape[0]
                context = context_tokens[batch_ind].unsqueeze(0)
                context = context.expand(neg_cands_num, -1)
                new_input_seq = torch.cat((context, neg_inputs[batch_ind]), dim=-1)
                new_input_seq = new_input_seq
                scores = self.model(new_input_seq).logits
                neg_scores = scores[:, len(context_tokens[batch_ind].view(-1)):, :]
                neg_scores = F.softmax(neg_scores, dim=-1)

                neg_tokens = neg_inputs[batch_ind].unsqueeze(-1)
                penalties = torch.gather(neg_scores, -1, neg_tokens)
                # проверочка
                # one_token = neg_tokens[0][0][0]
                # print(f'THE TOKEN: {one_token}')
                # print(f'THE SCORE OF TOKEN: {neg_scores[0][0][one_token]}')
                ul_loss = self.get_loss_from_penalties_torch(penalties)
                print(f'new loss: {ul_loss}')

            penalties = []
            for cand in neg_inputs[batch_ind]:

                if len(context_tokens[batch_ind]) > 0:
                    input_seq = torch.cat((context_tokens[batch_ind], cand), dim=-1)
                else:
                    input_seq = cand
                input_seq = input_seq.to(self.device)
                scores = self.model(input_seq).logits
                neg_scores = scores[len(context_tokens[batch_ind].view(-1)):, :]
                neg_scores = F.softmax(neg_scores, dim=-1)
                penalties_by_token = []
                for i, token in enumerate(cand):
                    if int(token) not in to_penalize.keys():
                        continue
                    penalties_by_token.append(neg_scores[i][token])
                penalties.append(penalties_by_token)

            ul_loss = self.get_loss_from_penalties(penalties)
            print(f'old loss: {ul_loss}')
            if ul_loss != 0:
                all_losses.append(ul_loss.reshape(1))
        if not all_losses:
            return 0
        final_ul_loss = torch.mean(torch.cat(all_losses, dim=0))
        return final_ul_loss

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                tokens = batch['input_ids'].to(self.device)
                labels = batch['input_ids'].to(self.device)

                output = self.model(input_ids=tokens, labels=labels)
                loss = output['loss']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()
                # TODO: add UL loss here too?

        avg_val_loss = val_loss / len(val_dataloader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))

        return avg_val_loss, val_ppl

    def train(self, train_dataloader, val_dataloader, optimizer, n_epoch=3, checkpoint_step=100):

        for epoch in range(n_epoch):

            self.model.train()

            train_loss = 0
            for step_num, batch in enumerate(train_dataloader):
                ids = batch['input_ids'].to(self.device)
                labels = batch['input_ids'].to(self.device)

                output = self.model(input_ids=ids, labels=labels)
                loss = output['loss']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                ul_loss = self.get_ul_loss(batch)
                # wandb.log({
                #     "batch train UL loss": ul_loss,
                #     "batch train NLL loss": loss
                # })
                if ul_loss != 0:
                    loss = loss + ul_loss

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                # wandb.log({"batch train loss": loss.item()})

                if step_num != 0 and step_num % checkpoint_step == 0:
                    step_loss = train_loss / step_num
                    step_ppl = torch.exp(torch.tensor(step_loss))
                    # wandb.log({
                    #     "step train loss": step_loss,
                    #     "step train ppl": step_ppl,
                    #     "step": step_num + epoch * len(train_dataloader)
                    # })
                    step_val_loss, step_val_ppl = self.validate(val_dataloader)
                    # wandb.log({
                    #     "step val loss": step_val_loss,
                    #     "step val ppl": step_val_ppl
                    # })

            avg_val_loss, val_ppl = self.validate(val_dataloader)
            avg_train_loss = train_loss / len(train_dataloader)
            train_ppl = torch.exp(torch.tensor(avg_train_loss))

            # wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
            #            "train ppl": train_ppl, "val ppl": val_ppl,
            #            "epoch": epoch})
        return self.model
