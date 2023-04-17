import torch
import wandb
import torch.nn.functional as F
from torch.nn import DataParallel
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class DialoGPTUnlikelihoodModel:

    def __init__(self, model, tokenizer, device, ul_training=True, parallel=False):
        self.model = model
        self.tokenizer = tokenizer
        self.ul_training = ul_training
        self.device = device
        self.parallel = parallel
        if self.parallel:
            self.model = DataParallel(self.model).to(device)
        else:
            self.model.to(device)

    def sample(self, ids, step_num):
        logger.info(f'Generating a sample at step num {step_num}...')
        output = self.model.generate(ids, temperature=0.9, max_length=150)
        for k, out in enumerate(output):
            last_eos_ind = torch.where(ids[k] == self.tokenizer.eos_token_id)[0][-1]
            context_ids = ids[k][:int(last_eos_ind) + 1]
            prompt = self.tokenizer.decode(context_ids)
            decoded_output = self.tokenizer.decode(out)
            logger.info(f'Input: {prompt}')
            logger.info(f'Output: {decoded_output.replace(prompt, "")}')

    def get_loss_from_penalties(self, penalties):
        log_penalties = torch.log(1 - penalties)
        ul_loss = - torch.sum(log_penalties)
        return ul_loss

    def get_ul_loss(self, batch):
        neg_inputs = batch['negatives'].to(self.device)
        context_tokens = batch['context'].to(self.device)

        batch_num = neg_inputs.shape[0]
        all_losses = []
        for batch_ind in range(batch_num):  # TODO: убрать цикл, использовать торч
            if (neg_inputs[batch_ind] == 0).all():
                continue

            if len(context_tokens[batch_ind]) > 0:
                neg_cands_num = neg_inputs[batch_ind].shape[0]
                context = context_tokens[batch_ind].unsqueeze(0)
                context = context.expand(neg_cands_num, -1)
                new_input_seq = torch.cat((context, neg_inputs[batch_ind]), dim=-1)
            else:
                new_input_seq = neg_inputs[batch_ind]

            scores = self.model(new_input_seq).logits
            neg_scores = scores[:, len(context_tokens[batch_ind].view(-1)):, :]
            neg_scores = F.softmax(neg_scores, dim=-1)

            neg_tokens = neg_inputs[batch_ind].unsqueeze(-1)
            penalties = torch.gather(neg_scores, -1, neg_tokens)
            ul_loss = self.get_loss_from_penalties(penalties)

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
                ids = batch['input_ids'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                loss = output['loss']

                if self.ul_training:
                    ul_loss = self.get_ul_loss(batch)
                    if ul_loss != 0:
                        loss = loss + ul_loss

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))
        return avg_val_loss, val_ppl

    def train(
            self,
            train_dataloader,
            val_dataloader,
            optimizer,
            n_epoch=3,
            checkpoint_step=20,
            log_wandb=False,
            sample=False,
    ):

        for epoch in range(n_epoch):

            self.model.train()

            train_loss = 0
            for step_num, batch in enumerate(train_dataloader):
                logger.info(f'Step {step_num}')
                ids = batch['input_ids'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                nll_loss = output['loss']
                loss = nll_loss
                ul_loss = 0

                if self.ul_training:
                    ul_loss = self.get_ul_loss(batch)
                    logger.info(f'Got UL loss: {ul_loss:.4f}')
                    if ul_loss != 0:
                        loss = nll_loss + ul_loss

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()
                logger.info(f'Got loss: {loss:.4f}')

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if log_wandb:
                    losses_to_log = {'batch train NLL loss': nll_loss, 'batch train loss': loss.item()}
                    if self.ul_training:
                        losses_to_log['batch train UL loss'] = ul_loss
                    wandb.log(losses_to_log)

                if step_num != 0 and step_num % checkpoint_step == 0:
                    step_loss = train_loss / step_num
                    step_ppl = torch.exp(torch.tensor(step_loss))
                    if sample:
                        self.sample(ids, step_num)

                    step_val_loss, step_val_ppl = self.validate(val_dataloader)

                    if log_wandb:
                        wandb.log({
                            "step train loss": step_loss,
                            "step train ppl": step_ppl,
                            "step": step_num + epoch * len(train_dataloader)
                        })
                        wandb.log({"step val loss": step_val_loss, "step val ppl": step_val_ppl})

            avg_val_loss, val_ppl = self.validate(val_dataloader)
            avg_train_loss = train_loss / len(train_dataloader)
            train_ppl = torch.exp(torch.tensor(avg_train_loss))

            if log_wandb:
                wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
                           "train ppl": train_ppl, "val ppl": val_ppl,
                           "epoch": epoch})
        return self.model
