import torch
import wandb
import torch.nn.functional as F
from torch.nn import DataParallel
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class DialoGPTUnlikelihoodModel:

    def __init__(self, model, tokenizer, device, ul_training=False, parallel=False):
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

    def get_mle_loss(self, notnull, batch_rewards, scores_view, targets_view):
        mle_notnull = notnull & (batch_rewards > 0).unsqueeze(1).expand_as(notnull)
        mle_target_tokens = mle_notnull.long().sum()
        mle_loss = (
                F.nll_loss(
                    scores_view, targets_view, reduction='none'
                ).view_as(mle_notnull)
                * mle_notnull.float()
        ).sum()
        if mle_target_tokens > 0:
            mle_loss /= mle_target_tokens
        return mle_loss

    def get_ul_loss(self, notnull, batch_rewards, scores_view, targets_view):
        ul_notnull = notnull & (batch_rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum()
        range_ = torch.arange(targets_view.size(0)).to(self.device)
        ul_scores = scores_view[range_, targets_view]
        ul_loss = (
                -torch.log(torch.clamp(1.0 - ul_scores.exp(), min=1e-6)).view_as(
                    ul_notnull
                )
                * ul_notnull.float()
        ).sum()
        # TODO: why parlAI logs losses before average?
        if ul_target_tokens > 0:
            ul_loss /= ul_target_tokens
        return ul_loss

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                ids = batch['input_ids'].to(self.device)
                batch_rewards = batch['rewards'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                scores = output.logits

                scores = F.log_softmax(scores, dim=-1)
                scores_view = scores.view(-1, scores.size(-1))
                targets = batch
                targets_view = targets.view(-1)

                notnull = targets.ne(self.tokenizer.pad_token_id[0])
                mle_loss = self.get_mle_loss(notnull, batch_rewards, scores_view, targets_view)
                ul_loss = self.get_ul_loss(notnull, batch_rewards, scores_view, targets_view)

                loss = mle_loss + ul_loss

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
                batch_rewards = batch['rewards'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                scores = output.logits

                scores = F.log_softmax(scores, dim=-1)
                scores_view = scores.view(-1, scores.size(-1))
                targets = batch
                targets_view = targets.view(-1)

                notnull = targets.ne(self.tokenizer.pad_token_id[0])
                mle_loss = self.get_mle_loss(notnull, batch_rewards, scores_view, targets_view)
                print(f'mle loss: {mle_loss:.4f}')
                ul_loss = self.get_ul_loss(notnull, batch_rewards, scores_view, targets_view)
                print(f'ul loss: {ul_loss:.4f}')

                loss = mle_loss + ul_loss
                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()
                logger.info(f'Got loss: {loss:.4f}')

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if log_wandb:
                    losses_to_log = {'batch train NLL loss': mle_loss, 'batch train loss': loss.item(),
                                     'batch train UL loss': ul_loss}
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
