import torch
import wandb
import torch.nn.functional as F
from torch.nn import DataParallel
import logging
import sys
from copy import copy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class DialoGPTUnlikelihoodModel:

    def __init__(self, model, tokenizer, device, ul_training=False, parallel=False):
        self.model = model
        self.tokenizer = tokenizer
        self.ul_training = ul_training
        self.device = device
        self.parallel = parallel
        self.save_dir = 'ul_save' if ul_training else 'nll_save'
        if self.parallel:
            self.model = DataParallel(self.model).to(device)
        else:
            self.model.to(device)

    def sample(self, ids, step_num, log_wandb):
        self.model.eval()
        with torch.no_grad():
            test_text = 'i like to go to country concerts on weekends.\ni have two dogs.\n' \
                        'my favorite music is country.\ni like to work on vintage cars. ' \
                        '<|persona|> do you have any pets? <|endoftext|>'
            new_ids = self.tokenizer.encode(test_text, return_tensors='pt').to(self.device)
            if self.parallel:
                output = self.model.module.generate(new_ids, temperature=0.9, max_length=150, pad_token_id=self.tokenizer.pad_token_id)
            else:
                output = self.model.generate(new_ids, temperature=0.9, max_length=150,
                                             pad_token_id=self.tokenizer.pad_token_id)
            test_res = self.tokenizer.decode(output[0]).replace(test_text, '')

            cur_ids = ids[0][ids[0] != self.tokenizer.pad_token_id]
            last_eos_inds = torch.where(cur_ids == self.tokenizer.eos_token_id)[0]
            if last_eos_inds.shape[-1] == 0:
                context_ids = cur_ids
            else:
                context_ids = cur_ids[:int(last_eos_inds[-1]) + 1]
            if self.parallel:
                output = self.model.module.generate(
                    context_ids.unsqueeze(0), temperature=0.9, max_length=150, pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                output = self.model.generate(
                    context_ids.unsqueeze(0), temperature=0.9, max_length=150, pad_token_id=self.tokenizer.pad_token_id
                )
            prompt = self.tokenizer.decode(context_ids)
            decoded_output = self.tokenizer.decode(output[0])
            result = decoded_output.replace(prompt, '')
            if not log_wandb:
                logger.info(f'Generated sample at step num {step_num}...')
                logger.info(f'Test Input: {test_text}')
                logger.info(f'Test Output: {test_res}')
                logger.info(f'Input: {prompt}')
                logger.info(f'Output: {result}')
        return prompt, result, test_text, test_res

    def get_mle_loss(self, notnull, batch_rewards, scores_view, targets_view):
        mle_notnull = notnull & (batch_rewards > 0).expand_as(notnull)
        mle_target_tokens = mle_notnull.long().sum()
        mle_losses = (
                F.nll_loss(
                    scores_view, targets_view, reduction='none', ignore_index=self.tokenizer.pad_token_id
                ).view_as(mle_notnull)
                * mle_notnull.float()
        )
        mle_loss = mle_losses.sum()
        if mle_target_tokens > 0:
            mle_loss /= mle_target_tokens
        return mle_loss

    def get_ul_loss(self, notnull, batch_rewards, scores_view, targets_view):
        ul_notnull = notnull & (batch_rewards < 0).expand_as(notnull)
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

        val_loss, val_only_nll = 0, 0
        positive_count = 0
        with torch.no_grad():
            for batch in val_dataloader:
                ids = batch['input_ids'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                scores = output.logits

                shift_scores = scores[..., :-1, :].contiguous()
                scores = F.log_softmax(shift_scores, dim=-1)
                scores_view = scores.view(-1, scores.size(-1))
                targets = ids
                shift_targets = targets[..., 1:].contiguous()
                targets_view = shift_targets.view(-1)
                notnull = shift_targets.ne(self.tokenizer.pad_token_id)

                if self.ul_training:
                    batch_rewards = batch['reward'].to(self.device)
                    shift_rewards = batch_rewards[..., 1:].contiguous()

                    mle_loss = self.get_mle_loss(notnull, shift_rewards, scores_view, targets_view)
                    ul_loss = self.get_ul_loss(notnull, shift_rewards, scores_view, targets_view)
                    loss = mle_loss + ul_loss
                    positive_count += ((batch_rewards > 0).sum(dim=-1) > 0).sum()
                else:
                    mask = batch['mask'].to(self.device)
                    shift_mask = mask[..., 1:].contiguous()
                    mle_loss = self.get_mle_loss(notnull, shift_mask, scores_view, targets_view)
                    loss = mle_loss
                    positive_count += mask.shape[0]

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()
                val_only_nll += mle_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        nll_only_loss = 0
        if positive_count > 0:
            nll_only_loss = val_only_nll / positive_count
        val_ppl = torch.exp(torch.tensor(nll_only_loss))
        return avg_val_loss, val_ppl

    def train(
            self,
            train_dataloader,
            val_dataloader,
            optimizer,
            n_epoch=3,
            checkpoint_step=50,
            log_wandb=False,
            sample=False,
            save_step=3000,
    ):

        if log_wandb and sample:
            sample_table = wandb.Table(columns=['step', 'input', 'output'])

        for epoch in range(n_epoch):

            self.model.train()

            train_loss, train_only_nll = 0, 0
            positive_count = 0
            for step_num, batch in enumerate(train_dataloader):

                ids = batch['input_ids'].to(self.device)
                output = self.model(input_ids=ids, labels=ids)
                scores = output.logits

                shift_scores = scores[..., :-1, :].contiguous()
                scores = F.log_softmax(shift_scores, dim=-1)
                scores_view = scores.view(-1, scores.size(-1))
                targets = ids
                shift_targets = targets[..., 1:].contiguous()
                targets_view = shift_targets.view(-1)
                notnull = shift_targets.ne(self.tokenizer.pad_token_id)

                if self.ul_training:
                    batch_rewards = batch['reward'].to(self.device)
                    shift_rewards = batch_rewards[..., 1:].contiguous()

                    mle_loss = self.get_mle_loss(notnull, shift_rewards, scores_view, targets_view)
                    ul_loss = self.get_ul_loss(notnull, shift_rewards, scores_view, targets_view)
                    loss = mle_loss + ul_loss
                    train_only_nll += mle_loss.item()
                    positive_count += ((batch_rewards > 0).sum(dim=-1) > 0).sum()
                else:
                    mask = batch['mask'].to(self.device)
                    shift_mask = mask[..., 1:].contiguous()
                    loss = self.get_mle_loss(notnull, shift_mask, scores_view, targets_view)
                    mle_loss = loss  # for wandb logging
                    ul_loss = 0  # for wandb logging
                    positive_count += mask.shape[0]

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                if log_wandb:
                    losses_to_log = {'batch train NLL loss': mle_loss, 'batch train loss': loss.item(),
                                     'batch train UL loss': ul_loss}
                    wandb.log(losses_to_log)

                if step_num % checkpoint_step == 0:
                    if step_num == 0:
                        step_num = 1
                    step_loss = train_loss / step_num

                    # calculating perplexity
                    nll_only_loss = 0
                    if positive_count > 0:
                        nll_only_loss = train_only_nll / positive_count
                    step_ppl = torch.exp(torch.tensor(nll_only_loss))

                    # generating samples
                    if sample:
                        prompt, result, test_prompt, test_result = self.sample(ids, step_num, log_wandb)
                        if log_wandb:
                            sample_table.add_data(step_num, prompt, result)
                            sample_table.add_data(step_num, test_prompt, test_result)
                            if self.ul_training:
                                wandb.log({"ul_generated_samples": copy(sample_table)})
                            else:
                                wandb.log({"nll_generated_samples": copy(sample_table)})

                    # validating
                    step_val_loss, step_val_ppl = self.validate(val_dataloader)

                    self.model.train()

                    # logging
                    if log_wandb:
                        wandb.log({
                            "step train loss": step_loss,
                            "step train ppl": step_ppl,
                            "step": step_num + epoch * len(train_dataloader)
                        })
                        wandb.log({"step val loss": step_val_loss, "step val ppl": step_val_ppl})

                # saving checkpoints
                if step_num != 0 and step_num % save_step == 0:
                    if self.parallel:
                        state_dict = self.model.module.state_dict()
                    else:
                        state_dict = self.model.state_dict()
                    torch.save(state_dict, f'{self.save_dir}/checkpoint_step_{step_num}_epoch_{epoch}.pt')

            avg_val_loss, val_ppl = self.validate(val_dataloader)
            avg_train_loss = train_loss / len(train_dataloader)

            # saving checkpoints after an epoch
            if self.parallel:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            torch.save(state_dict, f'{self.save_dir}/checkpoint_step_epoch_{epoch}.pt')

            if log_wandb:
                wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
                           "epoch": epoch})
        return self.model
