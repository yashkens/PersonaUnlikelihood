import torch
from torch.utils.data import Dataset


MAX_NEG_EXAMPLES = 2


class PersonaChatDataset(Dataset):
    def __init__(self, data, tokenizer, mask_context):
        self.tokenizer = tokenizer
        self.data = data
        self.mask_context_for_positives = mask_context
        self.columns = ['context_3', 'context_2', 'context_1', 'response']

    def encode_persona(self, idx):
        persona = self.data.iloc[idx]['speaker_persona'] + ' <|persona|> '
        encoding = self.tokenizer.encode(
            persona,
            return_tensors='pt'
            )
        return encoding

    def __getitem__(self, idx):
        persona_encoding = self.encode_persona(idx).squeeze(0)

        context = ''
        response = ''
        for column in self.columns:
            input_tokens = self.data.iloc[idx][column]
            if not input_tokens.strip():
                continue
            if column != 'response':
                context += input_tokens
                context += ' <|endoftext|> '
            else:
                response += input_tokens

        response_encoding = self.tokenizer.encode(response, return_tensors='pt').squeeze(0)
        if context:
            context_encoding = self.tokenizer.encode(context, return_tensors='pt').squeeze(0)
            encoding = torch.cat([persona_encoding, context_encoding, response_encoding], dim=-1)
            context_len = context_encoding.shape[-1]
        else:
            encoding = torch.cat([persona_encoding, response_encoding], dim=-1)
            context_len = 0

        if self.mask_context_for_positives:
            mask = [0] * (persona_encoding.shape[-1] + context_len)
        else:
            mask = [1] * (persona_encoding.shape[-1] + context_len)
        mask.extend([1] * response_encoding.shape[-1])
        mask = torch.tensor(mask)

        return {'input_ids': encoding, 'mask': mask}

    def __len__(self):
        return len(self.data)


class NegativesAsSeparateExDataset(Dataset):
    def __init__(self, data, tokenizer, mask_context):
        self.tokenizer = tokenizer
        self.data = data
        self.mask_context_for_positives = mask_context
        self.columns = ['context_3', 'context_2', 'context_1', 'response']

    def encode_persona(self, idx):
        persona = self.data.iloc[idx]['speaker_persona'] + ' <|persona|> '
        encoding = self.tokenizer.encode(
            persona,
            return_tensors='pt'
            )
        return encoding

    def __getitem__(self, idx):

        persona_encoding = self.encode_persona(idx).squeeze(0)
        reward_value = int(self.data.iloc[idx]['reward'])

        context = ''
        response = ''
        for column in self.columns:
            input_tokens = self.data.iloc[idx][column]
            if not input_tokens.strip():
                continue
            if column != 'response':
                context += input_tokens
                context += ' <|endoftext|> '
            else:
                response += input_tokens

        response_encoding = self.tokenizer.encode(response, return_tensors='pt').squeeze(0)

        if context:
            context_encoding = self.tokenizer.encode(context, return_tensors='pt').squeeze(0)
            encoding = torch.cat([persona_encoding, context_encoding, response_encoding], dim=-1)
            context_len = context_encoding.shape[-1]
        else:
            encoding = torch.cat([persona_encoding, response_encoding], dim=-1)
            context_len = 0

        # rewards are given only to responses
        # so loss will be computed only for responses too
        reward_seq = [0] * (persona_encoding.shape[-1] + context_len)
        # if true allow losses on all tokens for positive examples
        if not self.mask_context_for_positives:
            if reward_value > 0:
                reward_seq = [reward_value] * (persona_encoding.shape[-1] + context_len)
        reward_seq.extend([reward_value] * response_encoding.shape[-1])
        reward_seq = torch.tensor(reward_seq)
        return {'input_ids': encoding, 'reward': reward_seq}

    def __len__(self):
        return len(self.data)
