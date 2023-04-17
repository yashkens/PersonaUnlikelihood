import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class EnPersonaChat(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.columns = ['context_3', 'context_2', 'context_1', 'response']

    def __getitem__(self, idx):
        flatten = lambda l: [item for sublist in l for item in sublist]
        encodings = []
        for column in self.columns:
            input_tokens = self.data.iloc[idx][column]
            if not input_tokens.strip():
                continue
            encoding = self.tokenizer.encode(input_tokens) + [self.tokenizer.eos_token_id]
            encodings.append(encoding)
        encodings = flatten(encodings)
        return torch.tensor(encodings)

    def __len__(self):
        return len(self.data)


class EnPersonaChatUnlikelihood(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.columns = ['context_3', 'context_2', 'context_1', 'response']

    def encode_negatives(self, idx):
        negatives = self.data.iloc[idx]['negatives']
        if type(negatives) != str:
            return torch.zeros(5, 1)
        negatives = negatives.split('\n')[:5]
        neg_encodings = []
        for neg_cand in negatives:
            encoding = self.tokenizer.encode(neg_cand, return_tensors='pt')
            neg_encodings.append(encoding.squeeze(0))
        if len(neg_encodings) < 5:  # это будет супер тупо, когда буду брать больше 5
            for i in range(5 - len(neg_encodings)):
                neg_encodings.append(torch.zeros(1))
        neg_encodings = pad_sequence(neg_encodings, batch_first=True, padding_value=50256)
        return neg_encodings

    def encode_persona(self, idx):
        persona = self.data.iloc[idx]['speaker_persona'] + ' <|persona|>'
        encoding = self.tokenizer.encode(
            persona,
            return_tensors='pt'
            )
        return encoding

    def __getitem__(self, idx):
        persona_encoding = self.encode_persona(idx)

        full_input = ''
        context = ''
        for column in self.columns:
            input_tokens = self.data.iloc[idx][column]
            if not input_tokens.strip():
                continue
            full_input += input_tokens
            if column != 'response':
                context += input_tokens
                context += ' <|endoftext|> '
                full_input += ' <|endoftext|> '

        encoding = self.tokenizer.encode(
            full_input,
            return_tensors='pt',
        )
        encoding = torch.cat([persona_encoding.squeeze(0), encoding.squeeze(0)], dim=-1)
        context_encoding = self.tokenizer.encode(
            full_input,
            return_tensors='pt',
        )
        context_encoding = torch.cat([persona_encoding.squeeze(0), context_encoding.squeeze(0)], dim=-1)
        negative_encoding = self.encode_negatives(idx)
        return {'input_ids': encoding, 'negatives': negative_encoding, 'context': context_encoding}

    def __len__(self):
        return len(self.data)
