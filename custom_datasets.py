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
            return torch.zeros(1, 10)
        negatives = negatives.split('\n')[:5]
        neg_encodings = []
        for neg_cand in negatives:
            encoding = self.tokenizer.encode(neg_cand, return_tensors='pt')
            neg_encodings.append(encoding.squeeze(0))
        neg_encodings = pad_sequence(neg_encodings, batch_first=True, padding_value=50256)
        print(f'neg_encodings_shape: {neg_encodings.shape}')
        return neg_encodings

    def __getitem__(self, idx):
        flatten = lambda l: [item for sublist in l for item in sublist]
        encodings = []
        context = []
        for column in self.columns:
            input_tokens = self.data.iloc[idx][column]
            if not input_tokens.strip():
                continue
            encoding = self.tokenizer.encode(input_tokens) + [self.tokenizer.eos_token_id]
            encodings.append(encoding)
            if column != 'response':
                context.append(encoding)
        encodings = flatten(encodings)
        if context:
            context = flatten(context)
        else:
            context = torch.tensor([])
        negative_encodings = self.encode_negatives(idx)
        return {'input_ids': torch.tensor(encodings), 'negatives': negative_encodings, 'context': torch.tensor(context)}

    def __len__(self):
        return len(self.data)
