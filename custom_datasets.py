import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


MAX_NEG_EXAMPLES = 2


class PersonaChatDataset(Dataset):
    def __init__(self, data, tokenizer, add_negatives):
        self.tokenizer = tokenizer
        self.data = data
        self.columns = ['context_3', 'context_2', 'context_1', 'response']
        self.add_negatives = add_negatives

    def encode_negatives(self, idx):
        negatives = self.data.iloc[idx]['negatives']
        if type(negatives) != str:
            return torch.zeros(MAX_NEG_EXAMPLES, 1)
        negatives = negatives.split('\n')[:MAX_NEG_EXAMPLES]
        neg_encodings = []
        for neg_cand in negatives:
            encoding = self.tokenizer.encode(neg_cand, return_tensors='pt')
            neg_encodings.append(encoding.squeeze(0))
        if len(neg_encodings) < MAX_NEG_EXAMPLES:  # TODO: не добавлять лишней фигни, если примеров больше нет
            for i in range(MAX_NEG_EXAMPLES - len(neg_encodings)):
                neg_encodings.append(torch.zeros(1))
        neg_encodings = pad_sequence(neg_encodings, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return neg_encodings

    def encode_persona(self, idx):
        persona = self.data.iloc[idx]['speaker_persona'] + ' <|persona|> '
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

        encoding = self.tokenizer.encode(full_input, return_tensors='pt')
        encoding = torch.cat([persona_encoding.squeeze(0), encoding.squeeze(0)], dim=-1)

        if not self.add_negatives:
            return {'input_ids': encoding}
        else:
            context_encoding = self.tokenizer.encode(full_input, return_tensors='pt')
            context_encoding = torch.cat([persona_encoding.squeeze(0), context_encoding.squeeze(0)], dim=-1)
            negative_encoding = self.encode_negatives(idx)
            return {'input_ids': encoding, 'negatives': negative_encoding, 'context': context_encoding}

    def __len__(self):
        return len(self.data)
