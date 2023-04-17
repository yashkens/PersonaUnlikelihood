from transformers import AutoTokenizer
import random
import wandb
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from models import DialoGPTModel, DialoGPTUnlikelihoodModel
from custom_datasets import EnPersonaChat, EnPersonaChatUnlikelihood


def create_config():
    sweep_config = {'method': 'grid'}
    metric = {
        'name': 'val F1',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'data_part': {
            'values': ['only-valid']
        }
    }
    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 2
        },
        'lr': {
            'value': 1e-05
        },
        'batch_size': {
            'value': 8
        },
        'seed': {
            'value': 42
        },
        'loss_type': {
            'value': 'ul'
        }
    })

    return sweep_config


def collate_regular(examples):
    return pad_sequence(examples, batch_first=True, padding_value=50256)


def collate_with_negatives(examples):
    inputs, negs, conts = [], [], []
    for ex in examples:
        inputs.append(ex['input_ids'])
        negs.append(ex['negatives'].T)
        conts.append(ex['context'])
    negatives = pad_sequence(negs, batch_first=True, padding_value=50256)
    negatives.transpose(2, 1)
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=50256),
            'negatives': negatives,
            'context': pad_sequence(conts, batch_first=True, padding_value=50256)}


def prepare_data(tokenizer, data_part, loss_type, batch_size):
    if data_part == 'only-valid':
        df = pd.read_csv('data/valid.csv')
        valid = df.iloc[:1500]
        train = df.iloc[1500:]
    else:
        valid = pd.read_csv('data/valid.csv')
        train = pd.read_csv('data/train.csv')

    if loss_type == 'nll':
        valid_dataset = EnPersonaChat(valid, tokenizer)
        train_dataset = EnPersonaChat(train, tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
    else:
        valid_dataset = EnPersonaChatUnlikelihood(valid, tokenizer)
        train_dataset = EnPersonaChatUnlikelihood(train, tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_with_negatives, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_with_negatives, shuffle=True
        )
    return train_dataloader, valid_dataloader


def train_net(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        name_str = f"{config.loss_type}-loss_{config.data_part}-data"
        run.name = name_str

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer._pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

        train_dataloader, valid_dataloader = prepare_data(
            tokenizer,
            config.data_part,
            config.loss_type,
            config.batch_size
        )

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-05)
        device = 'cuda'
        if config.loss_type == 'nll':
            answer_model = DialoGPTModel(model, device=device)
        else:
            answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device=device)

        trained_model = answer_model.train(train_dataloader, valid_dataloader, optimizer)


def debug_train():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer._pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|persona|>']})
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.resize_token_embeddings(len(tokenizer))

    train_dataloader, valid_dataloader = prepare_data(
        tokenizer,
        'only-valid',
        'ul',
        2
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-05)
    device = 'cuda'

    answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device=device, parallel=True)
    trained_model = answer_model.train(train_dataloader, valid_dataloader, optimizer)


if __name__ == "__main__":
    debug_train()
    # wandb.login()
    #
    # sweep_config = create_config()
    # sweep_id = wandb.sweep(sweep_config, project="unlikelihood-loss")
    # wandb.agent(sweep_id, train_net)
