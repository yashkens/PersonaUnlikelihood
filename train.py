from transformers import AutoTokenizer
import random
import wandb
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from models import DialoGPTUnlikelihoodModel
from custom_datasets import PersonaChatDataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--debug', help="if True, logging is in cmd instead of wandb", action='store_true')
parser.add_argument("--loss", help="choose loss: nll or ul", type=str, default='nll')
parser.add_argument("--bs", help="batch size", type=int, default=1)


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
            'value': 1e-04
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
    inputs = []
    for ex in examples:
        inputs.append(ex['input_ids'])
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=50256)}


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


def collate_with_negatives_separately(examples):
    inputs, rewards = [], []
    for ex in examples:
        inputs.append(ex['input_ids'])
        rewards.append(ex['reward'])
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=50256),
            'reward': pad_sequence(rewards, batch_first=True, padding_value=50256)}


def prepare_data(tokenizer, data_part, loss_type, batch_size):
    if data_part == 'only-valid':
        df = pd.read_csv('data/valid.csv')
        valid = df.iloc[:1500]
        train = df.iloc[1500:]
    else:
        valid = pd.read_csv('data/valid.csv')
        train = pd.read_csv('data/train.csv')

    if loss_type == 'nll':
        valid_dataset = PersonaChatDataset(valid, tokenizer, add_negatives=False)
        train_dataset = PersonaChatDataset(train, tokenizer, add_negatives=False)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
    else:
        valid_dataset = PersonaChatDataset(valid, tokenizer, add_negatives=True)
        train_dataset = PersonaChatDataset(train, tokenizer, add_negatives=True)
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
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|persona|>']})
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        model.resize_token_embeddings(len(tokenizer))

        train_dataloader, valid_dataloader = prepare_data(
            tokenizer,
            config.data_part,
            config.loss_type,
            config.batch_size
        )

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)
        device = 'cuda'
        if config.loss_type == 'nll':
            answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device='cuda', parallel=False)
        else:
            answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device=device, ul_training=True)
        trained_model = answer_model.train(train_dataloader, valid_dataloader, optimizer, log_wandb=True, sample=False)


def debug_train(loss_type, batch_size):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer._pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|persona|>']})
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.resize_token_embeddings(len(tokenizer))

    train_dataloader, valid_dataloader = prepare_data(
        tokenizer,
        'only-valid',
        loss_type=loss_type,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-04)
    answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device='cuda', parallel=False, ul_training=True)
    trained_model = answer_model.train(train_dataloader, valid_dataloader, optimizer, log_wandb=False, sample=True)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        debug_train(args.loss, args.bs)
    else:
        wandb.login()

        sweep_config = create_config()
        sweep_id = wandb.sweep(sweep_config, project="unlikelihood-loss")
        wandb.agent(sweep_id, train_net)
