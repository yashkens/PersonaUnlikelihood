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
from custom_datasets import PersonaChatDataset, NegativesAsSeparateExDataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--debug', help="if True, logging is in cmd instead of wandb", action='store_true')
parser.add_argument("--loss", help="choose loss: nll or ul", type=str, default='nll')
parser.add_argument("--bs", help="batch size", type=int, default=1)
parser.add_argument("--lr", help="learning rate", type=int, default=1e-04)
parser.add_argument("--data", help="'only-valid' for testing or 'full' for training", type=str, default='full')
parser.add_argument("--mask_context", help="if True, loss will be computed only on response tokens", action='store_true')
parser.add_argument("--parallel", help="if True, use multiple gpus", action='store_true')
parser.add_argument('-n', "--project_name", help="name of the wandb project", type=str, default='unlikelihood-loss')

PAD_VALUE = 50257


def create_config(args):
    sweep_config = {'method': 'grid'}
    metric = {
        'name': 'step val ppl',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'loss_type': {
            'values': [args.loss]
        }
    }
    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 2
        },
        'lr': {
            'value': args.lr
        },
        'batch_size': {
            'value': args.bs
        },
        'seed': {
            'value': 42
        },
        'data_part': {
            'value': args.data
        },
        'mask_context': {
            'value': args.mask_context
        },
        'parallel': {
            'value': args.parallel
        }
    })

    return sweep_config


def collate_regular(examples):
    inputs = []
    masks = []
    for ex in examples:
        inputs.append(ex['input_ids'])
        masks.append(ex['mask'])
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=PAD_VALUE),
            'mask': pad_sequence(masks, batch_first=True, padding_value=0)}


def collate_with_negatives(examples):
    inputs, negs, conts = [], [], []
    for ex in examples:
        inputs.append(ex['input_ids'])
        negs.append(ex['negatives'].T)
        conts.append(ex['context'])
    negatives = pad_sequence(negs, batch_first=True, padding_value=PAD_VALUE)
    negatives.transpose(2, 1)
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=PAD_VALUE),
            'negatives': negatives,
            'context': pad_sequence(conts, batch_first=True, padding_value=PAD_VALUE)}


def collate_with_negatives_separately(examples):
    inputs, rewards = [], []
    for ex in examples:
        inputs.append(ex['input_ids'])
        rewards.append(ex['reward'])
    return {'input_ids': pad_sequence(inputs, batch_first=True, padding_value=PAD_VALUE),
            'reward': pad_sequence(rewards, batch_first=True, padding_value=0)}


def prepare_data(tokenizer, data_part, loss_type, batch_size, mask_context):
    if loss_type == 'nll':
        data_path = 'data'
    else:
        data_path = 'data_neg_sep'
    if data_part == 'only-valid':
        df = pd.read_csv(f'{data_path}/valid.csv')
        valid = df.iloc[:1500]
        train = df.iloc[1500:]
    else:
        valid = pd.read_csv(f'{data_path}/valid.csv')
        train = pd.read_csv(f'{data_path}/train.csv')

    if loss_type == 'nll':
        valid_dataset = PersonaChatDataset(valid, tokenizer, mask_context)
        train_dataset = PersonaChatDataset(train, tokenizer, mask_context)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_regular, shuffle=True
        )
    else:
        valid_dataset = NegativesAsSeparateExDataset(valid, tokenizer, mask_context)
        train_dataset = NegativesAsSeparateExDataset(train, tokenizer, mask_context)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_with_negatives_separately, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_with_negatives_separately, shuffle=True
        )
    return train_dataloader, valid_dataloader


def train_net(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        masking_status = 'with_context'
        if config.mask_context:
            masking_status = 'mask_context'
        name_str = f"{config.loss_type}_loss-{masking_status}"
        run.name = name_str

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|persona|>']})
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        model.resize_token_embeddings(len(tokenizer))

        train_dataloader, valid_dataloader = prepare_data(
            tokenizer,
            config.data_part,
            config.loss_type,
            config.batch_size,
            config.mask_context
        )

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)
        device = 'cuda'
        if config.loss_type == 'nll':
            answer_model = DialoGPTUnlikelihoodModel(
                model, tokenizer, device=device, ul_training=False, parallel=config.parallel
            )
        else:
            answer_model = DialoGPTUnlikelihoodModel(
                model, tokenizer, device=device, ul_training=True, parallel=config.parallel
            )
        trained_model = answer_model.train(
            train_dataloader,
            valid_dataloader,
            optimizer,
            log_wandb=True,
            sample=True,
            checkpoint_step=1000,
            save_step=3000
        )


def debug_train(args):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|persona|>']})
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.resize_token_embeddings(len(tokenizer))

    train_dataloader, valid_dataloader = prepare_data(
        tokenizer,
        'only-valid',
        loss_type=args.loss,
        batch_size=args.bs,
        mask_context=args.mask_context
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    device = 'cuda'

    ul_training = True
    if args.loss == 'nll':
        ul_training = False
    answer_model = DialoGPTUnlikelihoodModel(model, tokenizer, device=device, ul_training=ul_training, parallel=args.parallel)
    trained_model = answer_model.train(
        train_dataloader,
        valid_dataloader,
        optimizer,
        checkpoint_step=500,
        save_step=2000,
        log_wandb=False,
        sample=True
    )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        debug_train(args)
    else:
        wandb.login()

        sweep_config = create_config(args)
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        wandb.agent(sweep_id, train_net)
