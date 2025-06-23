import pandas
import random
import numpy as np
import torch
import os

from .Dataset import CNYieldDataset, SelDataset

def load_sel(data_path, condition_type='pretrain'):
    train_set = load_sel_one(data_path, 'train', condition_type)
    val_set = load_sel_one(data_path, 'val', condition_type)
    test_set = load_sel_one(data_path, 'test', condition_type)
    return train_set, val_set, test_set


def load_sel_one(data_path, part, condition_type='pretrain'):
    train_x = pandas.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, ligand, base, additive, catalyst = [[] for _ in range(6)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['Output'])
        catalyst.append(x['Catalyst'])

    return SelDataset(
        reactions=rxn, catalyst=catalyst, labels=out,
        condition_type=condition_type
    )

def load_cn_yield(data_path, condition_type='pretrain'):
    train_set = load_cn_yield_one(data_path, 'train', condition_type)
    val_set = load_cn_yield_one(data_path, 'val', condition_type)
    test_set = load_cn_yield_one(data_path, 'test', condition_type)
    return train_set, val_set, test_set


def load_cn_yield_one(data_path, part, condition_type='pretrain'):
    train_x = pandas.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, ligand, base, additive, catalyst = [[] for _ in range(6)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['Output'])
        ligand.append(x['Ligand'])
        base.append(x['Base'])
        additive.append(x['Additive'])
        catalyst.append(x['catalyst'])

    return CNYieldDataset(
        reactions=rxn, ligand=ligand, catalyst=catalyst, base=base,
        additive=additive, labels=out,  condition_type=condition_type
    )


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module):
    total_params = 0
    trainable_params = 0
    for param in module.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params
