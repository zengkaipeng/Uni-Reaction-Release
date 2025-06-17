import pandas
import random
import numpy as np
import torch

from .Dataset import CNYieldDataset


def load_cn_yield(data_path, condition_type='pretrain'):
    train_set = load_cn_yield_one(data_path, 'train')
    val_set = load_cn_yield_one(data_path, 'val')
    test_set = load_cn_yield_one(data_path, 'test')
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
        additive=additive, labels=out,  condition_type='pretrain'
    )


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
