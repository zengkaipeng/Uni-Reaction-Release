import pandas
import random
import numpy as np
import torch
import os

from .Dataset import CNYieldDataset, AzYieldDataset, SelDataset


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


def load_az_yield(
    data_path, condition_type='pretrain', vol_type='min_as_one',
    temperature_scale=100, solvent_vol_scale=1
):
    train_set = load_az_yield_one(
        data_path=data_path, part='train', condition_type=condition_type,
        vol_type=vol_type, temperature_scale=temperature_scale,
        solvent_vol_scale=solvent_vol_scale
    )
    if os.path.exists(os.path.join(data_path, 'val.csv')):
        val_set = load_az_yield_one(
            data_path=data_path, part='val', condition_type=condition_type,
            vol_type=vol_type, temperature_scale=temperature_scale,
            solvent_vol_scale=solvent_vol_scale
        )
    else:
        val_set = load_az_yield_one(
            data_path=data_path, part='train', condition_type=condition_type,
            vol_type=vol_type, temperature_scale=temperature_scale,
            solvent_vol_scale=solvent_vol_scale
        )
    test_set = load_az_yield_one(
        data_path=data_path, part='test', condition_type=condition_type,
        vol_type=vol_type, temperature_scale=temperature_scale,
        solvent_vol_scale=solvent_vol_scale
    )
    return train_set, val_set, test_set


def load_az_yield_one(
    data_path, part, condition_type='pretrain', vol_type='min_as_one',
    temperature_scale=100, solvent_vol_scale=1
):
    train_x = pandas.read_csv(os.path.join(data_path, f'{part}.csv'))
    reac1, reac2, out, ligand, meta, base, solvent = [[] for _ in range(7)]
    prod, ligand_vol, meta_vol, base_vol, sol_vol = [[] for _ in range(5)]
    reac1_vol, reac2_vol, temperature = [], [], []
    for i, x in train_x.iterrows():
        prod.append(x['mapped_rxn'].split('>>')[1])
        reac1.append(x['Aryl_halide_maaped'])
        reac2.append(x['Amine_mapped'])
        if vol_type == 'min_as_one':
            vbase = min(x['Aryl_halide_amount'], x['Amine_amount'])
        elif vol_type == 'absolute':
            vbase = 1
        else:
            raise ValueError(f'Invalid vol type {vol_type}')
        reac1_vol.append(x['Aryl_halide_amount'] / vbase)
        reac2_vol.append(x['Amine_amount'] / vbase)
        out.append(x['Yield'])
        ligand.append(x['Ligand'])
        ligand_vol.append(x['Ligand_amount'])

        if pandas.isna(x['Metal']):
            meta.append('')
            meta_vol.append(0)
        else:
            meta.append(x['Metal'])
            meta_vol.append(x['Metal_amount'] / vbase)

        base.append(x['Base'])
        base_vol.append(x['Base_amount'] / vbase)

        if pandas.isna(x['Solvent']):
            solvent.append('')
            sol_vol.append(0)
        else:
            solvent.append(x['Solvent'])
            sol_vol.append(x['Solvent_amount'] / solvent_vol_scale)

        temperature.append(
            float('nan') if pandas.isna(x['Temperature'])
            else x['Temperature'] / temperature_scale
        )

    return AzYieldDataset(
        mapped_reac1=reac1, mapped_reac2=reac2, mapped_prod=prod,
        labels=out, base=base, solvent=solvent, meta=meta,
        ligand=ligand, temperature=temperature, reac1_vol=reac1_vol,
        reac2_vol=reac2_vol, base_vol=base_vol, solvent_vol=sol_vol,
        meta_vol=meta_vol, ligand_vol=ligand_vol, condition_type=condition_type
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
