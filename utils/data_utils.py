import pandas
import random
import numpy as np
import torch
import os
import json

from .Dataset import (
    CNYieldDataset, AzYieldDataset, SMYieldDataset,
    SelDataset, ReactionPredDataset
)

from .tokenlizer import Tokenizer, smi_tokenizer


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


def load_sm_yield(data_path, condition_type='pretrain'):
    train_set = load_sm_yield_one(data_path, 'train', condition_type)
    if os.path.exists(os.path.join(data_path, 'val.csv')):
        val_set = load_sm_yield_one(data_path, 'val', condition_type)
    else:
        val_set = load_sm_yield_one(data_path, 'train', condition_type)

    test_set = load_sm_yield_one(data_path, 'test', condition_type)
    return train_set, val_set, test_set


def load_sm_yield_one(data_path, part, condition_type='pretrain'):
    train_x = pandas.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, ligand, solvent, catalyst = [[] for _ in range(5)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['y'])
        ligand.append(
            x['ligand_smiles'] if not pandas.isna(x['ligand_smiles']) else ''
        )
        solvent.append(
            x['solvent_smiles'] if not pandas.isna(x['solvent_smiles']) else ''
        )
        catalyst.append(
            x['catalyst_smiles'] if not pandas.isna(x['catalyst_smiles'])
            else ''
        )

    return SMYieldDataset(
        reactions=rxn, ligand=ligand, catalyst=catalyst,
        solvent=solvent, labels=out,  condition_type=condition_type
    )


def load_uspto_mt_500_gen(data_path, remap=None, part=None):
    if remap is None:
        with open(os.path.join(data_path, 'all_tokens.json')) as F:
            reag_list = json.load(F)
        remap = Tokenizer(reag_list, {'<UNK>', '<CLS>', '<END>', '<PAD>', '`'})

    with open(os.path.join(data_path, 'all_reagents.json')) as F:
        INFO = json.load(F)
    reag_order = {k: idx for idx, k in enumerate(INFO)}

    rxns, px = [[], [], []], 0
    labels = [[], [], []]
    if part is None:
        iterx = ['train.json', 'val.json', 'test.json']
    else:
        iterx = [f'{part}.json']
    for infos in iterx:
        F = open(os.path.join(data_path, infos))
        setx = json.load(F)
        F.close()
        for lin in setx:
            rxns[px].append(lin['new_mapped_rxn'])
            lin['reagent_list'].sort(key=lambda x: reag_order[x])
            lbs = []
            for tdx, x in enumerate(lin['reagent_list']):
                if tdx > 0:
                    lbs.append('`')
                lbs.extend(smi_tokenizer(x))
            labels[px].append(lbs)
        px += 1

    if part is not None:
        return ReactionPredDataset(
            reactions=rxns[0], labels=labels[0],
            cls_id='<CLS>', end_id='<END>'
        ), remap

    train_set = ReactionPredDataset(
        reactions=rxns[0], labels=labels[0],
        cls_id='<CLS>', end_id='<END>'
    )

    val_set = ReactionPredDataset(
        reactions=rxns[1], labels=labels[1],
        cls_id='<CLS>', end_id='<END>'
    )

    test_set = ReactionPredDataset(
        reactions=rxns[2], labels=labels[2],
        cls_id='<CLS>', end_id="<END>"
    )

    return train_set, val_set, test_set, remap


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def load_uspto_condition(data_path, mapper_path, verbose=True):
    raw_info = pandas.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')
    with open(mapper_path) as Fin:
        mapper = json.load(Fin)

    mapper['<CLS>'] = mapper.get('<CLS>', len(mapper))

    all_datas = {
        'train_reac': [], 'train_label': [],
        'val_reac': [], 'val_label': [],
        'test_reac': [], 'test_label': []
    }

    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        rxn_type = element['dataset']
        all_datas[f'{rxn_type}_reac'].append(element['mapped_rxn'])
        labels = [
            mapper[element['catalyst1']],
            mapper[element['solvent1']], mapper[element['solvent2']],
            mapper[element['reagent1']], mapper[element['reagent2']]
        ]
        all_datas[f'{rxn_type}_label'].append(labels)

    train_set = ReactionPredDataset(
        reactions=all_datas['train_reac'],
        labels=all_datas['train_label'], cls_id=mapper['<CLS>'],

    )
    val_set = ReactionPredDataset(
        reactions=all_datas['val_reac'],
        labels=all_datas['val_label'], cls_id=mapper['<CLS>']
    )

    test_set = ReactionPredDataset(
        reactions=all_datas['test_reac'],
        labels=all_datas['test_label'], cls_id=mapper["<CLS>"]
    )

    return train_set, val_set, test_set
