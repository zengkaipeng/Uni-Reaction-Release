import pandas
import random
import numpy as np
import torch
import os
import json
from tqdm import tqdm

from .Dataset import (
    CNYieldDataset, SelDataset, ReactionPredDataset,
    ReactionSeqInferenceDataset
)

from .tokenlizer import Tokenizer, smi_tokenizer


def load_sel(data_path, condition_type='pretrain', has_reag=True):
    train_set = load_sel_one(data_path, 'train', condition_type, has_reag)
    val_set = load_sel_one(data_path, 'val', condition_type, has_reag)
    test_set = load_sel_one(data_path, 'test', condition_type, has_reag)
    return train_set, val_set, test_set


def load_sel_one(data_path, part, condition_type='pretrain', has_reag=True):
    train_x = pandas.read_csv(os.path.join(data_path, f'{part}.csv'))
    rxn, out, catalyst = [[] for _ in range(3)]
    for i, x in train_x.iterrows():
        rxn.append(x['mapped_rxn'])
        out.append(x['Output'])
        if has_reag:
            catalyst.append(x['Catalyst'])

    return SelDataset(
        reactions=rxn, catalyst=catalyst if len(catalyst) else None,
        labels=out, condition_type=condition_type
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


def load_uspto_mt500_inference(data_path, remap):
    rxns, labels = [], []
    with open(data_path) as F:
        setx = json.load(F)
        for lin in setx:
            rxns.append(lin['new_mapped_rxn'])
            labels.append('.'.join(lin['reagent_list']))

    dataset = ReactionSeqInferenceDataset(rxns, labels, True)
    return dataset


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def load_uspto_condition(data_path, mapper_path='', verbose=True, mapper=None):
    raw_info = pandas.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')

    if mapper is None:
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

    return train_set, val_set, test_set, mapper


def load_uspto_condition_inference(data_path, mapper):
    raw_info = pandas.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')
    reac, all_labels = [], []

    for i, element in enumerate(tqdm(raw_info)):
        if element['dataset'] != 'test':
            continue
        reac.append(element['mapped_rxn'])
        labels = [
            mapper[element['catalyst1']],
            mapper[element['solvent1']], mapper[element['solvent2']],
            mapper[element['reagent1']], mapper[element['reagent2']]
        ]
        all_labels.append(labels)

    dataset = ReactionSeqInferenceDataset(reac, all_labels, True)
    return dataset


def load_uspto_yield(data_path, part='all'):
    """
    加载JSONL格式的数据，并创建USPTOYieldDataset实例

    Args:
        data_path: str, JSONL文件的路径
        part: str, 可以是 'train', 'val', 'test', 'all'
            如果是 'all'，则返回一个字典，包含三个部分的数据集
            否则返回指定部分的数据集

    Returns:
        如果 part == 'all': 返回字典 {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        否则: 返回对应的USPTOYieldDataset实例
    """

    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 读取所有数据
    all_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    all_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue

    if not all_data:
        raise ValueError(f"No valid data found in {data_path}")

    # 根据part参数筛选数据
    if part.lower() == 'all':
        # 分别筛选train/val/test数据
        train_data = \
            [item for item in all_data if item.get('dataset') == 'train']
        val_data = [item for item in all_data if item.get('dataset') == 'val']
        test_data = \
            [item for item in all_data if item.get('dataset') == 'test']

        # 创建对应的数据集
        datasets = {}
        for part_name, data in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            if data:  # 如果该部分有数据
                reactants, products, reagents, temperatures, yields = \
                    _prepare_data_uspto_yield(data)
                datasets[part_name] = USPTOYieldDataset(
                    reactants=reactants,
                    products=products,
                    reagents=reagents,
                    temperatures=temperatures,
                    yields=yields
                )
            else:
                print(f"Warning: No {part_name} data found")
                datasets[part_name] = None

        return datasets

    else:
        # 筛选指定部分的数据
        part_lower = part.lower()
        if part_lower not in ['train', 'val', 'test']:
            raise ValueError(
                f"Invalid part value: {part}. "
                f"Must be 'train', 'val', 'test', or 'all'"
            )

        filtered_data = [item for item in all_data if item.get(
            'dataset') == part_lower]

        if not filtered_data:
            raise ValueError(f"No {part} data found in {data_path}")

        # 准备数据
        reactants, products, reagents, temperatures, yields = \
            _prepare_data_uspto_yield(filtered_data)

        # 创建数据集
        dataset = USPTOYieldDataset(
            reactants=reactants,
            products=products,
            reagents=reagents,
            temperatures=temperatures,
            yields=yields
        )

        return dataset


def _prepare_data_uspto_yield(data_list):
    """
    将原始数据转换为USPTOYieldDataset需要的格式

    Args:
        data_list: list of dict, 原始数据列表

    Returns:
        reactants: list of list, 每个内层list是[(smiles, vol, amount), ...]
        products: list of str, 产物的mapped_smiles
        reagents: list of list, 每个内层list是[(smiles, vol, amount), ...]
        temperatures: list of float/None, 温度值
    """
    reactants = []
    products = []
    reagents = []
    temperatures = []
    yields = []

    for item in data_list:
        # 提取reactants
        sample_reactants = []
        for r in item.get('reactants', []):
            smiles = r['smiles']
            # 处理可能的None值
            amount = r.get('amount')
            volume = r.get('volume')
            sample_reactants.append((smiles, volume, amount))
        reactants.append(sample_reactants)

        # 提取product
        products.append(item['product'])

        # 提取reagents
        sample_reagents = []
        for r in item.get('reagents', []):
            smiles = r['smiles']
            # 处理可能的None值
            amount = r.get('amount')
            volume = r.get('volume')
            sample_reagents.append((smiles, volume, amount))
        reagents.append(sample_reagents)

        # 提取temperature
        temperatures.append(item.get('temperature'))
        yields.append(item['yield'])

    return reactants, products, reagents, temperatures, yields
