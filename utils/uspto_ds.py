import torch
import torch_geometric
import numpy as np
from numpy import concatenate as npcat
from rdkit import Chem

from .chemistry_parse import get_reaction_core
from .graph_utils import smiles2graph


class USPTOYieldDataset(torch.utils.data.Dataset):
    def __init__(self, reactants, products, yields, reagents=None, temperatures=None):
        """
        Args:
            reactants: list of list, 外层list长度是样本数，内层list元素是(mapped_smiles, vol, amount)
            products: list of str, mapped smiles
            reagents: list of list, 外层list长度是样本数，内层list元素是(smiles, vol, amount)
            temperatures: list of float/None, 反应温度
        """
        super(USPTOYieldDataset, self).__init__()
        self.reactants = reactants
        self.products = products
        self.yields = yields
        self.reagents = reagents if reagents is not None\
            else [[] for _ in range(len(reactants))]
        self.temperatures = temperatures if temperatures is not None\
            else [None] * len(reactants)

        # 检查数据长度一致性
        assert len(reactants) == len(products), \
            "Reactants and products must have same length"
        assert len(reactants) == len(self.reagents), \
            "Reactants and reagents must have same length"
        assert len(reactants) == len(self.temperatures), \
            "Reactants and temperatures must have same length"

    def __len__(self):
        return len(self.reactants)

    def get_all_am_and_check(self, smiles):
        """检查smiles中所有原子是否有atom mapping并获取所有atom map numbers"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid Smiles {smiles}')

        all_ams = set()
        for atm in mol.GetAtoms():
            if not atm.HasProp('molAtomMapNumber'):
                return False, set()
            all_ams.add(atm.GetAtomMapNum())

        return True, all_ams

    def get_aligned_graphs(self, index):
        """获取对齐的反应物和产物图"""
        reactants = self.reactants[index]  # list of (mapped_smiles, vol, amount)
        prod_smiles = self.products[index]  # str

        # 1. 检查每个反应物的原子映射并收集所有atom map numbers
        all_reactant_ams = set()
        am2vol = {}  # atom_map_number -> volume
        am2amount = {}  # atom_map_number -> amount
        reactant_smiles_list = []

        for smiles, vol, amount in reactants:
            # 检查原子映射
            has_am, ams = self.get_all_am_and_check(smiles)
            if not has_am:
                raise ValueError(
                    f'Reactant {smiles} has atoms without atom mapping'
                )

            # 检查重复的atom map numbers
            for am in ams:
                if am in all_reactant_ams:
                    raise ValueError(
                        f'Duplicated Atom Mapping found in {smiles}, '
                        f'curr_ams {all_reactant_ams}, curr smiles '
                        f'{[x[0] for x in reactants]}, {prod_smiles}'
                    )
                all_reactant_ams.add(am)
                am2vol[am] = vol
                am2amount[am] = amount

            reactant_smiles_list.append(smiles)

        # 2. 处理产物
        reac_smiles = '.'.join(reactant_smiles_list)

        # 获取反应核心
        reac_rcs, prod_rcs = get_reaction_core(reac_smiles, prod_smiles)

        # 3. 生成反应物和产物的图
        reac_mol, reac_amap = smiles2graph(reac_smiles, with_amap=True)
        prod_mol, prod_amap = smiles2graph(prod_smiles, with_amap=True)

        # 4. 对齐原子，使相同atom map的原子有相同的索引
        am2rank = {}
        for am, arank in prod_amap.items():
            am2rank[am] = arank

        for am in reac_amap:
            if am not in am2rank:
                am2rank[am] = len(am2rank)

        # 检查是否有重复的atom map numbers
        if len(am2rank) != len(all_reactant_ams):
            raise ValueError(f'Duplicated Atom Mapping found in {reac_smiles}')

        # 5. 标记反应核心
        prod_mol['is_rc'] = [False] * len(prod_amap)
        reac_mol['is_rc'] = [False] * len(reac_amap)

        for k in reac_rcs:
            if k in am2rank:
                reac_mol['is_rc'][am2rank[k]] = True

        for k in prod_rcs:
            if k in prod_amap:
                prod_mol['is_rc'][prod_amap[k]] = True

        # 6. 重排反应物图中的节点
        remap = {v: am2rank[x] for x, v in reac_amap.items()}
        reac_x = np.zeros_like(reac_mol['node_feat'])
        reac_e = reac_mol['edge_index'].tolist()

        for k, v in remap.items():
            reac_x[v] = reac_mol['node_feat'][k]

        reac_e = [
            [remap[x] for x in reac_e[0]],
            [remap[x] for x in reac_e[1]]
        ]

        reac_mol['node_feat'] = reac_x
        reac_mol['edge_index'] = np.array(reac_e, dtype=np.int64)

        # 7. 标记产物原子
        isprod = [False] * reac_mol['num_nodes']
        for x in prod_amap:
            isprod[am2rank[x]] = True

        reac_mol['isprod'] = np.array(isprod, dtype=bool)

        # 8. 添加体积和物质的量信息到反应物图
        n_atoms = reac_mol['num_nodes']

        # 初始化体积数组和未知标记
        vol_array = np.full(n_atoms, float('-inf'), dtype=np.float32)
        vol_unk = np.zeros(n_atoms, dtype=bool)

        # 初始化物质的量数组和未知标记
        amount_array = np.full(n_atoms, float('-inf'), dtype=np.float32)
        amount_unk = np.zeros(n_atoms, dtype=bool)

        for am in am2vol:
            if am in am2rank:
                idx = am2rank[am]

                # 处理体积
                vol = am2vol[am]
                if vol is None:
                    vol_array[idx] = float('-inf')
                    vol_unk[idx] = True
                else:
                    vol_array[idx] = vol
                    vol_unk[idx] = False

                # 处理物质的量
                amount = am2amount[am]
                if amount is None:
                    amount_array[idx] = float('-inf')
                    amount_unk[idx] = True
                else:
                    amount_array[idx] = amount
                    amount_unk[idx] = False

        reac_mol['volumn'] = vol_array
        reac_mol['vol_unk'] = vol_unk
        reac_mol['amount'] = amount_array
        reac_mol['amount_unk'] = amount_unk

        return reac_mol, prod_mol

    def __getitem__(self, index):
        """获取一个数据样本"""
        # 1. 获取对齐的反应物和产物图
        reac_mol, prod_mol = self.get_aligned_graphs(index)

        # 2. 处理试剂
        reagents_data = self.reagents[index]
        reags_graphs = []

        for smiles, vol, amount in reagents_data:
            if smiles:  # 如果有试剂
                reagent_graph = smiles2graph(smiles, with_amap=False)
                n_atoms = reagent_graph['num_nodes']

                # 添加体积信息
                if vol is None:
                    vol_array = np.full(
                        n_atoms, float('-inf'), dtype=np.float32)
                    vol_unk = np.ones(n_atoms, dtype=bool)
                else:
                    vol_array = np.full(n_atoms, vol, dtype=np.float32)
                    vol_unk = np.zeros(n_atoms, dtype=bool)

                # 添加物质的量信息
                if amount is None:
                    amount_array = np.full(
                        n_atoms, float('-inf'), dtype=np.float32)
                    amount_unk = np.ones(n_atoms, dtype=bool)
                else:
                    amount_array = np.full(n_atoms, amount, dtype=np.float32)
                    amount_unk = np.zeros(n_atoms, dtype=bool)

                reagent_graph['volumn'] = vol_array
                reagent_graph['vol_unk'] = vol_unk
                reagent_graph['amount'] = amount_array
                reagent_graph['amount_unk'] = amount_unk

                reags_graphs.append(reagent_graph)

        # 3. 获取温度
        temperature = self.temperatures[index]
        label = self.yields[index]

        return reac_mol, prod_mol, reags_graphs, temperature, label


def graph_col_fn_ds(batch):
    batch_size, edge_idx, node_feat, edge_feat = len(batch), [], [], []
    node_ptr,  node_batch, lstnode, isprod = [0], [], 0, []
    max_node, is_rc = max(x['num_nodes'] for x in batch), []
    batch_mask = torch.zeros(batch_size, max_node).bool()
    amts, amt_unks, vols, vol_unks = [], [], [], []

    for idx, gp in enumerate(batch):
        node_cnt = gp['num_nodes']
        if node_cnt == 0:
            node_ptr.append(lstnode)
            continue

        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)

        if 'is_rc' in gp:
            is_rc.append(torch.Tensor(gp['is_rc']).bool())
        if 'isprod' in gp:
            isprod.append(gp['isprod'])
        if 'volumn' in gp:
            vols.append(gp['volumn'])
            vol_unks.append(gp['vol_unk'])
        if 'amount' in gp:
            amts.append(gp['amount'])
            amt_unks.append(gp['amount_unk'])

        batch_mask[idx, :node_cnt] = True
        lstnode += node_cnt
        node_batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        node_ptr.append(lstnode)

    result = {
        'x': torch.from_numpy(npcat(node_feat, axis=0)),
        "edge_attr": torch.from_numpy(npcat(edge_feat, axis=0)),
        'ptr': torch.LongTensor(node_ptr),
        'batch': torch.from_numpy(npcat(node_batch, axis=0)),
        'edge_index': torch.from_numpy(npcat(edge_idx, axis=-1)),
        'num_nodes': lstnode,
        'batch_mask': batch_mask
    }

    if len(is_rc) > 0:
        result['is_rc'] = torch.cat(is_rc, dim=0)

    if len(isprod) > 0:
        result['is_prod'] = torch.from_numpy(npcat(isprod, axis=0))

    if len(vols) > 0:
        result['volumn'] = torch.from_numpy(npcat(vols, axis=0))
        result['vol_unk'] = torch.from_numpy(npcat(vol_unks, axis=0))

    if len(amts) > 0:
        result['amount'] = torch.from_numpy(npcat(amts, axis=0))
        result['amount_unk'] = torch.from_numpy(npcat(amt_unks, axis=0))

    return torch_geometric.data.Data(**result)


def ds_col_fn(batch):
    reac_mol = graph_col_fn_ds([x[0] for x in batch])
    prod_mol = graph_col_fn_ds([x[1] for x in batch])
    reag_x = [sum(t['num_nodes'] for t in x[2]) for x in batch]
    all_reag, all_temps, temp_unk, group_ids = [], [], [], []
    group_mask = torch.zeros((len(batch), max(reag_x)), dtype=torch.bool)

    for idx, x in enumerate(batch):
        all_reag.extend(x[2])
        group_mask[idx, :reag_x[idx]] = True
        group_ids.extend([idx] * reag_x[idx])
        all_temps.append(x[3] if x[3] is not None else float('-inf'))
        temp_unk.append(x[3] is None)

    reag_mol = graph_col_fn_ds(all_reag)
    temperature_tensor = torch.FloatTensor(all_temps)
    temperature_unk = torch.BoolTensor(temp_unk)
    reag_mol.group_id = torch.LongTensor(group_ids)
    reag_mol.group_mask = group_mask
    label = torch.FloatTensor([x[4] for x in batch])

    return reac_mol, prod_mol, reag_mol, temperature_tensor, temperature_unk, label
