import torch
import torch_geometric
import numpy as np
from numpy import concatenate as npcat
from rdkit import Chem

from .chemistry_parse import get_reaction_core
from .graph_utils import smiles2graph, pretrain_s2g


class RAlignDatasetBase(torch.utils.data.Dataset):
    def __init__(self, reactions):
        super(RAlignDatasetBase, self).__init__()
        self.reactions = reactions

    def __len__(self):
        return len(self.reactions)

    def get_aligned_graphs(self, index):
        reac, prod = self.reactions[index].strip().split('>>')
        reac_rcs, prod_rcs = get_reaction_core(reac, prod)

        reac_mol, reac_amap = smiles2graph(reac, with_amap=True)
        prod_mol, prod_amap = smiles2graph(prod, with_amap=True)

        # align the atoms so that atom with same idx
        # have the same atom map
        am2rank = {}
        for am, arank in prod_amap.items():
            am2rank[am] = arank

        for am in reac_amap:
            if am not in am2rank:
                am2rank[am] = len(am2rank)

        prod_mol['is_rc'] = [False] * len(prod_amap)
        reac_mol['is_rc'] = [False] * len(reac_amap)

        for k in reac_rcs:
            reac_mol['is_rc'][am2rank[k]] = True

        for k in prod_rcs:
            prod_mol['is_rc'][prod_amap[k]] = True

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

        isprod = [False] * reac_mol['num_nodes']
        for x in prod_amap:
            isprod[am2rank[x]] = True

        reac_mol['isprod'] = np.array(isprod, dtype=bool)
        return reac_mol, prod_mol

    def __getitem__(self, index):
        msg = 'the __getitem__ is not implemented for Base Dataset'
        raise NotImplementedError(msg)


class CNYieldDataset(RAlignDatasetBase):
    def __init__(
        self, reactions, ligand, catalyst, base, additive,
        labels, condition_type='pretrain'
    ):
        super(CNYieldDataset, self).__init__(reactions)
        self.ligand = ligand
        self.catalyst = catalyst
        self.base = base
        self.additive = additive
        self.labels = labels
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __getitem__(self, index):
        reac_mol, prod_mol = self.get_aligned_graphs(index)
        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        return reac_mol, prod_mol, gf(self.ligand[index]), \
            gf(self.base[index]), gf(self.additive[index]), \
            gf(self.catalyst[index]), self.labels[index]


def graph_col_fn(batch):
    batch_size, edge_idx, node_feat, edge_feat = len(batch), [], [], []
    node_ptr,  node_batch, lstnode, isprod, vols = [0], [], 0, [], []
    max_node, is_rc = max(x['num_nodes'] for x in batch), []
    batch_mask = torch.zeros(batch_size, max_node).bool()

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

    return torch_geometric.data.Data(**result)


def cn_colfn(batch):
    reac, prod, all_conditions, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        all_conditions.extend(x[2: -1])
        lbs.append(x[-1])

    return graph_col_fn(reac), graph_col_fn(prod), \
        graph_col_fn(all_conditions), torch.FloatTensor(lbs)


class AzYieldDataset(torch.utils.data.Dataset):
    def __init__(
        self, mapped_reac1, mapped_reac2, mapped_prod, labels, base, solvent,
        meta, ligand, temperature, reac1_vol, reac2_vol, base_vol,
        solvent_vol, meta_vol, ligand_vol, condition_type='pretrain'
    ):
        super(AzYieldDataset, self).__init__()
        self.mapped_reac1 = mapped_reac1
        self.mapped_reac2 = mapped_reac2
        self.labels = labels
        self.base = base
        self.solvent = solvent
        self.meta = meta
        self.ligand = ligand
        self.temperature = temperature
        self.reac1_vol = reac1_vol
        self.reac2_vol = reac2_vol
        self.base_vol = base_vol
        self.solvent_vol = solvent_vol
        self.meta_vol = meta_vol
        self.ligand_vol = ligand_vol
        self.mapped_prod = mapped_prod
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __len__(self):
        return len(self.labels)

    def process_rxn(self, reac1, reac2, prod, reac1_vol, reac2_vol):
        mol_reac1 = Chem.MolFromSmiles(reac1)
        mol_reac2 = Chem.MolFromSmiles(reac2)
        reac1_atom_num = mol_reac1.GetNumAtoms()
        reac2_atom_num = mol_reac2.GetNumAtoms()
        reac_vols_raw = [reac1_vol] * reac1_atom_num +\
            [reac2_vol] * reac2_atom_num

        reac = f'{reac1}.{reac2}'
        reac_rcs, prod_rcs = get_reaction_core(reac, prod)

        reac_mol, reac_amap = smiles2graph(reac, with_amap=True)
        prod_mol, prod_amap = smiles2graph(prod, with_amap=True)
        assert reac_mol['node_feat'].shape[0] == len(reac_vols_raw), \
            "The number of atoms mismatch when converting graph"

        # align the atoms so that atom with same idx
        # have the same atom map
        am2rank = {}
        for am, arank in prod_amap.items():
            am2rank[am] = arank

        for am in reac_amap:
            if am not in am2rank:
                am2rank[am] = len(am2rank)

        prod_mol['is_rc'] = [False] * len(prod_amap)
        reac_mol['is_rc'] = [False] * len(reac_amap)

        for k in reac_rcs:
            reac_mol['is_rc'][am2rank[k]] = True

        for k in prod_rcs:
            prod_mol['is_rc'][prod_amap[k]] = True

        remap = {v: am2rank[x] for x, v in reac_amap.items()}
        reac_x = np.zeros_like(reac_mol['node_feat'])
        reac_vols = np.zeros(len(reac_vols_raw), dtype=np.float32)
        reac_e = reac_mol['edge_index'].tolist()

        for k, v in remap.items():
            reac_x[v] = reac_mol['node_feat'][k]
            reac_vols[v] = reac_vols_raw[k]

        reac_e = [
            [remap[x] for x in reac_e[0]],
            [remap[x] for x in reac_e[1]]
        ]

        reac_mol['node_feat'] = reac_x
        reac_mol['volumn'] = reac_vols
        reac_mol['edge_index'] = np.array(reac_e, dtype=np.int64)

        isprod = [False] * reac_mol['num_nodes']
        for x in prod_amap:
            isprod[am2rank[x]] = True

        reac_mol['isprod'] = np.array(isprod, dtype=bool)
        return reac_mol, prod_mol

    def __getitem__(self, index):
        reac_mol, prod_mol = self.process_rxn(
            reac1=self.mapped_reac1[index], reac2=self.mapped_reac2[index],
            prod=self.mapped_prod[index], reac1_vol=self.reac1_vol[index],
            reac2_vol=self.reac2_vol[index]
        )

        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        return (
            reac_mol, prod_mol, gf(self.meta[index]), gf(self.ligand[index]),
            gf(self.solvent[index]), gf(self.base[index]),
            self.meta_vol[index], self.ligand_vol[index],
            self.solvent_vol[index], self.base_vol[index],
            self.temperature[index], self.labels[index]
        )


def az_colfn(batch):
    reac, prod, all_conditions, lbs, temp = [], [], [], [], []
    key_to_nums = {k: [] for k in ['meta', 'ligand', 'solvent', 'base']}
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        all_conditions.extend(x[2: 6])
        lbs.append(x[-1])
        key_to_nums['meta'].append(x[6])
        key_to_nums['ligand'].append(x[7])
        key_to_nums['solvent'].append(x[8])
        key_to_nums['base'].append(x[9])
        temp.append(x[-2])

    key_to_nums = {k: torch.FloatTensor(v) for k, v in key_to_nums.items()}

    return (
        graph_col_fn(reac), graph_col_fn(prod), graph_col_fn(all_conditions),
        key_to_nums, torch.FloatTensor(temp), torch.FloatTensor(lbs)
    )


class SelDataset(RAlignDatasetBase):
    def __init__(
        self, reactions, labels, catalyst=None, condition_type='pretrain'
    ):
        super(SelDataset, self).__init__(reactions)
        self.catalyst = catalyst
        self.labels = labels
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        if self.catalyst is None:
            return reac_mol, prod_mol, self.labels[idx]
        else:
            return reac_mol, prod_mol, gf(self.catalyst[idx]), self.labels[idx]


def sel_with_cat_colfn(batch):
    reac, prod, catalyst, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        lbs.append(x[-1])
        catalyst.append(x[2])

    return graph_col_fn(reac), graph_col_fn(prod), \
        graph_col_fn(catalyst), torch.FloatTensor(lbs)


def sel_wo_cat_colfn(batch):
    reac, prod, catalyst, lbs = [], [], [], []
    for x in batch:
        reac.append(x[0])
        prod.append(x[1])
        lbs.append(x[2])

    return graph_col_fn(reac), graph_col_fn(prod), torch.FloatTensor(lbs)


class SMYieldDataset(RAlignDatasetBase):
    def __init__(
        self, reactions, ligand, catalyst, solvent,
        labels, condition_type='pretrain'
    ):
        super(SMYieldDataset, self).__init__(reactions)
        self.catalyst = catalyst
        self.ligand = ligand
        self.solvent = solvent
        self.labels = labels
        self.condition_type = condition_type

        assert condition_type in ['pretrain', 'raw'], \
            f'Invalid condition type {condition_type}'

    def __getitem__(self, index):
        reac_mol, prod_mol = self.get_aligned_graphs(index)
        gf = smiles2graph if self.condition_type == 'raw' else pretrain_s2g
        return reac_mol, prod_mol, \
            gf(self.ligand[index]), gf(self.catalyst[index]), \
            gf(self.solvent[index]), self.labels[index]


class ReactionPredDataset(RAlignDatasetBase):
    def __init__(self, reactions, labels, cls_id, end_id=None):
        super(ReactionPredDataset, self).__init__(reactions)
        self.labels = labels
        self.cls_id = cls_id
        self.end_id = end_id

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)

        tlabel = [self.cls_id] + self.labels[idx]
        if self.end_id is not None:
            tlabel += [self.end_id]
        return reac_mol, prod_mol, tlabel


def gen_fn(batch):
    reac = graph_col_fn([x[0] for x in batch])
    prod = graph_col_fn([x[1] for x in batch])
    return reac, prod, [x[2] for x in batch]


def pred_fn(batch):
    reac = graph_col_fn([x[0] for x in batch])
    prod = graph_col_fn([x[1] for x in batch])
    return reac, prod, torch.LongTensor([x[2] for x in batch])


class ReactionSeqInferenceDataset(RAlignDatasetBase):
    def __init__(self, reactions, labels=None, return_raw=True):
        super(ReactionSeqInferenceDataset, self).__init__(reactions)
        self.labels = labels
        self.return_raw = return_raw

    def __getitem__(self, idx):
        reac_mol, prod_mol = self.get_aligned_graphs(idx)
        out_ans = [reac_mol, prod_mol]
        if self.return_raw:
            out_ans.append(self.reactions[idx])
        if self.labels is not None:
            out_ans.append(self.labels[idx])


def gen_inf_fn(batch):
    out_ans = [
        graph_col_fn([x[0] for x in batch]),
        graph_col_fn([x[1] for x in batch])
    ]
    if len(batch[0]) > 2:
        out_ans.append([x[2] for x in batch])
    if len(batch[0]) > 3:
        out_ans.append([x[3] for x in batch])


def pred_inf_fn(batch):
    out_ans = [
        graph_col_fn([x[0] for x in batch]),
        graph_col_fn([x[1] for x in batch])
    ]
    if len(batch[0]) > 2:
        if isinstance(batch[0][2], str):
            out_ans.append([x[2] for x in batch])
        else:
            out_ans.append(torch.LongTensor([x[2] for x in batch]))
    if len(batch[0]) > 3:
        out_ans.append(torch.LongTensor([x[3] for x in batch]))

    return out_ans
