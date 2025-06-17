import torch
import torch_geometric
import numpy as np
from numpy import concatenate as npcat

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
    node_ptr,  node_batch, lstnode, isprod = [0], [], 0, []
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
