import torch
import torch_geometric
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
        reac, prod = self.reactions[idx].strip().split('>>')
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

    def __getitem__(self, index):
        msg = 'the __getitem__ is not implemented for Base Dataset'
        raise NotImplementedError(msg)


class CNYieldDataset(RAlignDatasetBase):
	def __init__(self, reactions, ligand, catalyst, )