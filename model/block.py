import torch

from .layers import RAlignGATBlock, DualGATBlock, TransDecLayer
from .utils import graph2batch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class TranDec(torch.nn.Module):
    def __init__(
        self, n_layers, emb_dim, heads, dropout=0,
        kvdim=None, dim_ff=None
    ):
        super(TranDec, self).__init__()
        self.layers = torch.nn.ModuleList([
            TransDecLayer(emb_dim, heads, dropout, kvdim, dim_ff)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        for i in range(self.n_layers):
            tgt = self.layers[i](
                tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return tgt


class RAlignEncoder(torch.nn.Module):
    def __init__(
        self, n_layer, emb_dim, heads, edge_dim, reac_batch_infos={},
        reac_num_keys={}, prod_batch_infos={}, prod_num_keys={},
        dropout=0.1, negative_slope=0.2, update_last_edge=False
    ):
        super(RAlignEncoder, self).__init__()
        self.n_layers = n_layer
        self.layers = torch.nn.ModuleList()
        for i in range(n_layer):
            update_edge = (i < n_layer - 1) or update_last_edge
            self.layers.append(RAlignGATBlock(
                emb_dim=emb_dim, heads=heads, edge_dim=edge_dim,
                reac_batch_infos=reac_batch_infos, reac_num_keys=reac_num_keys,
                prod_batch_infos=prod_batch_infos, prod_num_keys=prod_num_keys,
                negative_slope=negative_slope, dropout=dropout,
                edge_update=update_edge
            ))

        self.update_last_edge = update_last_edge

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(
        self, reac_graph, prod_graph,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        reac_x = self.atom_encoder(reac_graph.x)
        prod_x = self.atom_encoder(prod_graph.x)
        reac_e = self.bond_encoder(reac_graph.edge_attr)
        prod_e = self.bond_encoder(prod_graph.edge_attr)
        for i in range(self.n_layers):
            reac_x, prod_x, reac_e, prod_e = self.layers[i](
                reac_x=reac_x, reac_e=reac_e,
                reac_eidx=reac_graph.edge_index,
                reac_bmask=reac_graph.batch_mask,
                shared_mask=reac_graph.is_prod,
                reac_batched_condition=reac_batched_condition,
                reac_num_conditions=reac_num_conditions,
                prod_x=prod_x, prod_e=prod_e,
                prod_eidx=prod_graph.edge_index,
                prod_bmask=prod_graph.batch_mask,
                prod_batched_condition=prod_batched_condition,
                prod_num_conditions=prod_num_conditions
            )

        return reac_x, prod_x, reac_e, prod_e
