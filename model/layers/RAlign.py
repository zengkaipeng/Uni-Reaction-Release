import torch
from .GATconv import SelfLoopGATConv
from .shared import SparseEdgeUpdateLayer
from .ConditionAdapter import ConditionAdaptor
from utils.tensor_utils import graph2batch


class RAlingLayer(torch.nn.Module):
    def __init__(self, dim, dropout=0):
        super(RAlingLayer, self).__init__()
        self.comm_lin = torch.nn.Sequential(
            torch.nn.Linear(dim + dim, dim + dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim + dim, dim + dim)
        )

        self.lg_lin = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim)
        )
        self.dim = dim

    def forward(self, x_prod, x_reac, reac_mask):
        new_reac = torch.zeros_like(x_reac)
        shared_result = torch.cat([x_prod, x_reac[reac_mask]], dim=-1)
        shared_result = self.comm_lin(shared_result)
        new_prod = shared_result[:, :self.dim]
        new_reac[reac_mask] = shared_result[:, self.dim:]

        if torch.any(~reac_mask).item():
            new_reac[~reac_mask] = self.lg_lin(x_reac[~reac_mask])

        return new_prod, new_reac


class RAlignGATBlock(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, edge_dim, reac_batch_infos={}, reac_num_keys={},
        prod_batch_infos={}, prod_num_keys={}, dropout=0.1,
        negative_slope=0.2, edge_update=True, fusion_order='ca_first'
    ):
        super(RAlignGATBlock, self).__init__()

        # Create condition adaptors for reactant and product graphs
        self.reac_condition_adaptor = ConditionAdaptor(
            emb_dim=emb_dim,
            batched_infos=reac_batch_infos,
            num_keys=reac_num_keys,
            dropout=dropout,
            fusion_order=fusion_order
        )

        self.prod_condition_adaptor = ConditionAdaptor(
            emb_dim=emb_dim,
            batched_infos=prod_batch_infos,
            num_keys=prod_num_keys,
            dropout=dropout,
            fusion_order=fusion_order
        )

        assert emb_dim % heads == 0, 'emb_dim must be divisible by heads'

        # GNN layers
        self.reac_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.prod_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )

        self.edge_update = edge_update

        # Cross-graph alignment layer
        self.fusion_layer = RAlingLayer(emb_dim, dropout)

        # LayerNorm layers
        self.reac_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.reac_fusion_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_fusion_ln = torch.nn.LayerNorm(emb_dim)

        # Edge update layers (if needed)
        if self.edge_update:
            self.reac_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.prod_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.reac_edge_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_edge_ln = torch.nn.LayerNorm(emb_dim)

        self.drop_f = torch.nn.Dropout(dropout)

    def forward(
        self, reac_x, reac_e, reac_eidx, reac_bmask, shared_mask,
        prod_x, prod_e, prod_eidx, prod_bmask,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        # 1. Message passing within each graph
        reac_conv = self.reac_mpnn(
            x=reac_x, edge_attr=reac_e, edge_index=reac_eidx
        )
        prod_conv = self.prod_mpnn(
            x=prod_x, edge_attr=prod_e, edge_index=prod_eidx
        )

        # 2. Residual connection and LayerNorm
        prod_x = self.prod_mpnn_ln(self.drop_f(prod_conv) + prod_x)
        reac_x = self.reac_mpnn_ln(self.drop_f(reac_conv) + reac_x)

        # 3. Cross-graph alignment
        prod_u, reac_u = self.fusion_layer(
            x_prod=prod_x, x_reac=reac_x, reac_mask=shared_mask
        )

        prod_x = self.prod_fusion_ln(self.drop_f(prod_u) + prod_x)
        reac_x = self.reac_fusion_ln(reac_x + self.drop_f(reac_u))

        # 4. Condition fusion using the new adaptor classes
        reac_x = self.reac_condition_adaptor(
            feats=reac_x,
            batch_mask=reac_bmask,
            batched_conditions=reac_batched_condition,
            num_conditions=reac_num_conditions
        )

        prod_x = self.prod_condition_adaptor(
            feats=prod_x,
            batch_mask=prod_bmask,
            batched_conditions=prod_batched_condition,
            num_conditions=prod_num_conditions
        )

        # 5. Edge update (if enabled)
        if self.edge_update:
            reac_e_u = self.reac_ue(
                edge_feats=reac_e, node_feats=reac_x, edge_index=reac_eidx
            )
            prod_e_u = self.prod_ue(
                edge_feats=prod_e, node_feats=prod_x, edge_index=prod_eidx
            )
            reac_e = self.reac_edge_ln(reac_e + self.drop_f(reac_e_u))
            prod_e = self.prod_edge_ln(prod_e + self.drop_f(prod_e_u))

        return reac_x, prod_x, reac_e, prod_e
