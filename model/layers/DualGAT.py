import torch
from .GATconv import SelfLoopGATConv
from .shared import SparseEdgeUpdateLayer, graph2batch


class DualGATBlock(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, edge_dim, reac_batch_infos={}, reac_num_keys={},
        prod_batch_infos={}, prod_num_keys={}, condtion_heads=None,
        dropout=0.1, negative_slope=0.2, edge_update=True
    ):
        super(DualGATBlock, self).__init__()
        condition_heads = heads if condtion_heads is None else condtion_heads
        self.reac_batch_adapter = torch.nn.ModuleDict({
            k: torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=condition_heads,
                batch_first=True, dropout=dropout, kdim=v, vdim=v
            ) for k, v in reac_batch_infos.items()
        })
        self.prod_batch_adapter = torch.nn.ModuleDict({
            k: torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=condition_heads,
                batch_first=True, dropout=dropout, kdim=v, vdim=v
            ) for k, v in prod_batch_infos.items()
        })
        self.reac_num_adapter = torch.nn.ModuleDict({
            k: torch.nn.ModuleDict(
                {
                    'beta': torch.nn.Linear(v, emb_dim),
                    'gamma': torch.nn.Linear(v, emb_dim)
                }
            ) for k, v in reac_num_keys.items()
        })
        self.prod_num_adapter = torch.nn.ModuleDict({
            k: torch.nn.ModuleDict(
                {
                    'beta': torch.nn.Linear(v, emb_dim),
                    'gamma': torch.nn.Linear(v, emb_dim)
                }
            ) for k, v in prod_num_keys.items()
        })
        assert emb_dim % heads == 0, 'emb_dim must be divisible by heads'
        self.reac_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )
        self.prod_mpnn = SelfLoopGATConv(
            in_channels=emb_dim, out_channels=emb_dim // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, negative_slope=negative_slope
        )

        self.edge_update = edge_update
        self.reac_mpnn_ln = torch.nn.LayerNorm(emb_dim)
        self.prod_mpnn_ln = torch.nn.LayerNorm(emb_dim)

        if self.edge_update:
            self.reac_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.prod_ue = SparseEdgeUpdateLayer(edge_dim, emb_dim, dropout)
            self.reac_edge_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_edge_ln = torch.nn.LayerNorm(emb_dim)

        if len(batch_infos) > 0:
            self.reac_cond_ln = torch.nn.LayerNorm(emb_dim)
            self.prod_cond_ln = torch.nn.LayerNorm(emb_dim)
        else:
            self.reac_cond_ln = self.prod_cond_ln = None

        self.drop_f = torch.nn.Dropout(dropout)

    def forward(
        self, reac_x, reac_e, reac_eidx, reac_bmask, shared_mask,
        prod_x, prod_e, prod_eidx, prod_bmask,
        reac_batched_condition={}, reac_num_conditions={},
        prod_batched_condition={}, prod_num_conditions={}
    ):
        reac_conv = self.reac_mpnn_ln(self.reac_mpnn(
            x=reac_x, edge_attr=reac_e, edge_index=reac_eidx
        ))

        prod_conv = self.prod_mpnn_ln(self.prod_mpnn(
            x=prod_x, edge_attr=prod_e, edge_index=prod_eidx
        ))

        prod_x = self.drop_f(torch.relu(prod_conv)) + prod_x
        reac_x = self.drop_f(torch.relu(reac_conv)) + reac_x

        reac_x = graph2batch(reac_x, reac_bmask)
        prod_x = graph2batch(prod_x, prod_bmask)

        reac_bias = torch.zeros_like(reac_x)
        prod_bias = torch.zeros_like(prod_x)

        for k, v in self.reac_batch_adapter.items():
            this_info = reac_batched_condition[k]
            bias, _w = v(
                query=reac_x, key=this_info['embedding'],
                value=this_info['embedding'],
                key_padding_mask=this_info.get('padding_mask', None)
            )
            reac_bias += self.drop_f(bias)

        for k, v in self.prod_batch_adapter.items():
            this_info = prod_batched_condition[k]
            bias, _w = v(
                query=prod_x, key=this_info['embedding'],
                value=this_info['embedding'],
                key_padding_mask=this_info.get('padding_mask', None)
            )
            prod_bias += self.drop_f(bias)

        if self.prod_cond_ln is not None:
            prod_x = self.prod_cond_ln(prod_x + prod_bias)
            reac_x = self.reac_cond_ln(reac_bias + reac_x)

        reac_bias = torch.zeros_like(reac_x)
        prod_bias = torch.zeros_like(prod_x)

        for k, v in self.reac_num_adapter.items():
            gamma = v['gamma'](reac_num_conditions[k])
            beta = v['beta'](reac_num_conditions[k])
            reac_bias += gamma * reac_x + beta

        for k, v in self.prod_num_adapter.items():
            gamma = v['gamma'](prod_num_conditions[k])
            beta = v['beta'](prod_num_conditions[k])
            prod_bias += gamma * prod_x + beta

        prod_x = (prod_x + prod_bias)[reac_bmask]
        reac_x = (reac_x + reac_bias)[prod_bmask]

        if self.edge_update:
            reac_e_u = self.reac_edge_ln(self.reac_ue(
                edge_feats=reac_e, node_feats=reac_x, edge_index=reac_eidx
            ))
            prod_e_u = self.prod_edge_ln(self.prod_ue(
                edge_feats=prod_e, node_feats=prod_x, edge_index=prod_eidx
            ))
            reac_e = reac_e + self.drop_f(torch.relu(reac_e_u))
            prod_e = prod_e + self.drop_f(torch.relu(prod_e_u))

        return reac_x, prod_x, reac_e, prod_e
