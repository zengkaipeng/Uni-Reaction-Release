import torch
from .layers import DotMhAttn
from .utils import graph2batch


class CNYieldModel(torch.nn.Module):
    def __init__(self, encoder, condition_encoder, emb_dim, dropout=0.1):
        super(CNYieldModel, self).__init__()
        self.encoder = encoder
        self.condition_encoder = condition_encoder
        self.out_head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 2)
        )
        self.pool_keys = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.pooler = DotMhAttn(
            Qdim=dim, Kdim=dim, Vdim=dim, Odim=dim,
            emb_dim=dim, num_heads=heads, dropout=dropout
        )

    def forward(self, reac_graph, prod_graph, conditions, cross_mask=None):
        condition_dict = self.condition_encoder(conditions)
        reac_x, prod_x, _, _ = self.encoder(
            reac_graph=reac_graph, reac_batched_condition=condition_dict,
            prod_graph=prod_graph, prod_batched_condition=condition_dict
        )

        x_reac = graph2batch(x_reac, reac_graph.batch_mask)
        x_prod = graph2batch(x_prod, prod_graph.batch_mask)
        memory = torch.cat([x_reac, x_prod], dim=1)
        memory_mask = [reac_graph.batch_mask, prod_graph.batch_mask]
        memory_mask = torch.logical_not(torch.cat(memory_mask, dim=1))

        pool_key = self.pool_keys.repeat(memory.shape[0], 1, 1)
        pooled_results, p_attn = self.pooler(
            query=pool_key, key=memory, value=memory,
            key_padding_mask=memory_pad, attn_mask=cross_mask
        )
        reaction_emb = self.xln(pooled_results.squeeze(dim=1))
        return self.out_head(reaction_emb)
