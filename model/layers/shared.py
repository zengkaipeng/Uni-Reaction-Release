import torch


class SparseEdgeUpdateLayer(torch.nn.Module):
    def __init__(
        self, edge_dim: int = 64, node_dim: int = 64,
        dropout: float = 0
    ):
        super(SparseEdgeUpdateLayer, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, edge_dim)
        )

    def forward(
        self, node_feats: torch.Tensor, edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        node_i = node_feats[edge_index[0]]
        node_j = node_feats[edge_index[1]]
        x = torch.cat([node_i, node_j, edge_feats], dim=-1)
        return self.mlp(x)


def graph2batch(
    node_feat: torch.Tensor, batch_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_node = batch_mask.shape
    answer = torch.zeros(batch_size, max_node, node_feat.shape[-1])
    answer = answer.to(node_feat.device)
    answer[batch_mask] = node_feat
    return answer


class DotMhAttn(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, Odim, emb_dim,
        num_heads, dropout=0.0,
    ):
        super(DotMhAttn, self).__init__()
        self.heads = num_heads
        assert emb_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by heads'
        self.Qproj = torch.nn.Linear(Qdim, emb_dim)
        self.Kproj = torch.nn.Linear(Kdim, emb_dim)
        self.Vproj = torch.nn.Linear(Vdim, emb_dim)
        self.Oproj = torch.nn.Linear(emb_dim, Odim)
        self.drop_fun = torch.nn.Dropout(dropout)
        self.temp = math.sqrt(emb_dim / num_heads)
        self.xdim = emb_dim // self.heads

    def forward(
        self, query, key, value, attn_mask=None,
        key_padding_mask=None
    ):
        (BS, Q_len), K_len = query.shape[:2], key.shape[1]
        Qp = self.Qproj(query).reshape(BS, Q_len, self.xdim, self.heads)
        Kp = self.Kproj(key).reshape(BS, K_len, self.xdim, self.heads)
        Vp = self.Vproj(value).reshape(BS, K_len, self.xdim, self.heads)
        attn_w = torch.einsum('abcd,aecd->aebd', Qp, Kp) / self.temp

        # [BS, key_len, query_len, dim]

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (BS, K_len), \
                'The key padding mask should have shape (BS, Key)'
            attn_w[key_padding_mask] = 1 - (1 << 32)

        if attn_mask is not None:
            if attn_mask.ndim == 4:
                assert attn_mask.shape == (BS, Q_len, K_len, self.heads),\
                    "The attn mask should be (BS, Query, Key, heads)"
                attn_w[torch.transpose(attn_mask, 1, 2)] = 1 - (1 << 32)
            else:
                assert attn_mask.shape == (Q_len, K_len),\
                    "The attn mask should be two dim (Query, Key) or " + \
                    "four dim (BS, Query, Key, heads)"
                attn_w[:, torch.transpose(attn_mask, 0, 1)] = 1 - (1 << 32)

        attn_w = self.drop_fun(torch.softmax(attn_w, dim=1))
        attn_o = torch.einsum('acbd,aced->abed', attn_w, Vp)
        attn_o = self.Oproj(attn_o.reshape(BS, Q_len, -1))

        # return shape:
        # attn_o: [BS, query_len, dim]
        # attn_w: [BS, query_len, key_len, dim]
        return attn_o, attn_w.transpose(1, 2)


