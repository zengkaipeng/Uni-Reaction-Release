import torch
from .shared import DotMhAttn


class TransDecLayer(torch.nn.Module):
    def __init__(
        self, emb_dim, heads, dropout=0, kvdim=None, dim_ff=None
    ):
        super(TransDecLayer, self).__init__()

        self.emb_dim, self.heads = emb_dim, heads
        self.kvdim = emb_dim if kvdim is None else kvdim
        self.dim_ff = emb_dim * 2 if dim_ff is None else dim_ff

        assert self.emb_dim % heads == 0, \
            'The emb dim should be evenly divided by heads'

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.dim_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.dim_ff, self.emb_dim)
        )
        self.dropfun1 = torch.nn.Dropout(dropout)
        self.dropfun2 = torch.nn.Dropout(dropout)
        self.dropfun3 = torch.nn.Dropout(dropout)

        self.sa = DotMhAttn(
            Qdim=self.emb_dim, Kdim=self.emb_dim, Vdim=self.emb_dim,
            Odim=self.emb_dim, emb_dim=self.emb_dim,
            num_heads=self.heads, dropout=dropout
        )

        self.ca = DotMhAttn(
            Qdim=self.emb_dim, Kdim=self.kvdim, Vdim=self.kvdim,
            Odim=self.emb_dim, emb_dim=self.emb_dim,
            num_heads=self.heads, dropout=dropout
        )

        self.ln1 = torch.nn.LayerNorm(self.emb_dim)
        self.ln2 = torch.nn.LayerNorm(self.emb_dim)
        self.ln3 = torch.nn.LayerNorm(self.emb_dim)

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None,
    ):
        x = self.ln1(tgt + self.dropfun1(self.sa(
            query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]))
        x = self.ln2(x + self.dropfun2(self.ca(
            query=x, key=memory, value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]))

        x = self.ln3(x + self.dropfun3(self.ffn(x)))
        return x
