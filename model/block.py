import torch


from .layers import (
    RAlignGATBlock, SelfLoopGATConv, graph2batch, SparseEdgeUpdateLayer,
    TransDecLayer, PositionalEncoding
)


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
