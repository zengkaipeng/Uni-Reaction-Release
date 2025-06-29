import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .layers import SelfLoopGATConv, SparseEdgeUpdateLayer

class DualMPNN(torch.nn.Module):
    def __init__(
        self, emb_dim: int, n_layer: int, dropout: float = 0,
        heads: int = 1, negative_slope: float = 0.2
    ):
        super(DualMPNN, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        if n_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = n_layer, heads
        self.dropout_fun = torch.nn.Dropout(dropout)
        assert emb_dim % heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(SelfLoopGATConv(
                in_channels=emb_dim, heads=heads,
                out_channels=emb_dim // heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=emb_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(emb_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                emb_dim, emb_dim, dropout=dropout
            ))
            self.lns.append(torch.nn.LayerNorm(emb_dim))

            self.convs.append(SelfLoopGATConv(
                in_channels=emb_dim, heads=heads,
                out_channels=emb_dim // heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=emb_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(emb_dim))
            self.edge_update.append(SparseEdgeUpdateLayer(
                emb_dim, emb_dim, dropout=dropout
            ))
            self.lns.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, reac_graph, prod_graph, return_attetnion_weights=False) -> torch.Tensor:
        reac_x = self.atom_encoder(reac_graph.x)
        prod_x = self.atom_encoder(prod_graph.x)
        reac_e = self.bond_encoder(reac_graph.edge_attr)
        prod_e = self.bond_encoder(prod_graph.edge_attr)

        sparse_edge_attn = []
        for i in range(self.num_layers):
            if return_attetnion_weights:
                conv_res, reac_attn = self.convs[i << 1](
                    x=reac_x, edge_attr=reac_e, edge_index=reac_graph.edge_index, 
                    return_attetnion_weights = True
                )
                conv_res = self.batch_norms[i << 1](conv_res)
            else:

                conv_res = self.batch_norms[i << 1](self.convs[i << 1](
                    x=reac_x, edge_attr=reac_e, edge_index=reac_graph.edge_index,
                ))

            reac_x = self.dropout_fun(torch.relu(conv_res)) + reac_x

            if return_attetnion_weights:
                conv_res, prod_attn = self.convs[i << 1 | 1](
                    x=prod_x, edge_attr=prod_e, edge_index=prod_graph.edge_index, 
                    return_attetnion_weights = True
                )
                conv_res = self.batch_norms[i << 1 | 1](conv_res)
                sparse_edge_attn.append((reac_attn, prod_attn))
            else:
                conv_res = self.batch_norms[i << 1 | 1](self.convs[i << 1 | 1](
                    x=prod_x, edge_attr=prod_e, edge_index=prod_graph.edge_index,
                ))
            prod_x = self.dropout_fun(torch.relu(conv_res)) + prod_x

            edge_res = self.lns[i << 1](self.edge_update[i << 1](
                edge_feats=reac_e, node_feats=reac_x,
                edge_index=reac_graph.edge_index
            ))

            reac_e = self.dropout_fun(torch.relu(edge_res)) + reac_e

            edge_res = self.lns[i << 1 | 1](self.edge_update[i << 1 | 1](
                edge_feats=prod_e, node_feats=prod_x,
                edge_index=prod_graph.edge_index
            ))

            prod_e = self.dropout_fun(torch.relu(edge_res)) + prod_e

        if return_attetnion_weights:
            return reac_x, prod_x, reac_e, prod_e, sparse_edge_attn
        else:
            return reac_x, prod_x, reac_e, prod_e
