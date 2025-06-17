import os
import logging

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6
# including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) \
            + self.edge_embedding2(edge_attr[:, 1])

        out = self.propagate(
            edge_index=edge_index, x=x,
            edge_attr=edge_embeddings, aggr=self.aggr
        )
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class PretrainGIN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node/graph representations

    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0):
        super(PretrainGIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.batch_norms.append(torch.nn.LayerNorm(emb_dim))

        self.out_dim = emb_dim

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h = F.dropout(
                h if layer == self.num_layer - 1 else F.relu(h),
                self.drop_ratio, training=self.training
            )
            h_list.append(h)

        # Different implementations of Jk-concat
        node_representation = h_list[-1]
        return node_representation

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        sd = {}
        for k, v in state_dict.items():
            k_ = k.replace("gnn.", "")
            sd[k_] = v

        msg = self.load_state_dict(sd, strict=False)

        logging.info("Loading info: {}".format(msg))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
