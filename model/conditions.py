import os
import logging

import torch
import torch.nn.functional as F

from .layers import SelfLoopGATConv, PretrainGINConv, graph2batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class SimpleCondGAT(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, num_heads: int = 4,
        embedding_dim: int = 64, dropout: float = 0.7,
        negative_slope: float = 0.2
    ):
        super(SimpleCondGAT, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.edge_update = torch.nn.ModuleList()
        self.num_layers, self.num_heads = num_layers, num_heads
        self.drop_f = torch.nn.Dropout(dropout)
        assert embedding_dim % num_heads == 0, \
            'The embedding dim should be evenly divided by num_heads'
        for layer in range(self.num_layers):
            self.convs.append(SelfLoopGATConv(
                in_channels=embedding_dim, heads=num_heads,
                out_channels=embedding_dim // num_heads,
                negative_slope=negative_slope,
                dropout=dropout, edge_dim=embedding_dim
            ))
            self.batch_norms.append(torch.nn.LayerNorm(embedding_dim))
            self.lns.append(torch.nn.LayerNorm(embedding_dim))
            if layer < self.num_layers - 1:
                self.edge_update.append(SparseEdgeUpdateLayer(
                    embedding_dim, embedding_dim, dropout=dropout
                ))
        self.atom_encoder = AtomEncoder(embedding_dim)
        self.bond_encoder = BondEncoder(embedding_dim)

    def forward(self, G) -> torch.Tensor:
        node_feats = self.atom_encoder(G.x)
        edge_feats = self.bond_encoder(G.edge_attr)
        for layer in range(self.num_layers):
            conv_res = self.batch_norms[layer](self.convs[layer](
                x=node_feats, edge_attr=edge_feats, edge_index=G.edge_index,
            ))
            node_feats = self.drop_f(torch.relu(conv_res)) + node_feats

            if layer < self.num_layers - 1:
                edge_res = self.lns[layer](self.edge_update[layer](
                    edge_feats=edge_feats, node_feats=node_feats,
                    edge_index=G.edge_index
                ))
                edge_feats = self.drop_f(torch.relu(edge_res)) + edge_feats

        return node_feats


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
            self.gnns.append(PretrainGINConv(emb_dim, aggr="add"))

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
                h if layer == self.num_layer - 1 else F.relu(F),
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


class NumEmbedding(torch.nn.Module):
    def __init__(self, n_cls, n_dim, noisy_training=False):
        super(NumEmbedding, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, n_cls),
            torch.nn.Softmax(dim=-1),
            torch.nn.Linear(n_cls, n_dim)
        )
        self.noisy_training = noisy_training

    def foward(self, num_x):
        out = self.net(num_x.unsqueeze(dim=-1))
        if self.training and self.noisy_training:
            out = out + torch.randn_like(out)
        return out


class AzConditionEncoder(torch.nn.Module):
    def __init__(
        self, gnn, gnn_dim, num_emb_dim, use_temp=False,
        use_sol_vol=False, use_vol=False
    ):
        super(AzConditionEncoder, self).__init__()
        if use_sol_vol:
            self.sol_adapter_gamma = torch.nn.Linear(num_emb_dim, gnn_dim)
            self.sol_adapter_beta = torch.nn.Linear(num_emb_dim, gnn_dim)
        if use_vol:
            self.volumn_adapter_gamma = torch.nn.Linear(num_emb_dim, gnn_dim)
            self.volumn_adapter_beta = torch.nn.Linear(num_emb_dim, gnn_dim)
        if use_temp:
            self.temp_adapter_gamma = torch.nn.Linear(num_emb_dim, gnn_dim)
            self.temp_adapter_beta = torch.nn.Linear(num_emb_dim, gnn_dim)

        self.gnn = gnn
        self.use_sol_vol = use_sol_vol
        self.use_vol = use_vol
        self.use_temp = use_temp
        self.empty_mol = torch.nn.Parameter(torch.randn(gnn_dim))

    def forward(
        self, shared_graph, key_to_volumn_feats=None,
        merge_meta_ligand=True, temperatures_feats=None
    ):
        nfeat = self.gnn(shared_graph)
        nfeat = graph2batch(nfeat, shared_graph.batch_mask)

        key_list = []
        raise NotImplementedError('Key list not filled')

        if self.use_temp:
            assert temperatures is not None, "Require temperature input"
            temp_emb = self.temperature_encoder(temperatures)

            temp_bias = self.temp_adapter_gamma(temp_emb).unsqueeze(dim=1) *\
                nfeat + self.temp_adapter_beta(temp_emb).unsqueeze(dim=1)
        else:
            temp_bias, temp_emb = torch.zeros_like(nfeat), None

        answer, xg = {}, len(key_list)
        for idx, key in enumerate(key_list):
            if self.use_vol and key != 'solvent':
                assert key_to_volumn is not None, "Require Volumn input"
                assert not torch.any(torch.isnan(key_to_volumn[key])),\
                    "Nan value in volumn infos"
                emb = self.volumn_encoder(key_to_volumn[key])  # [bs, dim]
                bias = (
                    self.volumn_adapter_gamma(emb).unsqueeze(dim=1) *
                    nfeat[idx::xg] +
                    self.volumn_adapter_beta(emb).unsqueeze(dim=1)
                )
            elif self.use_sol_vol and key == 'solvent':
                assert key_to_volumn is not None, "Require Volumn input"
                assert not torch.any(torch.isnan(key_to_volumn[key])),\
                    "Nan value in solvent volumn infos"
                emb = self.sol_volumn_encoder(key_to_volumn[key])
                bias = (
                    self.sol_adapter_gamma(emb).unsqueeze(dim=1) *
                    nfeat[idx::xg] +
                    self.sol_adapter_beta(emb).unsqueeze(dim=1)
                )
            else:
                bias = torch.zeros_like(nfeat[idx::xg])

            this_emb = nfeat[idx::xg] + bias + temp_bias[idx::xg]
            this_mask = shared_graph.batch_mask[idx::xg]

            answer[key] = {'embedding': this_emb, 'meaningful_mask': this_mask}

        if merge_meta_ligand:
            answer['meta_and_ligand'] = {
                'embedding': torch.cat([
                    answer['meta']['embedding'],
                    answer['ligand']['embedding']
                ], dim=1),
                'meaningful_mask': torch.cat([
                    answer['meta']['meaningful_mask'],
                    answer['ligand']['meaningful_mask']
                ], dim=1)
            }
            del answer['meta']
            del answer['ligand']

        for k, v in answer.items():
            this_empty = ~torch.any(v['meaningful_mask'], dim=1)
            v['meaningful_mask'][this_empty, 0] = True
            v['embedding'][this_empty, 0] = self.empty_mol
            v['padding_mask'] = torch.logical_not(v['meaningful_mask'])

        return answer, {'temperature': temp_emb}


class CNConditionEncoder(torch.nn.Module):
    def __init__(self, gnn_dim, gnn, mode='mix-all'):
        super(CNConditionEncoder, self).__init__()
        self.gnn, self.mode = gnn, mode
        self.empty_mol = torch.nn.Parameter(torch.randn(gnn_dim))
        assert mode in ['mix-all', 'mix-catalyst-ligand', 'independent'],\
            "Invalid condition output mode"

    def forward(self, shared_gnn):
        node_feat = self.gnn(shared_gnn)
        node_feat = graph2batch(node_feat, shared_gnn.batch_mask)
        key_list = ['ligand', 'base', 'additive', 'catalyst']
        answer = {
            key: {
                'embedding': node_feat[idx::4],
                'meaningful_mask': shared_gnn.batch_mask[idx::4]
            } for idx, key in enumerate(key_list)
        }

        if self.mode == 'mix-catalyst-ligand':
            answer['catalyst-ligand'] = {
                'embedding': torch.cat([
                    answer['catalyst']['embedding'],
                    answer['ligand']['embedding']
                ], dim=1),
                'meaningful_mask': torch.cat([
                    answer['catalyst']['meaningful_mask'],
                    answer['ligand']['meaningful_mask']
                ], dim=1)
            }
            del answer['catalyst']
            del answer['ligand']
        elif self.mode == 'mix-all':
            all_emb = [answer[k]['embedding'] for k in key_list]
            all_mask = [answer[k]['meaningful_mask'] for k in key_list]
            answer = {
                'mixed': {
                    'embedding': torch.cat(all_emb, dim=1),
                    'meaningful_mask': torch.cat(all_mask, dim=1)
                }
            }

        for k, v in answer.items():
            this_empty = ~torch.any(v['meaningful_mask'], dim=1)
            v['meaningful_mask'][this_empty, 0] = True
            v['embedding'][this_empty, 0] = self.empty_mol
            v['padding_mask'] = torch.logical_not(v['meaningful_mask'])

        return answer
