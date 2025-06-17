import torch

from ..utils import graph2batch
from .pretrain_gnns import PretrainGIN
from .SimpleGAT import SimpleCondGAT


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
            answer['catalyst and ligand'] = {
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


def build_cn_condition_encoder(config, dropout):
    if config['type'] == 'pretrain':
        dropout = config['arch'].get('drop_ratio', dropout)
        config['arch']['drop_ratio'] = dropout
        if config.get('pretrain_ckpt', '') != '':
            gnn = PretrainGIN(num_layer=5, emb_dim=300, drop_ratio=dropout)
            gnn.load_from_pretrained(config['pretrain_ckpt'])
            if config.get('freeze_gnn', False):
                gnn.requires_grad_(False)
            encoder = CNConditionEncoder(300, gnn, config['mode'])
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = CNConditionEncoder(config['dim'], gnn, config['mode'])
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = CNConditionEncoder(config['dim'], gnn, config['mode'])
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder
