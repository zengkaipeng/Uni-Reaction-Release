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
        self.n_dim = n_dim

    def foward(self, num_x):
        out = self.net(num_x.unsqueeze(dim=-1))
        if self.training and self.noisy_training:
            out = out + torch.randn_like(out)
        return out


class NumEmbeddingWithNan(NumEmbedding):
    def __init__(self, n_cls, n_dim, noisy_training=False):
        super(NumEmbeddingWithNan, self).__init__(n_cls, n_dim, noisy_training)
        self.nan_embedding = torch.nn.Parameter(torch.randn(n_dim))

    def forward(self, num_x):
        empty_mask, bs = torch.isnan(x), num_x.shape[0]
        out_result = torch.zeros((bs, self.n_dim)).to(num_x)
        out_result[~empty_mask] = self.net(num_x[~empty_mask])
        out_result[empty_mask] = self.nan_embedding
        return out_result


class AzConditionEncoder(torch.nn.Module):
    def __init__(
        self, gnn, gnn_dim, num_emb_dim,  use_sol_vol=False,
        use_vol=False, use_temp=False, merge_mode='independent'
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
        self.merge_mode = merge_mode
        assert self.merge_mode in [
            'independent', 'mix-meta-ligand', 'mix-all'
        ], f"Invalid merge mode {self.merge_mode}"

    def forward(
        self, shared_graph, key_to_volumn_feats={}, temperatures_feats=None
    ):
        nfeat = self.gnn(shared_graph)
        nfeat = graph2batch(nfeat, shared_graph.batch_mask)

        key_list = ['meta', 'ligand', 'solvent', 'base']

        if self.use_temp:
            assert temperatures_feats is not None, "Require temperature input"
            gamma = self.temp_adapter_gamma(temperatures_feats)[:, None]
            beta = self.temp_adapter_beta(temperatures_feats)[:, None]
            temp_bias = gamma * nfeat + beta
        else:
            temp_bias = torch.zeros_like(nfeat)

        answer = {}
        for idx, key in enumerate(key_list):
            if self.use_vol and key != 'solvent':
                assert key_to_volumn is not None, "Require Volumn input"
                gamma = self.volumn_adapter_gamma(key_to_volumn_feats[key])
                beta = self.volumn_adapter_beta(key_to_volumn_feats[key])
                bias = gamma[:, None] * nfeat[idx::4] + beta[:, None]
            elif self.use_sol_vol and key == 'solvent':
                assert key_to_volumn is not None, "Require Volumn input"
                gamma = self.sol_adapter_gamma(key_to_volumn_feats[key])
                beta = self.sol_adapter_beta(key_to_volumn_feats[key])
                bias = gamma[:, None] * nfeat[idx::4] + beta[:, None]
            else:
                bias = torch.zeros_like(nfeat[idx::4])

            this_emb = nfeat[idx::4] + bias + temp_bias[idx::4]
            this_mask = shared_graph.batch_mask[idx::4]

            answer[key] = {'embedding': this_emb, 'meaningful_mask': this_mask}

        if self.merge_mode == 'mix-meta-ligand':
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
        elif self.merge_mode == 'mix-all':
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
            if torch.any(this_empty).item():
                v['meaningful_mask'] = torch.cat([
                    v['meaningful_mask'], this_empty.reshape(-1, 1)
                ], dim=1)
                v['embedding'] = torch.cat([
                    v['embedding'],
                    self.empty_mol.repeat(this_empty.shape[0], 1, 1)
                ], dim=1)
            v['padding_mask'] = torch.logical_not(v['meaningful_mask'])

        return answer


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
            if torch.any(this_empty).item():
                v['meaningful_mask'] = torch.cat([
                    v['meaningful_mask'], this_empty.reshape(-1, 1)
                ], dim=1)
                v['embedding'] = torch.cat([
                    v['embedding'],
                    self.empty_mol.repeat(this_empty.shape[0], 1, 1)
                ], dim=1)
            v['padding_mask'] = torch.logical_not(v['meaningful_mask'])

        return answer


def build_cn_condition_encoder(config, dropout):
    if config['type'] == 'pretrain':
        dropout = config['arch'].get('drop_ratio', dropout)
        config['arch']['drop_ratio'] = dropout
        if config.get('pretrain_ckpt', '') != '':
            gnn = PretrainGIN(num_layer=5, emb_dim=300, drop_ratio=dropout)
            gnn.load_from_pretrained(config['pretrain_ckpt'])
            freeze_mode = config.get('freeze_mode', 'none')
            if freeze_mode.startswith('freeze'):
                freeze_layer = int(freeze_mode.split('-')[1])
                assert freeze_layer < 5, \
                    "last layer norm changed, finetune required"
                gnn.requires_grad_(False)
                for x in range(freeze_layer):
                    gnn.batch_norms[x].eval()

                for x in range(freeze_layer, 5):
                    gnn.batch_norms[x].requires_grad_(True)
                    gnn.gnns[x].requires_grad_(True)
            else:
                assert freeze_mode == 'none', \
                    f"Invalid freeze mode {freeze_mode}"
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


def build_az_condition_encoder(
    config, dropout, use_temperature, use_sol_vol, use_vol, num_emb_dim
):
    if config['type'] == 'pretrain':
        dropout = config['arch'].get('drop_ratio', dropout)
        config['arch']['drop_ratio'] = dropout
        if config.get('pretrain_ckpt', '') != '':
            gnn = PretrainGIN(num_layer=5, emb_dim=300, drop_ratio=dropout)
            gnn.load_from_pretrained(config['pretrain_ckpt'])
            freeze_mode = config.get('freeze_mode', 'none')
            if freeze_mode.startwith('freeze'):
                freeze_layer = int(freeze_mode.split('-')[1])
                assert freeze_layer < 5, \
                    "last layer norm changed, finetune required"
                gnn.requires_grad_(False)
                for x in range(freeze_layer):
                    gnn.batch_norms[x].eval()

                for x in range(freeze_layer, 5):
                    gnn.batch_norms[x].requires_grad_(True)
                    gnn.gnns[x].requires_grad_(True)
            else:
                assert freeze_mode == 'none', \
                    f"Invalid freeze mode {freeze_mode}"
            encoder = AzConditionEncoder(
                gnn, gnn_dim=300, num_emb_dim=num_emb_dim,
                use_sol_vol=use_sol_vol, use_vol=use_vol,
                use_temp=use_temperature, merge_mode=config['mode']
            )
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = AzConditionEncoder(
                gnn, gnn_dim=config['dim'], num_emb_dim=num_emb_dim,
                use_sol_vol=use_sol_vol, use_vol=use_vol,
                use_temp=use_temperature, merge_mode=config['mode']
            )
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = AzConditionEncoder(
            gnn, gnn_dim=config['dim'], num_emb_dim=num_emb_dim,
            use_sol_vol=use_sol_vol, use_vol=use_vol,
            use_temp=use_temperature, merge_mode=config['mode']
        )
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder


class DMConditionEncoder(torch.nn.Module):
    def __init__(self, gnn_dim, gnn, mode='default'):
        super(DMConditionEncoder, self).__init__()
        self.gnn, self.mode = gnn, mode
        self.empty_mol = torch.nn.Parameter(torch.randn(gnn_dim))
        assert mode in ['default'],\
            "Invalid condition output mode"

    def forward(self, shared_gnn):
        node_feat = self.gnn(shared_gnn)
        node_feat = graph2batch(node_feat, shared_gnn.batch_mask)
        answer = {
            'catalyst': {
                'embedding': node_feat,
                'meaningful_mask': shared_gnn.batch_mask
            }
        }

        for k, v in answer.items():
            this_empty = ~torch.any(v['meaningful_mask'], dim=1)
            if torch.any(this_empty).item():
                v['meaningful_mask'][this_empty, 0] = True
                v['embedding'][this_empty, 0] = self.empty_mol
            v['padding_mask'] = torch.logical_not(v['meaningful_mask'])

        return answer


def build_dm_condition_encoder(config, dropout):
    if config['type'] == 'pretrain':
        dropout = config['arch'].get('drop_ratio', dropout)
        config['arch']['drop_ratio'] = dropout
        if config.get('pretrain_ckpt', '') != '':
            gnn = PretrainGIN(num_layer=5, emb_dim=300, drop_ratio=dropout)
            gnn.load_from_pretrained(config['pretrain_ckpt'])
            freeze_mode = config.get('freeze_mode', 'none')
            if freeze_mode.startswith('freeze'):
                freeze_layer = int(freeze_mode.split('-')[1])
                assert freeze_layer < 5, \
                    "last layer norm changed, finetune required"
                gnn.requires_grad_(False)
                for x in range(freeze_layer):
                    gnn.batch_norms[x].eval()

                for x in range(freeze_layer, 5):
                    gnn.batch_norms[x].requires_grad_(True)
                    gnn.gnns[x].requires_grad_(True)
            else:
                assert freeze_mode == 'none', \
                    f"Invalid freeze mode {freeze_mode}"
            encoder = DMConditionEncoder(300, gnn, config['mode'])
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = DMConditionEncoder(config['dim'], gnn, config['mode'])
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = DMConditionEncoder(config['dim'], gnn, config['mode'])
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder
