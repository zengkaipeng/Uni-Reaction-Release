import torch

from ..utils import graph2batch
from .pretrain_gnns import PretrainGIN
from .SimpleGAT import SimpleCondGAT
from functools import partial


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


def build_cn_condition_encoder(config, dropout, condition_encoder=CNConditionEncoder):
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
            encoder = condition_encoder(300, gnn, config['mode'])
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = condition_encoder(config['dim'], gnn, config['mode'])
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = condition_encoder(config['dim'], gnn, config['mode'])
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder


def build_cn_condition_encoder_with_eval(config, dropout):
    eval_mode_layers = []
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
                    eval_mode_layers.append(gnn.batch_norms[x])

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

    return encoder, eval_mode_layers


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


class SMConditionEncoder(torch.nn.Module):
    def __init__(self, gnn_dim, gnn, mode='mix-all'):
        super(SMConditionEncoder, self).__init__()
        self.gnn, self.mode = gnn, mode
        self.empty_mol = torch.nn.Parameter(torch.randn(gnn_dim))
        assert mode in ['mix-all', 'mix-catalyst-ligand', 'independent'],\
            "Invalid condition output mode"

    def forward(self, shared_gnn):
        node_feat = self.gnn(shared_gnn)
        node_feat = graph2batch(node_feat, shared_gnn.batch_mask)
        key_list = ['ligand', 'catalyst', 'solvent']
        answer = {
            key: {
                'embedding': node_feat[idx::3],
                'meaningful_mask': shared_gnn.batch_mask[idx::3]
            } for idx, key in enumerate(key_list)
        }

        if self.mode == 'mix-catalyst-ligand':
            answer['catalyst and ligand'] = {
                'embedding': torch.cat([
                    answer['catalyst']['embedding'],
                    answer['ligand']['embedding']
                ], dim=1),
                'meaningful_mask': torch.cat([
                    answer['ligand']['meaningful_mask'],
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


def build_basic_condition_encoder(config, dropout, EncoderClass):
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
            encoder = EncoderClass(300, gnn, config['mode'])
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = EncoderClass(config['dim'], gnn, config['mode'])
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = EncoderClass(config['dim'], gnn, config['mode'])
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder


build_sm_condition_encoder = partial(
    build_basic_condition_encoder, EncoderClass=SMConditionEncoder)
# build_cn_condition_encoder = partial(build_basic_condition_encoder, EncoderClass=CNConditionEncoder)
