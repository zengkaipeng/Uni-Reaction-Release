import torch


from typing import Literal, Dict, List, Optional
from utils.tensor_utils import graph2batch
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

    def forward(self, num_x):
        out = self.net(num_x.unsqueeze(dim=-1))
        if self.training and self.noisy_training:
            out = out + torch.randn_like(out)
        return out


class MultiNumEmb(torch.nn.Module):
    def __init__(
        self, emb_dim, num_cls, allow_unk=False, noisy_training=False
    ):
        super(MultiNumEmb, self).__init__()
        self.all_module = torch.nn.ModuleList([
            NumEmbedding(x, emb_dim, noisy_training) for x in num_cls
        ])
        self.allow_unk = allow_unk
        self.emb_dim = emb_dim
        if allow_unk:
            self.unk_emb = torch.nn.Parameter(torch.randn(emb_dim))

    def forward(self, *args):
        assert all(t[1].shape == args[0][1].shape for t in args), \
            "The shapes are not all the same"
        pre_dim = args[0][1].ndim
        all_mask = torch.zeros_like(args[0][1])
        if self.allow_unk:
            out_cls = self.unk_emb.reshape([1] * pre_dim + [-1])
            out_cls = out_cls.repeat(*args[0][1].shape, 1)
        else:
            out_cls = torch.zeros((*args[0][1].shape, self.emb_dim))
            out_cls = out_cls.to(args[0][0])

        for idx, (num, msk) in enumerate(args):
            alivex = torch.logical_not(msk)
            if torch.any(alivex & all_mask).item():
                raise ValueError('A pos have multiple attrs')
            all_mask |= alivex
            out_cls[alivex] = self.all_module[idx](num[alivex])

        if not self.allow_unk and not torch.all(all_mask).item():
            raise ValueError('A pos is not filled with no unk emb set')

        return out_cls


class ReagentEncoderWithAmount(torch.nn.Module):
    def __init__(
        self,
        gnn_dim: int,
        heads: int,
        n_layer: int,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        amount_encoder: Optional[torch.nn.Module] = None,
        num_emb_dim: Optional[int] = None,
        use_amount: bool = True,
        merge_mode: Literal['sep', 'mix'] = 'sep'
    ):
        super(ReagentEncoderWithAmount, self).__init__()
        if use_amount:
            assert amount_encoder is not None, 'Require Amount Encoder'
            assert num_emb_dim is not None, \
                'Require the hidden size of amount encoder'
            self.amount_adapter = torch.nn.Linear(num_emb_dim, gnn_dim * 2)
            self.amount_encoder = amount_encoder

        self.gnn = SimpleCondGAT(
            num_layers=n_layer, num_heads=heads, embedding_dim=gnn_dim,
            dropout=dropout, negative_slope=negative_slope
        )
        self.empty_mol = torch.nn.Parameter(torch.randn(1, 1, gnn_dim))
        self.merge_mode = merge_mode
        self.use_amount = use_amount

    def forward(self, reagents):
        node_feats = self.gnn(reagents)
        if self.use_amount:
            amount_emb = self.amount_encoder(
                (reagents.amount, reagents.amount_unk),
                (reagents.volumn, reagents.vol_unk)
            )
            cond_para = self.amount_adapter(amount_emb)
            gamma, beta = torch.chunk(cond_para, 2, dim=-1)
            node_feats = node_feats * gamma + beta

        if self.merge_mode == 'sep':
            empty_line = ~torch.any(reagents.group_mask, dim=1)
            x_n = graph2batch(node_feats, reagents.batch_mask)
            group_id = reagents.group_id[reagents.ptr[:-1]]
            assert group_id.shape[0] == x_n.shape[0], "Group Id mismatch"
            if torch.any(empty_line).item():
                tlist = torch.arange(reagents.group_mask.shape[0])
                req_tdx = tlist.to(x_n.device)[empty_line]
                to_p = torch.zeros((req_tdx.shape[0], *x_n.shape[1:])).to(x_n)
                to_p[:, 0] = self.empty_mol
                to_m = torch.zeros(
                    (req_tdx.shape[0], x_n.shape[1]),
                    dtype=torch.bool, device=x_n.device
                )
                to_m[:, 0] = True
                uf_mask = torch.cat([reagents.batch_mask, to_m], dim=0)
                group_id = torch.cat([group_id, req_tdx], dim=0)
                embedding = torch.cat([x_n, to_p], dim=0)
                padding_mask = torch.logical_not(uf_mask)
            else:
                padding_mask = torch.logical_not(reagents.batch_mask)
                embedding = x_n
            return {
                'shared': {
                    'padding_mask': padding_mask,
                    'embedding': embedding,
                    "group_id": group_id
                }
            }
        elif self.merge_mode == 'mix':
            empty_line = ~torch.any(reagents.group_mask, dim=1)
            x_n = graph2batch(node_feats, reagents.group_mask)
            if torch.any(empty_line).item():
                to_pad = self.empty_mol.repeat(empty_line.shape[0], 1, 1)
                padding_mask = torch.logical_not(torch.cat([
                    empty_line.unsqueeze(dim=-1), reagents.group_mask
                ], dim=1))
                embedding = torch.cat([to_pad, x_n], dim=1)
            else:
                padding_mask = torch.logical_not(reagents.group_mask)
                embedding = x_n
            return {
                'shared': {
                    'padding_mask': padding_mask,
                    'embedding': embedding
                }
            }
        else:
            raise ValueError(f'Invalid merge mode {self.merge_mode}')


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
    def __init__(self, gnn_dim, gnn):
        super(DMConditionEncoder, self).__init__()
        self.gnn, self.mode = gnn, 'independent'
        self.empty_mol = torch.nn.Parameter(torch.randn(gnn_dim))

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
            encoder = DMConditionEncoder(300, gnn)
        else:
            gnn = PretrainGIN(**config['arch'])
            encoder = DMConditionEncoder(config['dim'], gnn, config['mode'])
    elif config['type'] == 'gat':
        dropout = config['arch'].get('dropout', dropout)
        config['arch']['dropout'] = dropout
        gnn = SimpleCondGAT(**config['arch'])
        encoder = DMConditionEncoder(config['dim'], gnn)
    else:
        raise NotImplementedError(f'Invalid gnn type {config["type"]}')

    return encoder
