from tqdm import tqdm
import numpy as np
import torch

from torch.nn.functional import cross_entropy
from enum import Enum
import torch.distributed as torch_dist

from ..tensor_utils import (
    generate_local_global_mask, generate_tgt_mask,
    calc_trans_loss, convert_log_into_label
)
from .training import warmup_lr_scheduler


class Summary(Enum):
    NONE, SUM, AVERAGE, COUNT = 0, 1, 2, 3


class MetricCollector(object):
    def __init__(self, name, type_fmt=':f', summary_type=Summary.AVERAGE):
        super(MetricCollector, self).__init__()
        self.name, self.type_fmt = name, type_fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val, self.sum, self.cnt, self.avg = [0] * 4

    def update(self, val, num=1):
        self.val = val
        self.sum += val
        self.cnt += num
        self.avg = self.sum / self.cnt

    def all_reduce(self, device):
        infos = torch.FloatTensor([self.sum, self.cnt]).to(device)
        torch_dist.all_reduce(infos, torch_dist.ReduceOp.SUM)
        self.sum, self.cnt = infos.tolist()
        self.avg = self.sum / self.cnt

    def __str__(self):
        return ''.join([
            '{name}: {val', self.type_fmt, '} avg: {avg', self.type_fmt, '}'
        ]).format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {cnt:.3f}'
        else:
            raise ValueError(f'Invaild summary type {self.summary_type} found')

        return fmtstr.format(**self.__dict__)

    def get_value(self):
        if self.summary_type is Summary.AVERAGE:
            return self.avg
        elif self.summary_type is Summary.SUM:
            return self.sum
        elif self.summary_type is Summary.COUNT:
            return self.cnt
        else:
            raise ValueError(
                f'Invaild summary type {self.summary_type} '
                'for get_value()'
            )


class MetricManager(object):
    def __init__(self, metrics):
        super(MetricManager, self).__init__()
        self.metrics = metrics

    def all_reduct(self, device):
        for idx in range(len(self.metrics)):
            self.metrics[idx].all_reduce(device)

    def summary_all(self, split_string='  '):
        return split_string.join(x.summary() for x in self.metrics)

    def get_all_value_dict(self):
        return {x.name: x.get_value() for x in self.metrics}


def ddp_train_uspto_condition(
    loader, model, optimizer, device, total_heads=None,
    local_heads=0, warmup=False, verbose=False
):
    model = model.train()
    loss_cur = MetricCollector(name='train_loss', type_fmt=':.2f')
    manager = MetricManager([loss_cur])

    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    if verbose:
        iterx = tqdm(loader, desc='train')
    else:
        iterx = loader
    for reac, prod, label in iterx:
        reac = reac.to(device, non_blocking=True)
        prod = prod.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, 'cpu')
        pad_mask = pad_mask.to(device, non_blocking=True)
        sub_mask = sub_mask.to(device, non_blocking=True)

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, tgt_in.shape[1], total_heads, local_heads
            )
        else:
            cross_mask = None
        res = model(
            reac, prod, tgt_in, tgt_mask=sub_mask,
            tgt_key_padding_mask=pad_mask, cross_mask=cross_mask
        )

        loss = calc_trans_loss(res, tgt_out, -1000)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_cur.update(loss.item())

        if warmup:
            warmup_sher.step()

        if verbose:
            iterx.set_postfix_str(manager.summary_all(split_string=','))

    return manager


def ddp_eval_uspto_condition(
    loader, model, device, total_heads=None, local_heads=0, verbose=False
):
    model = model.eval()
    cat_acc = MetricCollector('catalyst', type_fmt=':.2f')
    sov1_acc = MetricCollector('solvent1', type_fmt=":.2f")
    sov2_acc = MetricCollector('solvent2', type_fmt=':.2f')
    reg1_acc = MetricCollector('reagent1', type_fmt=':.2f')
    reg2_acc = MetricCollector('reagent2', type_fmt=':.2f')
    ov = MetricCollector('overall', type_fmt=':.2f')
    man = MetricManager([cat_acc, sov1_acc, sov2_acc, reg1_acc, reg2_acc, ov])

    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']

    if verbose:
        iterx = tqdm(loader, desc='eval')
    else:
        iterx = loader

    for reac, prod, label in iterx:
        reac = reac.to(device, non_blocking=True)
        prod = prod.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]
        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, 'cpu')
        pad_mask = pad_mask.to(device, non_blocking=True)
        sub_mask = sub_mask.to(device, non_blocking=True)

        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, tgt_in.shape[1], total_heads, local_heads
            )
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(
                reac, prod, tgt_in, tgt_mask=sub_mask,
                tgt_key_padding_mask=pad_mask, cross_mask=cross_mask
            )

            result = convert_log_into_label(res, mod='softmax')

        ovr = None
        for idx, k in enumerate(keys):
            pt = result[:, idx] == tgt_out[:, idx]
            A, B = pt.sum().item(), pt.shape[0]
            man.metrics[idx].update(val=A, num=B)
            ovr = pt if ovr is None else (ovr & pt)
        ov.update(ovr.sum().item(), ovr.shape[0])

        if verbose:
            iterx.set_postfix_str(man.summary_all(split_string=','))

    return man
