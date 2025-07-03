import torch
import numpy as np

from tqdm import tqdm
from torch.nn.functional import kl_div, mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..tensor_utils import (
    generate_local_global_mask,
    generate_tgt_mask, calc_trans_loss,
    correct_trans_output, data_eval_trans,
    convert_log_into_label
)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_mol_yield(
    loader, model, optimizer, device, heads=None,
    local_global=False, warmup=False, loss_fun='kl'
):
    if local_global and heads is None:
        raise ValueError("require num heads for local global mask")
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, reag, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        reag, label = reag.to(device), label.to(device)
        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        res = model(reac, prod, reag, cross_mask=cross_mask)
        if loss_fun == 'kl':
            assert res.shape[-1] == 2, 'kl requires two outputs'
            res = torch.log_softmax(res, dim=-1)
            sm_label = torch.stack([label, 100 - label], dim=1) / 100
            loss = kl_div(res, sm_label, reduction='batchmean')
        else:
            assert loss_fun == 'mse', f'Invalid loss_fun {loss_fun}'
            assert res.shape[-1] == 1, 'requires single output'
            loss = mse_loss(label / 100, res.squeeze(dim=-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_mol_yield(loader, model, device, heads=None, local_global=False, return_raw=False):
    model, ytrue, ypred = model.eval(), [], []
    for reac, prod, reag, label in tqdm(loader):
        reac, prod, reag = reac.to(device), prod.to(device), reag.to(device)
        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(reac, prod, reag, cross_mask=cross_mask)
            if res.shape[-1] == 2:
                res = res.softmax(dim=-1)[:, 0] * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())
            else:
                res = torch.clamp(res, 0, 1) * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())

    ypred = np.concatenate(ypred, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)

    result = {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }

    if return_raw:
        result['ytrue'] = ytrue
        result['ypred'] = ypred
    return result


def train_az_yield(
    loader, model, optimizer, device, heads=None,
    local_global=False, warmup=False, loss_fun='kl'
):
    if local_global and heads is None:
        raise ValueError("require num heads for local global mask")
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for data in tqdm(loader):
        reac, prod, reag, vols, temps, label = data
        reac, prod, temps = reac.to(device), prod.to(device), temps.to(device)
        reag, label = reag.to(device), label.to(device)
        label = torch.clamp(label, 0, 100)
        vols = {k: v.to(device) for k, v in vols.items()}

        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        res = model(
            reac_graph=reac, prod_graph=prod, conditions=reag,
            cross_mask=cross_mask, temperatures=temps, keys_to_volumns=vols
        )
        if loss_fun == 'kl':
            assert res.shape[-1] == 2, 'kl requires two outputs'
            res = torch.log_softmax(res, dim=-1)
            sm_label = torch.stack([label, 100 - label], dim=1) / 100
            loss = kl_div(res, sm_label, reduction='batchmean')
        else:
            assert loss_fun == 'mse', f'Invalid loss_fun {loss_fun}'
            assert res.shape[-1] == 1, 'requires single output'
            loss = mse_loss(label / 100, res.squeeze(dim=-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_az_yield(loader, model, device, heads=None, local_global=False):
    model, ytrue, ypred = model.eval(), [], []
    for data in tqdm(loader):
        reac, prod, reag, vols, temps, label = data
        reac, prod = reac.to(device), prod.to(device)
        reag, temps = reag.to(device), temps.to(device)
        vols = {k: v.to(device) for k, v in vols.items()}
        label = torch.clamp(label, 0, 100)
        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(
                reac_graph=reac, prod_graph=prod, conditions=reag,
                cross_mask=cross_mask, temperatures=temps, keys_to_volumns=vols
            )
            if res.shape[-1] == 2:
                res = res.softmax(dim=-1)[:, 0] * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())
            else:
                res = torch.clamp(res, 0, 1) * 100
                ytrue.append(label.numpy())
                ypred.append(res.cpu().numpy())

    ypred = np.concatenate(ypred, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)

    return {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }


def train_regression(
    loader, model, optimizer, device, heads=None,
    local_global=False, warmup=False
):
    if local_global and heads is None:
        raise ValueError("require num heads for local global mask")
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, reag, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        reag, label = reag.to(device), label.to(device)
        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        res = model(reac, prod, reag, cross_mask=cross_mask)
        assert res.shape[-1] == 1, 'requires single output'
        loss = mse_loss(label, res.squeeze(dim=-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_regression(loader, model, device, heads=None, local_global=False, return_raw=False):
    model, ytrue, ypred = model.eval(), [], []
    for reac, prod, reag, label in tqdm(loader):
        reac, prod, reag = reac.to(device), prod.to(device), reag.to(device)
        if local_global:
            cross_mask = generate_local_global_mask(reac, prod, 1, heads)
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(reac, prod, reag, cross_mask=cross_mask)
            ytrue.append(label.numpy())
            ypred.append(res.cpu().numpy())

    ypred = np.concatenate(ypred, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)

    result = {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }

    if return_raw:
        result['ytrue'] = ytrue
        result['ypred'] = ypred
    return result

def train_gen(
    loader, model, optimizer, device, pad_idx, toker,
    heads=None, local_global=False, warmup=False,
):
    if local_global and heads is None:
        raise ValueError("require num heads for local global mask")
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        tgt = toker.encode2d(label)
        tgt = torch.LongTensor(tgt).to(device)

        trans_dec_ip = tgt[:, :-1]
        trans_dec_op = tgt[:, 1:]
        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, pad_idx=pad_idx, device=device
        )
        if local_global:
            Qlen = trans_dec_ip.shape[1]
            cross_mask = generate_local_global_mask(reac, prod, Qlen, heads)
        else:
            cross_mask = None

        trans_logs = model(
            reac_graph=reac, prod_graph=prod, tgt=trans_dec_ip,
            tgt_mask=diag_mask, cross_mask=cross_mask,
            tgt_key_padding_mask=trans_op_mask
        )

        loss = calc_trans_loss(trans_logs, trans_dec_op, pad_idx)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_gen(
    loader, model, device, pad_idx, end_idx,
    toker, heads=None, local_global=False
):
    model, accx = model.eval(), []
    for reac, prod, label in tqdm(loader):
        reac, prod = reac.to(device), prod.to(device)
        tgt = toker.encode2d(label)
        tgt = torch.LongTensor(tgt).to(device)

        trans_dec_ip = tgt[:, :-1]
        trans_dec_op = tgt[:, 1:]
        trans_op_mask, diag_mask = generate_tgt_mask(
            trans_dec_ip, pad_idx=pad_idx, device=device
        )
        if local_global:
            Qlen = trans_dec_ip.shape[1]
            cross_mask = generate_local_global_mask(reac, prod, Qlen, heads)
        else:
            cross_mask = None

        with torch.no_grad():
            trans_logs = model(
                reac_graph=reac, prod_graph=prod, tgt=trans_dec_ip,
                tgt_mask=diag_mask, cross_mask=cross_mask,
                tgt_key_padding_mask=trans_op_mask
            )

        trans_pred = convert_log_into_label(trans_logs, mod='softmax')
        trans_pred = correct_trans_output(trans_pred, end_idx, pad_idx)
        trans_acc = data_eval_trans(trans_pred, trans_dec_op, True)
        accx.append(trans_acc)

    accx = torch.cat(accx, dim=0).float()
    return accx.mean().item()


def train_uspto_condition(
    loader, model, optimizer, device, heads=None,
    local_global=False, warmup=False
):
    if local_global and heads is None:
        raise ValueError("require num heads for local global mask")
    model, los_cur = model.train(), []
    if warmup:
        warmup_iters = len(loader) - 1
        warmup_sher = warmup_lr_scheduler(optimizer, warmup_iters, 5e-2)

    for reac, prod, label in tqdm(loader):
        reac, prod, label = reac.to(device), prod.to(device), label.to(device)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]

        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, device)

        if local_global:
            Qlen = tgt_in.shape[1]
            cross_mask = generate_local_global_mask(reac, prod, Qlen, heads)
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
        los_cur.append(loss.item())
        if warmup:
            warmup_sher.step()

    return np.mean(los_cur)


def eval_uspto_condition(
    loader, model, device, heads=None, local_global=False
):
    model, accs, gt = model.eval(), [], []
    for reac, prod, label in tqdm(loader):
        reac, prod, label = reac.to(device), prod.to(device), label.to(device)
        tgt_in, tgt_out = label[:, :-1], label[:, 1:]
        pad_mask, sub_mask = generate_tgt_mask(tgt_in, -1000, device)

        if local_global:
            Qlen = tgt_in.shape[1]
            cross_mask = generate_local_global_mask(reac, prod, Qlen, heads)
        else:
            cross_mask = None

        with torch.no_grad():
            res = model(
                reac, prod, tgt_in, tgt_mask=sub_mask,
                tgt_key_padding_mask=pad_mask, cross_mask=cross_mask
            )

            result = convert_log_into_label(res, mod='softmax')

        accs.append(result)
        gt.append(tgt_out)

    accs = torch.cat(accs, dim=0)
    gt = torch.cat(gt, dim=0)

    keys = ['catalyst', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    results, overall = {}, None
    for idx, k in enumerate(keys):
        results[k] = accs[:, idx] == gt[:, idx]
        if idx == 0:
            overall = accs[:, idx] == gt[:, idx]
        else:
            overall &= (accs[:, idx] == gt[:, idx])

    results['overall'] = overall
    results = {k: v.float().mean().item() for k, v in results.items()}
    return results
