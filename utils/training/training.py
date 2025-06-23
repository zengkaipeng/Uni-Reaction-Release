import torch
import numpy as np

from tqdm import tqdm
from torch.nn.functional import kl_div, mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..tensor_utils import generate_local_global_mask


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


def eval_mol_yield(loader, model, device, heads=None, local_global=False):
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

    return {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }


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


def eval_regression(loader, model, device, heads=None, local_global=False):
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

    return {
        'MAE': float(mean_absolute_error(ytrue, ypred)),
        'MSE': float(mean_squared_error(ytrue, ypred)),
        'R2': float(r2_score(ytrue, ypred))
    }
