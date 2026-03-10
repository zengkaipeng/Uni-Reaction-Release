import torch
import os
import time
import argparse
import json
import rdkit.RDLogger as rdl

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_utils import load_uspto_yield, fix_seed, count_parameters
from utils.training import ddp_train_uspto_yield, ddp_eval_uspto_yield
from utils.uspto_ds import ds_col_fn

from model import (
    USPTOYield, RAlignEncoder, ReagentEncoderWithAmount, MultiNumEmb
)


def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    r2_dir = os.path.join(detail_dir, 'model.pth')
    mse_dir = os.path.join(detail_dir, 'mse.pth')
    return log_dir, r2_dir, mse_dir


# 创建一个RDKit日志器并设置为只记录错误
logger = rdl.logger()
logger.setLevel(rdl.ERROR)


def gather_to_rank0(tensor, device, rank, world_size):
    if rank == 0:
        # 根进程准备接收所有数据
        all_tensors = [tensor]  # 先加入自己的数据

        # 从其他进程接收数据
        for src_rank in range(1, world_size):
            # 先接收长度信息
            length_tensor = torch.zeros(1, dtype=torch.long, device=device)
            torch_dist.recv(length_tensor, src=src_rank)
            length = length_tensor.item()

            # 接收实际数据
            recv_tensor = torch.zeros(
                length, dtype=tensor.dtype, device=device
            )
            torch_dist.recv(recv_tensor, src=src_rank)
            all_tensors.append(recv_tensor)

        # 合并所有数据
        return torch.cat(all_tensors, dim=0)
    else:
        # 非根进程发送数据到根进程
        # 先发送长度信息
        length_tensor = torch.tensor(
            [tensor.shape[0]], dtype=torch.long, device=device
        )
        torch_dist.send(length_tensor, dst=0)

        # 发送实际数据
        torch_dist.send(tensor, dst=0)
        return None


def calc_metrics(pred, gt, device, world_size, rank):
    gathered_y_true = gather_to_rank0(gt, device, rank, world_size)
    gathered_y_pred = gather_to_rank0(pred, device, rank, world_size)
    if rank == 0:
        y_true_np = gathered_y_true.cpu().numpy()
        y_pred_np = gathered_y_pred.cpu().numpy()
        r2 = r2_score(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        mse = mean_squared_error(y_true_np, y_pred_np)
        rmse = mse ** 0.5
        metrics_tensor = torch.tensor([r2, mae, mse, rmse], device=device)
    else:
        metrics_tensor = torch.zeros(4, device=device)

    torch_dist.broadcast(metrics_tensor, src=0)
    return {
        'R2': metrics_tensor[0].item(),
        'MAE': metrics_tensor[1].item(),
        'MSE': metrics_tensor[2].item(),
        'RMSE': metrics_tensor[3].item()
    }


def main_worker(worker_idx, args, log_dir, model_dir, mse_dir):
    print(f'[INFO] Process {worker_idx} start')
    torch_dist.init_process_group(
        backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.num_gpus, rank=worker_idx
    )
    device = torch.device(f'cuda:{worker_idx}')
    verbose = (worker_idx == 0)

    all_ds = load_uspto_yield(args.data_path)
    print(f'[INFO] worker {worker_idx} Data Loaded')

    train_sampler = DistributedSampler(all_ds['train'], shuffle=True)
    valid_sampler = DistributedSampler(all_ds['val'], shuffle=False)
    test_sampler = DistributedSampler(all_ds['test'], shuffle=False)

    train_loader = DataLoader(
        all_ds['train'], batch_size=args.bs, shuffle=False,
        collate_fn=ds_col_fn, num_workers=args.num_worker,
        pin_memory=True, sampler=train_sampler
    )

    val_loader = DataLoader(
        all_ds['val'], batch_size=args.bs, shuffle=False,
        collate_fn=ds_col_fn, num_workers=args.num_worker,
        pin_memory=True, sampler=valid_sampler
    )

    test_loader = DataLoader(
        all_ds['test'], batch_size=args.bs, shuffle=False,
        collate_fn=ds_col_fn, num_workers=args.num_worker,
        pin_memory=True, sampler=test_sampler
    )

    condition_infos = {'shared': {'dim': args.dim, 'heads': args.heads}}
    reac_num_conditions, prod_num_conditions = {}, {}
    if args.amount_class > 0:
        reac_num_conditions['amount'] = args.dim
        amount_encoder = MultiNumEmb(
            emb_dim=args.dim, allow_unk=True,
            num_cls=[args.amount_class, args.amount_class]
        )
    else:
        amount_encoder = None
    if args.temperature_class > 0:
        temperature_encoder = MultiNumEmb(
            emb_dim=args.dim, allow_unk=True,
            num_cls=[args.temperature_class]
        )
        prod_num_conditions['temperature'] = args.dim
        reac_num_conditions['temperature'] = args.dim
    else:
        temperature_encoder = None

    condition_encoder = ReagentEncoderWithAmount(
        gnn_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
        negative_slope=args.negative_slope, dropout=args.dropout,
        amount_encoder=amount_encoder, num_emb_dim=args.dim,
        use_amount=args.amount_class > 0, merge_mode='sep'
    )
    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim,  edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos, prod_num_keys=prod_num_conditions,
        reac_num_keys=reac_num_conditions, dropout=args.dropout,
        negative_slope=args.negative_slope, update_last_edge=False,
        fusion_order='film_first'
    )

    model = USPTOYield(
        encoder=encoder, condition_encoder=condition_encoder,
        amount_encoder=amount_encoder, temperature_encoder=temperature_encoder,
        dim=args.dim, dropout=args.dropout, heads=args.heads
    ).to(device)

    total_params, _ = count_parameters(model)
    if verbose:
        print('total_params', total_params)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[worker_idx], output_device=worker_idx,
        find_unused_parameters=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ExponentialLR(optimizer, gamma=args.lrgamma)

    log_info = {
        'args': args.__dict__, 'train_loss': [], 'valid_metric': [],
        'test_metric': [], 'total_params': total_params,
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    best_pref, best_ep, best_mse, best_ep2 = [None] * 4

    for ep in range(args.epoch):
        if verbose:
            print(f'[INFO] training epoch {ep}')
        train_sampler.set_epoch(ep)
        loss = ddp_train_uspto_yield(
            train_loader, model, optimizer, device, warmup=(ep < args.warmup),
            total_heads=args.heads, local_heads=args.local_heads,
            loss_fun='kl', verbose=verbose
        )
        val_results = ddp_eval_uspto_yield(
            val_loader, model, device, total_heads=args.heads,
            local_heads=args.local_heads, verbose=verbose
        )
        test_results = ddp_eval_uspto_yield(
            test_loader, model, device, total_heads=args.heads,
            local_heads=args.local_heads, verbose=verbose
        )
        torch_dist.barrier()
        loss.all_reduct(device)
        valid_metrics = calc_metrics(
            val_results['prediction'], val_results['ground_truth'],
            device, args.num_gpus, worker_idx
        )
        test_metrics = calc_metrics(
            test_results['prediction'], test_results['ground_truth'],
            device, args.num_gpus, worker_idx
        )

        log_info['train_loss'].append(loss.get_all_value_dict())
        log_info['valid_metric'].append(valid_metrics)
        log_info['test_metric'].append(test_metrics)

        if ep >= args.warmup and ep >= args.step_start:
            lr_sher.step()
            if verbose:
                print('[lr]', lr_sher.get_last_lr())

        if verbose:
            print('[Train]', log_info['train_loss'][-1])
            print('[Valid]', log_info['valid_metric'][-1])
            print('[Test]', log_info['test_metric'][-1])

            with open(log_dir, 'w') as Fout:
                json.dump(log_info, Fout, indent=4)

            this_val = log_info['valid_metric'][-1]
            this_test = log_info['test_metric'][-1]

            if best_pref is None or this_val['R2'] > best_pref:
                best_pref, best_ep = this_val['R2'], ep
                torch.save(model.module.state_dict(), model_dir)

            if best_mse is None or this_test['MSE'] < best_mse:
                best_ep2, best_mse = ep, this_test['MSE']
                torch.save(model.module.state_dict(), mse_dir)

    if not verbose:
        torch_dist.destroy_process_group()
        return

    print(f'[INFO] best R2 epoch: {best_ep}')
    print(f'[INFO] best R2 valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best R2 test loss: {log_info["test_metric"][best_ep]}')

    print(f'[INFO] best MSE epoch: {best_ep2}')
    print(f'[INFO] best MSE valid loss: {log_info["valid_metric"][best_ep2]}')
    print(f'[INFO] best MSE test loss: {log_info["test_metric"][best_ep2]}')
    torch_dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for prediction model')
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset (required)'
    )
    parser.add_argument(
        '--dim', type=int, default=192,
        help='the number of dim for model (default: 192)'
    )
    parser.add_argument(
        '--heads', type=int, default=6,
        help='the number of heads for model (default: 6)'
    )
    parser.add_argument(
        '--n_layer', type=int, default=6,
        help='the number of layers of the model (default: 6)'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout ratio for model (default: 0.1)'
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='the number of epochs for warmup (default: 10)'
    )
    parser.add_argument(
        '--lrgamma', type=float, default=0.997,
        help='the lr decay rate for training (default: 0.997)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='the learning rate for training (default: 1e-4)'
    )
    parser.add_argument(
        '--epoch', type=int, default=200,
        help='the number for epochs for training (default: 200)'
    )
    parser.add_argument(
        '--base_log', type=str, default='log_uspto_yield',
        help='the path for contraining log'
    )
    parser.add_argument(
        '--num_worker', type=int, default=8,
        help='the number of worker for dataloader (default: 8)'
    )
    parser.add_argument(
        '--bs', type=int, default=512,
        help='the batch size for training (default: 512)'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model (default: 0.2)'
    )
    parser.add_argument(
        '--step_start', type=int, default=15,
        help='the step to start lr decay (default: 15)'
    )
    parser.add_argument(
        '--seed', type=int, default=2026,
        help='the random seed for training (default: 2026)'
    )
    parser.add_argument(
        '--local_heads', type=int, default=2,
        help='the number of local heads in attention (default: 2)'
    )
    parser.add_argument(
        '--amount_class', type=int, default=50,
        help='the number of class for amount embedding' +
        ' non-positive to disable the amount encoder (default: 50)'
    )
    parser.add_argument(
        '--temperature_class', type=int, default=-1,
        help='the number of class for temperature embedding' +
        ' non-positive to disable the temperature encoder (default: -1)'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=4,
        help='the number of gpus to run exp (default: 4)'
    )
    parser.add_argument(
        '--port', type=int, default=13487,
        help='the port id for ddp communications (default: 13487)'
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    log_dir, r2_dir, mse_dir = make_dir(args)

    torch_mp.spawn(
        main_worker, nprocs=args.num_gpus,
        args=(args, log_dir, r2_dir, mse_dir)
    )
