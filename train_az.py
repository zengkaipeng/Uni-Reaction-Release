import torch
import os
import time
import argparse
import json

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils.data_utils import load_az_yield, fix_seed, count_parameters
from utils.training import train_az_yield, eval_az_yield
from utils.Dataset import az_colfn

from model import AzYieldModel, RAlignEncoder, build_az_condition_encoder


def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    r2_dir = os.path.join(detail_dir, 'model.pth')
    mse_dir = os.path.join(detail_dir, 'mse.pth')
    return log_dir, r2_dir, mse_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for prediction model')
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset'
    )
    parser.add_argument(
        '--dim', type=int, default=512,
        help='the number of dim for model'
    )
    parser.add_argument(
        '--heads', type=int, default=8,
        help='the number of heads for model'
    )
    parser.add_argument(
        '--n_layer', type=int, default=8,
        help='the number of layers of the model'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout ratio for model'
    )
    parser.add_argument(
        '--warmup', type=int, default=0,
        help='the number of epochs for warmup'
    )
    parser.add_argument(
        '--lrgamma', type=float, default=1,
        help='the lr decay rate for training'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='the learning rate for training'
    )
    parser.add_argument(
        '--epoch', type=int, default=100,
        help='the number for epochs for training'
    )
    parser.add_argument(
        '--base_log', type=str, default='log_cn',
        help='the path for contraining log'
    )
    parser.add_argument(
        '--num_worker', type=int, default=8,
        help='the number of worker for dataloader'
    )
    parser.add_argument(
        '--bs', type=int, default=64,
        help='the batch size for training'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='the device id for traiing, negative for cpu'
    )
    parser.add_argument(
        '--step_start', type=int, default=10,
        help='the step to start lr decay'
    )
    parser.add_argument(
        '--seed', type=int, default=2025,
        help='the random seed for training'
    )
    parser.add_argument(
        '--condition_config', type=str, required=True,
        help='the path of json containing the config for condition encoder'
    )
    parser.add_argument(
        '--condition_both', action='store_true',
        help='the add condition to both reactant and product'
    )
    parser.add_argument(
        '--use_temperature', action='store_true',
        help='use the temperature information or not'
    )
    parser.add_argument(
        '--use_solvent_volumn', action='store_true',
        help='use the solvent volumn information or not'
    )
    parser.add_argument(
        '--use_volumn', action='store_true',
        help='use the volumn of other reagents or not'
    )
    parser.add_argument(
        '--volumn_norm', choices=['min_as_one', 'absolute'],
        default='absolute', type=str,
        help='the method to process the volumn information'
    )
    parser.add_argument(
        '--temperature_cls', default=20, type=int,
        help='the class num for temperature encoding (default: 20)'
    )
    parser.add_argument(
        '--volumn_cls', default=20, type=int,
        help='the class num for volumn encoding (default: 20)'
    )
    parser.add_argument(
        '--solvent_volumn_cls', default=20, type=int,
        help='the class num for solvent volumn encoding (default: 20)'
    )
    parser.add_argument(
        '--temperature_scale', default=100, type=float,
        help='the scaling for temperature (default: 100)'
    )
    parser.add_argument(
        '--solvent_volumn_scale', default=1, type=float,
        help='the scaling for solvent volumn (default: 1)'
    )
    parser.add_argument(
        '--loss', choices=['mse', 'kl'], default='kl',
        help='the loss type for training'
    )

    parser.add_argument(
        '--local_heads', type=int, default=0,
        help='the number of local heads in attention'
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.condition_config) as Fin:
        condition_config = json.load(Fin)

    train_set, val_set, test_set = load_az_yield(
        data_path=args.data_path,
        condition_type=condition_config['data_type'],
        vol_type=args.volumn_norm,
        temperature_scale=args.temperature_scale,
        solvent_vol_scale=args.solvent_volumn_scale
    )

    log_dir, r2_dir, mse_dir = make_dir(args)

    train_loader = DataLoader(
        train_set, batch_size=args.bs, shuffle=True,
        collate_fn=az_colfn, num_workers=args.num_worker,
    )

    val_loader = DataLoader(
        val_set, batch_size=args.bs, shuffle=False,
        collate_fn=az_colfn, num_workers=args.num_worker
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=az_colfn, num_workers=args.num_worker
    )

    if condition_config['mode'] == 'mix-all':
        condition_infos = {
            'mixed': {
                'dim': condition_config['dim'],
                'heads': args.heads
            }
        }
    elif condition_config['mode'] == 'mix-meta-ligand':
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['solvent', 'base', 'meta_and_ligand']
        }
    else:
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['ligand', 'base', 'meta', 'solvent']
        }

    reac_num_keys, prod_num_keys = {}, {}
    if args.use_volumn:
        reac_num_keys['volumn'] = args.dim
    if args.use_temperature:
        reac_num_keys['temperature'] = args.dim
        if args.condition_both:
            prod_num_keys['temperature'] = args.dim

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim, edge_dim=args.dim,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys=prod_num_keys, reac_batch_infos=condition_infos,
        reac_num_keys=reac_num_keys, dropout=args.dropout, heads=args.heads,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_az_condition_encoder(
        config=condition_config, dropout=args.dropout,
        use_temperature=args.use_temperature,
        use_sol_vol=args.use_solvent_volumn,
        use_vol=args.use_volumn, num_emb_dim=args.dim
    )

    model = AzYieldModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=args.dropout, heads=args.heads,
        use_temperature=args.use_temperature, use_volumn=args.use_volumn,
        use_sol_volumn=args.use_solvent_volumn,
        temperature_cls=args.temperature_cls,
        volumn_cls=args.volumn_cls, sol_volumn_cls=args.solvent_volumn_cls,
        out_dim=2 if args.loss == 'kl' else 1
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ExponentialLR(optimizer, gamma=args.lrgamma)

    log_info = {
        'args': args.__dict__, 'train_loss': [], 'valid_metric': [],
        'test_metric': [], 'total_params': total_params,
        'trainable_params': trainable_params
    }

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    best_pref, best_ep, best_mse, best_ep2 = [None] * 4

    for ep in range(args.epoch):
        print(f'[INFO] training epoch {ep}')
        loss = train_az_yield(
            train_loader, model, optimizer, device, warmup=(ep < args.warmup),
            total_heads=args.heads, local_heads=args.local_heads,
            loss_fun=args.loss
        )
        val_results = eval_az_yield(
            val_loader, model, device, total_heads=args.heads,
            local_heads=args.local_heads
        )
        test_results = eval_az_yield(
            test_loader, model, device, total_heads=args.heads,
            local_heads=args.local_heads
        )

        print('[Train]:', loss)
        print('[Valid]:', val_results)
        print('[Test]:', test_results)

        log_info['train_loss'].append(loss)
        log_info['valid_metric'].append(val_results)
        log_info['test_metric'].append(test_results)

        if ep >= args.warmup and ep >= args.step_start:
            lr_sher.step()
            print('[lr]', lr_sher.get_last_lr())

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

        if best_pref is None or val_results['R2'] > best_pref:
            best_pref, best_ep = val_results['R2'], ep
            torch.save(model.state_dict(), r2_dir)

        if best_mse is None or val_results['MSE'] < best_mse:
            best_ep2, best_mse = ep, val_results['MSE']
            torch.save(model.state_dict(), mse_dir)

    print(f'[INFO] best R2 epoch: {best_ep}')
    print(f'[INFO] best R2 valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best R2 test loss: {log_info["test_metric"][best_ep]}')

    print(f'[INFO] best MSE epoch: {best_ep2}')
    print(f'[INFO] best MSE valid loss: {log_info["valid_metric"][best_ep2]}')
    print(f'[INFO] best MSE test loss: {log_info["test_metric"][best_ep2]}')
