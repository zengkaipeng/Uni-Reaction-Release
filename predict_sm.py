import torch
import os
import time
import argparse
import json
import pandas as pd

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils.data_utils import load_sm_yield_one, count_parameters
from utils.training import train_mol_yield, eval_mol_yield
from utils.Dataset import cn_colfn

from model import CNYieldModel, RAlignEncoder, build_sm_condition_encoder


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
        '--part', '-p', default='test', type=str,
        help='the part of dataset to be used for prediction'
    )
    parser.add_argument(
        '--log_dir', '-l', type=str, default='logs',
        help='the path of log directory'
    )

    cmd_args = parser.parse_args()
    log_dir = cmd_args.log_dir
    with open(os.path.join(log_dir, 'log.json'), 'r') as fin:
        log_info = json.load(fin)

    args = argparse.Namespace(**log_info['args'])    
    data_path = args.data_path

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.condition_config) as Fin:
        condition_config = json.load(Fin)

    test_set = load_sm_yield_one(
        data_path, cmd_args.part, condition_config['data_type']
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=cn_colfn, num_workers=args.num_worker
    )

    if condition_config['mode'] == 'mix-all':
        condition_infos = {
            'mixed': {
                'dim': condition_config['dim'],
                'heads': args.heads
            }
        }
    elif condition_config['mode'] == 'mix-catalyst-ligand':
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['solvent', 'catalyst and ligand']
        }
    else:
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['ligand', 'catalyst', 'solvent']
        }

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim,  edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys={}, reac_num_keys={}, dropout=args.dropout,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_sm_condition_encoder(
        config=condition_config, dropout=args.dropout
    )

    model = CNYieldModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=args.dropout, heads=args.heads,
        out_dim=1 if args.loss == 'mse' else 2
    ).to(device)

    # load ckpt from log dir
    print('[INFO] loading model from', log_dir)
    model.load_state_dict(
        torch.load(os.path.join(log_dir, 'model.pth'), map_location=device)
    )    


    test_results = eval_mol_yield(
        test_loader, model, device, heads=args.heads, local_global=True,
        return_raw=True
    )

    print(test_results)
    pred_yield = test_results['ypred']
    true_yield = test_results['ytrue']
    df = pd.DataFrame({
        'pred_yield': pred_yield.flatten(),
        'true_yield': true_yield.flatten()
    })
    data_postfix = data_path.replace('/', '_').replace('\\', '_')
    df.to_csv(os.path.join(log_dir, f'pred_{cmd_args.part}.csv'), index=False)