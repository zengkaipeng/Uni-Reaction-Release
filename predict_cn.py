import torch
import os
import time
import argparse
import json
import pandas as pd

from torch.utils.data import DataLoader

from utils.data_utils import load_cn_yield_one, fix_seed
from utils.training import eval_mol_yield
from utils.Dataset import cn_colfn

from model import (
    CNYieldModel, RAlignEncoder, build_cn_condition_encoder_with_eval
)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for prediction model')
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset'
    )
    parser.add_argument(
        '--dim', type=int, default=128,
        help='the number of dim for model'
    )
    parser.add_argument(
        '--heads', type=int, default=8,
        help='the number of heads for model'
    )
    parser.add_argument(
        '--n_layer', type=int, default=3,
        help='the number of layers of the model'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='the dropout ratio for model'
    )
    parser.add_argument(
        '--num_worker', type=int, default=8,
        help='the number of worker for dataloader'
    )
    parser.add_argument(
        '--bs', type=int, default=128,
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
        '--local_heads', type=int, default=4,
        help='the number of local heads in attention'
    )
    parser.add_argument(
        '--output_path', required=True, type=str,
        help='the path of json file to store results'
    )

    cmd_args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.condition_config) as Fin:
        condition_config = json.load(Fin)

    test_set = load_cn_yield_one(
        args.data_path, 'test', condition_config['data_type']
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
            for k in ['additive', 'base', 'catalyst and ligand']
        }
    else:
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['ligand', 'base', 'additive', 'catalyst']
        }

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim,  edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys={}, reac_num_keys={}, dropout=args.dropout,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_cn_condition_encoder(
        config=condition_config, dropout=args.dropout
    )

    model = CNYieldModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=args.dropout, heads=args.heads
    ).to(device)

    # load ckpt from log dir
    print('[INFO] loading model from', args.checkpoint)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)

    test_results = eval_mol_yield(
        test_loader, model, device, total_heads=args.total_heads,
        local_heads=args.local_heads, return_raw=True
    )
    df = pd.DataFrame({
        'prediction': test_results['prediction'],
        'label': test_results['label']
    })

    with open(args.output_path, 'w') as f:
        json.dump(test_results, f, indent=4)

    print('MAE:', test_results['MAE'])
    print('RMSE:', test_results['MSR'] ** 0.5)
    print('R2:', test_results['R2'])
