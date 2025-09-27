import torch
import os
import time
import argparse
import json
import pandas as pd

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils.data_utils import load_sel, fix_seed, count_parameters
from utils.training import train_regression, eval_regression
from utils.Dataset import sel_with_cat_colfn

from model import RegressionModel, RAlignEncoder, build_dm_condition_encoder


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
        '--epoch', type=int, default=250,
        help='the number for epochs for training'
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
    parser.add_argument(
        '--checkpoint', required=True, type=str,
        help='the checkpoint for model'
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.condition_config) as Fin:
        condition_config = json.load(Fin)

    train_set, val_set, test_set = load_sel(
        args.data_path, condition_config['data_type']
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=sel_with_cat_colfn, num_workers=args.num_worker
    )

    condition_infos = {
        'catalyst': {'dim': condition_config['dim'], 'heads': args.heads}
    }

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim,  edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys={}, reac_num_keys={}, dropout=0,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_dm_condition_encoder(
        config=condition_config, dropout=0
    )

    model = RegressionModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=0, heads=args.heads
    ).to(device)

    print('[INFO] loading model from', args.checkpoint)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)

    test_results = eval_regression(
        test_loader, model, device, return_raw=True,
        total_heads=args.heads, local_heads=args.local_heads
    )

    with open(args.output_path, 'w') as f:
        json.dump(test_results, f, indent=4)

    print('MAE:', test_results['MAE'])
    print('RMSE:', test_results['MSE'] ** 0.5)
    print('R2:', test_results['R2'])
