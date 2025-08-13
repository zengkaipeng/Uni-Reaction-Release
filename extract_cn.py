import torch
import os
import time
import argparse
import json
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_utils import load_cn_yield_one, fix_seed
from utils.tensor_utils import generate_local_global_mask
from utils.Dataset import cn_colfn
from model import CNYieldModel, RAlignEncoder, build_cn_condition_encoder


def eval_and_extract(loader, model, device, total_heads=None, local_heads=0):
    model, ytrue, ypred = model.eval(), [], []
    for reac, prod, reag, label in tqdm(loader):
        reac, prod, reag = reac.to(device), prod.to(device), reag.to(device)
        if local_heads > 0:
            assert total_heads is not None, "require nheads for mask gen"
            cross_mask = generate_local_global_mask(
                reac, prod, 1, total_heads, local_heads
            )
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

    result['label'] = ytrue.tolist()
    result['prediction'] = ypred.tolist()
    return result


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
        '--output_dir', type=str, required=True,
        help='the json file containing results'
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
        '--local_heads', type=int, default=0,
        help='the number of local heads in attention'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='the checkpoint for inference'
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
        prod_num_keys={}, reac_num_keys={}, dropout=0,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_cn_condition_encoder(
        config=condition_config, dropout=0
    )

    model = CNYieldModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=0, heads=args.heads
    ).to(device)

    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)
    model = model.eval()

    result = eval_and_extract(
        test_loader, model, device, total_heads=args.heads,
        local_heads=args.local_heads
    )

    print('MAE:', result['MAE'])
    print('MSE:', result['MSE'])
    print('R2:', result['R2'])
    with open(args.output_dir, 'w') as Fout:
        json.dump(result, Fout, indent=4)
