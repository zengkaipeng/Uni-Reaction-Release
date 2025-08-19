import torch
import os
import time
import argparse
import json
import pandas as pd

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils.data_utils import load_ru_one, fix_seed, count_parameters
from utils.training import train_mol_yield, eval_mol_yield
from utils.Dataset import cn_colfn

from model import CNYieldModel, RAlignEncoder, build_cn_condition_encoder, RUConditionEncoder



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

    test_set = load_ru_one(
        args.data_path, cmd_args.part, condition_config['data_type']
    )

    val_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=cn_colfn, num_workers=args.num_worker
    )

    if condition_config['mode'] == 'independent':
    #     condition_infos = {
    #         'mixed': {
    #             'dim': condition_config['dim'],
    #             'heads': args.heads
    #         }
    #     }
    # elif condition_config['mode'] == 'mix-catalyst-ligand':
    #     condition_infos = {
    #         k: {'dim': condition_config['dim'], 'heads': args.heads}
    #         for k in ['additive', 'base', 'catalyst and ligand']
    #     }
    # else:
        condition_infos = {
            k: {'dim': condition_config['dim'], 'heads': args.heads}
            for k in ['ligand', 'solvent', 'additive', 'catalyst']
        }

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim,  edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos if args.condition_both else {},
        prod_num_keys={}, reac_num_keys={}, dropout=args.dropout,
        negative_slope=args.negative_slope, update_last_edge=False
    )

    condition_encoder = build_cn_condition_encoder(
        config=condition_config, dropout=args.dropout,
        condition_encoder=RUConditionEncoder
    )

    model = CNYieldModel(
        encoder=encoder, condition_encoder=condition_encoder,
        dim=args.dim, dropout=args.dropout, heads=args.heads
    ).to(device)

    print('[INFO] loading model from', log_dir)
    model.load_state_dict(
        torch.load(os.path.join(log_dir, 'model.pth'), map_location=device)
    )

    total_heads = args.heads
    if hasattr(args, 'local_heads'):
        local_heads = args.local_heads
    elif hasattr(args, 'local_global'):
        print('[INFO] OLD setting: local_global')
        local_heads = 0 if not args.local_global else total_heads >> 1
    else:
        local_heads = total_heads >> 1
        # assert False, 'local_heads or local_global must be specified'

    test_results = eval_mol_yield(
        val_loader, model, device, total_heads=total_heads, local_heads=local_heads,
        return_raw=True
    )
    df = pd.DataFrame({
        'prediction': test_results['prediction'],
        'label': test_results['label']
    })
    data_postfix = data_path.replace('/', '_').replace('\\', '_')
    result_csv = os.path.join(log_dir, f'pred_{cmd_args.part}.csv')
    df.to_csv(result_csv, index=False)
    result_json = result_csv.replace('.csv', '.json')

    with open(result_json, 'w') as f:
        json.dump(test_results, f, indent=4)