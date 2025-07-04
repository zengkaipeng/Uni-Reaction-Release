import torch
from utils.data_utils import load_uspto_condition, check_early_stop, fix_seed
import argparse
import time
import os
import pickle
from utils.Dataset import pred_fn
from torch.utils.data import DataLoader
from model import (
    RAlignEncoder, DualGATEncoder, TranDec,
    USPTOConditionModel, PositionalEncoding
)
from torch.optim.lr_scheduler import ExponentialLR
from utils.training import train_uspto_condition, eval_uspto_condition
import json


def make_dir(args):
    timestamp = time.time()
    detail_dir = os.path.join(args.base_log, f'{timestamp}')
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    log_dir = os.path.join(detail_dir, 'log.json')
    model_dir = os.path.join(detail_dir, 'model.pth')
    token_dir = os.path.join(detail_dir, 'token.pkl')
    return log_dir, model_dir, token_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for prediction model')
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset'
    )
    parser.add_argument(
        '--mapper_path', required=True, type=str,
        help='the path for label mapper'
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
        '--decoder_layer', type=int, default=-1,
        help='the layers of decoder, non-positive ' +
        "to set it the same as n_layer"
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
        '--epoch', type=int, default=200,
        help='the number for epochs for training'
    )
    parser.add_argument(
        '--base_log', type=str, default='log',
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
        '--local_global', action='store_true',
        help='use local global attention for decoder'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='the device id for traiing, negative for cpu'
    )
    parser.add_argument(
        '--early_stop', type=int, default=0,
        help='the number of epochs for checking early stop'
        ', ignored when less than 5'
    )
    parser.add_argument(
        '--step_start', type=int, default=10,
        help='the step to start lr decay'
    )
    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the random seed for training'
    )
    parser.add_argument(
        '--remove_align', action='store_true',
        help='remove the aligned layer for model'
    )

    args = parser.parse_args()

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    train_set, val_set, test_set, remap = \
        load_uspto_condition(args.data_path, args.mapper_path)

    log_dir, model_dir, token_dir = make_dir(args)

    train_loader = DataLoader(
        train_set, batch_size=args.bs, collate_fn=pred_fn,
        shuffle=True, num_workers=args.num_worker,
    )

    val_loader = DataLoader(
        val_set, batch_size=args.bs, collate_fn=pred_fn,
        shuffle=False, num_workers=args.num_worker
    )

    test_loader = DataLoader(
        test_set, batch_size=args.bs, collate_fn=pred_fn,
        shuffle=False, num_workers=args.num_worker
    )

    if args.remove_align:
        encoder = DualGATEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=args.dropout,
            negative_slope=args.negative_slope, update_last_edge=False
        )
    else:
        encoder = RAlignEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=args.dropout,
            negative_slope=args.negative_slope, update_last_edge=False
        )

    dec_layer = args.n_layer if args.decoder_layer <= 0 else args.decoder_layer
    decoder = TranDec(
        n_layers=dec_layer, emb_dim=args.dim, heads=args.heads,
        dropout=args.dropout, dim_ff=args.dim << 1
    )
    pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=50)

    model = USPTOConditionModel(
        encoder=encoder, decoder=decoder, pe=pos_env,
        n_words=len(remap), dim=args.dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sher = ExponentialLR(optimizer, gamma=args.lrgamma)

    log_info = {
        'args': args.__dict__, 'train_loss': [],
        'valid_metric': [], 'test_metric': []
    }

    with open(token_dir, 'wb') as Fout:
        pickle.dump(remap, Fout)

    with open(log_dir, 'w') as Fout:
        json.dump(log_info, Fout)

    best_pref, best_ep = None, None

    for ep in range(args.epoch):
        print(f'[INFO] training epoch {ep}')
        loss = train_uspto_condition(
            train_loader, model, optimizer, device, heads=args.heads,
            warmup=(ep < args.warmup), local_global=args.local_global
        )
        val_results = eval_uspto_condition(
            val_loader, model, device, heads=args.heads,
            local_global=args.local_global
        )
        test_results = eval_uspto_condition(
            test_loader, model, device, heads=args.heads,
            local_global=args.local_global
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

        if best_pref is None or val_results['overall'] > best_pref:
            best_pref, best_ep = val_results['overall'], ep
            torch.save(model.state_dict(), model_dir)

        if args.early_stop >= 5 and ep > max(10, args.early_stop):
            tx = log_info['valid_metric'][-args.early_stop:]
            keys = [
                'overall', 'catalyst', 'solvent1', 'solvent2',
                'reagent1', 'reagent2'
            ]
            tx = [[x[key] for x in tx] for key in keys]
            if check_early_stop(*tx):
                break

    print(f'[INFO] best acc epoch: {best_ep}')
    print(f'[INFO] best valid loss: {log_info["valid_metric"][best_ep]}')
    print(f'[INFO] best test loss: {log_info["test_metric"][best_ep]}')
