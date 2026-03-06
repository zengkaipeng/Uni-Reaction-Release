import torch
import argparse
import json
import rdkit.RDLogger as rdl

from torch.utils.data import DataLoader

from utils.data_utils import load_uspto_yield
from utils.training import eval_uspto_yield
from utils.uspto_ds import ds_col_fn

from model import (
    USPTOYield, RAlignEncoder, ReagentEncoderWithAmount, MultiNumEmb
)


# Silence RDKit warnings for cleaner logs
logger = rdl.logger()
logger.setLevel(rdl.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser for USPTO-yield inference')
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='path to the JSONL dataset file'
    )
    parser.add_argument(
        '--checkpoint', required=True, type=str,
        help='checkpoint path for model weights'
    )
    parser.add_argument(
        '--output_path', required=True, type=str,
        help='path to write JSON results'
    )
    parser.add_argument(
        '--part', type=str, default='test',
        help="dataset split to evaluate: 'train', 'val', or 'test'"
    )
    parser.add_argument(
        '--dim', type=int, default=256,
        help='model hidden dimension'
    )
    parser.add_argument(
        '--heads', type=int, default=8,
        help='number of attention heads'
    )
    parser.add_argument(
        '--n_layer', type=int, default=6,
        help='number of encoder layers'
    )
    parser.add_argument(
        '--num_worker', type=int, default=8,
        help='dataloader worker count'
    )
    parser.add_argument(
        '--bs', type=int, default=256,
        help='batch size for inference'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='negative slope for leaky relu'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='device id, negative for CPU'
    )
    parser.add_argument(
        '--local_heads', type=int, default=4,
        help='number of local heads in attention'
    )
    parser.add_argument(
        '--amount_class', type=int, default=50,
        help='class count for amount embedding; <=0 disables'
    )
    parser.add_argument(
        '--temperature_class', type=int, default=50,
        help='class count for temperature embedding; <=0 disables'
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    test_set = load_uspto_yield(args.data_path, part=args.part)
    if isinstance(test_set, dict):
        raise ValueError(
            "part must be 'train', 'val', or 'test' for inference"
        )
    test_loader = DataLoader(
        test_set, batch_size=args.bs, shuffle=False,
        collate_fn=ds_col_fn, num_workers=args.num_worker
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
        negative_slope=args.negative_slope, dropout=0,
        amount_encoder=amount_encoder, num_emb_dim=args.dim,
        use_amount=args.amount_class > 0, merge_mode='sep'
    )

    encoder = RAlignEncoder(
        n_layer=args.n_layer, emb_dim=args.dim, edge_dim=args.dim,
        heads=args.heads, reac_batch_infos=condition_infos,
        prod_batch_infos=condition_infos, prod_num_keys=prod_num_conditions,
        reac_num_keys=reac_num_conditions, dropout=0,
        negative_slope=args.negative_slope, update_last_edge=False,
        fusion_order='film_first'
    )

    model = USPTOYield(
        encoder=encoder, condition_encoder=condition_encoder,
        amount_encoder=amount_encoder, temperature_encoder=temperature_encoder,
        dim=args.dim, dropout=0, heads=args.heads
    ).to(device)

    print('[INFO] loading model from', args.checkpoint)
    weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weight)

    test_results = eval_uspto_yield(
        test_loader, model, device, total_heads=args.heads,
        local_heads=args.local_heads, return_raw=True
    )

    with open(args.output_path, 'w') as f:
        json.dump(test_results, f, indent=4)

    print('MAE:', test_results['MAE'])
    print('RMSE:', test_results['RMSE'])
    print('R2:', test_results['R2'])
