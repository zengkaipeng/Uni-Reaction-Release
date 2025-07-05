import time
import os
import torch
import pickle
import argparse
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.chemistry_parse import canonical_rxn
from utils.inference import inference_500mt

from model import (
    TranDec, USPTO500MTModel, PositionalEncoding,
    RAlignEncoder, DualGATEncoder
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--checkpoint', required=True, type=str,
        help='the path for checkpoint'
    )
    parser.add_argument(
        '--token_ckpt', required=True, type=str,
        help='the path for tokenizer remapper'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='the step size for saving results'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='the path for output results'
    )
    parser.add_argument(
        '--beam_size', type=int, default=10,
        help='the size for beam searching'
    )
    parser.add_argument(
        '--max_len', type=int, default=300,
        help='the maximal sequence number for inference'
    )
    parser.add_argument(
        '--remove_align', action='store_true',
        help='remove the alignment in encoder'
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.token_ckpt, 'rb') as Fin:
        remap = pickle.load(Fin)

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
    decoder = TranDec(
        n_layers=args.n_layer, emb_dim=args.dim, heads=args.heads,
        dropout=args.dropout, dim_ff=args.dim << 1
    )
    pos_env = PositionalEncoding(args.dim, args.dropout, maxlen=2000)

    model = USPTO500MTModel(
        encoder=encoder, decoder=decoder, pe=pos_env,
        n_words=len(remap), dim=args.dim
    ).to(device)
    model_weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()

    prediction_results = []
    rxn2gt = {}

    with open(args.data_path) as Fin:
        raw_info = json.load(Fin)

    out_file = os.path.join(args.output_dir, f'answer-{time.time()}.json')

    
    for idx, line in enumerate(tqdm(raw_info)):
        query_rxn = line['new_mapped_rxn']
        key = canonical_rxn(query_rxn)
        if args.mode == 'prediction':
            gt = sorted([remap[x] for x in line['reagent_list']])
        else:
            gt = '.'.join(line['reagent_list'])

        if key not in rxn2gt:
            rxn2gt[key] = []
        rxn2gt[key].append(gt)

        if args.mode == 'prediction':
            results = beam_search_500_mt_pred(
                model=model, remap=remap, mapped_rxn=query_rxn,
                device=device, max_len=args.max_len, size=args.beam_size,
                heads=args.heads, local_glocal=args.local_global,
                begin_token='<CLS>', end_token='<END>'
            )
        else:
            results = beam_search_500_mt_gen(
                model=model, tokenizer=remap, mapped_rxn=query_rxn,
                device=device, max_len=args.max_len, size=args.beam_size,
                heads=args.heads, local_glocal=args.local_global,
                begin_token='<CLS>', end_token='<END>'
            )

        prediction_results.append({
            'query': query_rxn,
            'prob_answer': results,
            'query_key': key
        })

        if len(prediction_results) % args.save_every == 0:
            outx = {
                'rxn2gt': rxn2gt,
                'answer': prediction_results,
                'args': args.__dict__
            }

            if args.mode == 'prediction':
                outx['remap'] = remap
                outx['idx2name'] = idx2name

            with open(out_file, 'w') as Fout:
                json.dump(outx, Fout, indent=4)

    with open(out_file, 'w') as Fout:
        outx = {
            'rxn2gt': rxn2gt,
            'answer': prediction_results,
            'args': args.__dict__
        }

        if args.mode == 'prediction':
            outx['remap'] = remap
            outx['idx2name'] = idx2name

        with open(out_file, 'w') as Fout:
            json.dump(outx, Fout, indent=4)
