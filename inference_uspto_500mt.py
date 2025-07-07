import os
import torch
import pickle
import argparse
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.chemistry_parse import canonical_rxn
from utils.inference import beam_search_500mt
from utils.Dataset import gen_inf_fn
from utils.data_utils import load_uspto_mt500_inference, count_parameters

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
        '--local_heads', type=int, default=0,
        help='the number of local heads in attention'
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
        '--output_file', type=str, required=True,
        help='the path for output results'
    )
    parser.add_argument(
        '--beam_size', type=int, default=10,
        help='the size for beam searching'
    )
    parser.add_argument(
        '--max_len', type=int, default=500,
        help='the maximal sequence number for inference'
    )
    parser.add_argument(
        '--remove_align', action='store_true',
        help='remove the alignment in encoder'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='the batch size for inference'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='the number of workers for inference'
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

    num_params = count_parameters(model)

    test_set = load_uspto_mt500_inference(args.data_path, remap)
    loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=gen_inf_fn
    )

    prediction_results, save_res, rxn2gt = [], 0, {}

    for reac, prod, raw_info, labels in tqdm(loader):
        answers = beam_search_500mt(
            model=model, reac=reac, prod=prod, device=device, toker=remap,
            begin_token='<CLS>',  max_len=args.max_len, beams=args.beam_size,
            total_heads=args.heads, local_heads=args.local_heads,
            end_token='<END>', pad_token='<PAD>'
        )
        for idx, rxn in enumerate(raw_info):
            key = canonical_rxn(rxn)
            if key not in rxn2gt:
                rxn2gt[key] = []
            rxn2gt[key].append(labels[idx])
            prediction_results.append({
                'query': rxn, 'query_key': key,
                'prob_answer': answers[idx],
            })

        save_res += len(answers)
        if save_res >= args.save_every:
            outx = {
                'rxn2gt': rxn2gt, 'args': args.__dict__,
                'answer': prediction_results, 'num_parameters': num_params
            }
            with open(args.output_file, 'w') as Fout:
                json.dump(outx, Fout, indent=4)
            save_res %= args.save_every

    with open(args.output_file, 'w') as Fout:
        outx = {
            'rxn2gt': rxn2gt, 'answer': prediction_results,
            'args': args.__dict__, "num_parameters": num_params
        }
        with open(out_file, 'w') as Fout:
            json.dump(outx, Fout, indent=4)
