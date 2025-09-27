import argparse
import json
import torch
import pickle
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.chemistry_parse import canonical_rxn
from utils.inference import beam_search_condition
from utils.data_utils import load_uspto_condition_inference, count_parameters
from utils.Dataset import seq_inf_fn

from model import (
    RAlignEncoder, DualGATEncoder, TranDec,
    USPTOConditionModel, PositionalEncoding
)


def permute_five(a, b, c, d, e):
    return [
        (a, b, c, d, e), (a, b, c, e, d),
        (a, c, b, d, e), (a, c, b, e, d)
    ]


def ensure_folder_exists(file_path):
    absolute_path = os.path.abspath(file_path)
    folder_path = os.path.dirname(absolute_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset'
    )
    parser.add_argument(
        '--dim', type=int, default=384,
        help='the number of dim for model'
    )
    parser.add_argument(
        '--heads', type=int, default=6,
        help='the number of heads for model'
    )
    parser.add_argument(
        '--n_layer', type=int, default=6,
        help='the number of layers of the model'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )
    parser.add_argument(
        '--local_heads', type=int, default=3,
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
        help='the file path to store results'
    )
    parser.add_argument(
        '--beam_size', type=int, default=10,
        help='the size for beam searching'
    )
    parser.add_argument(
        '--remove_align', action='store_true',
        help='remove the alignment layer in encoder'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='the batch size for inference'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='the num_workers in dataloader'
    )
    parser.add_argument(
        '--decoder_layer', type=int, default=-1,
        help='the number of decoder layers'
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    ensure_folder_exists(args.output_file)

    with open(args.token_ckpt, 'rb') as Fin:
        remap = pickle.load(Fin)
    idx2name = {v: k for k, v in remap.items()}
    begin_idx = remap['<CLS>']

    testset = load_uspto_condition_inference(args.data_path, remap)

    loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        collate_fn=seq_inf_fn, num_workers=args.num_workers
    )

    if args.remove_align:
        encoder = DualGATEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=0,
            negative_slope=args.negative_slope, update_last_edge=False
        )
    else:
        encoder = RAlignEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=0,
            negative_slope=args.negative_slope, update_last_edge=False
        )
    dec_layer = args.n_layer if args.decoder_layer <= 0 else args.decoder_layer
    decoder = TranDec(
        n_layers=dec_layer, emb_dim=args.dim, heads=args.heads,
        dropout=0, dim_ff=args.dim << 1
    )
    pos_env = PositionalEncoding(args.dim, 0, maxlen=50)

    model = USPTOConditionModel(
        encoder=encoder, decoder=decoder, pe=pos_env,
        n_words=len(remap), dim=args.dim
    ).to(device)

    model_weight = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()
    num_params = count_parameters(model)

    prediction_results, rxn2gt, dis_res = [], {}, 0

    for reac, prod, raw_info, labels in tqdm(loader):
        answers = beam_search_condition(
            model=model, reac=reac, prod=prod, device=device,
            begin_idx=begin_idx, beams=args.beam_size,
            total_heads=args.heads, local_heads=args.local_heads
        )
        for idx, rxn in enumerate(raw_info):
            key = canonical_rxn(rxn)
            if key not in rxn2gt:
                rxn2gt[key] = []
            rxn2gt[key].extend(permute_five(*labels[idx]))
            prediction_results.append({
                'query': rxn, 'query_key': key,
                'prob_answer': answers[idx]
            })
        dis_res += len(answers)

        if dis_res >= args.save_every:
            with open(args.output_file, 'w') as Fout:
                json.dump({
                    'rxn2gt': rxn2gt,
                    'answer': prediction_results,
                    'remap': remap,
                    'idx2name': idx2name,
                    'args': args.__dict__,
                    "num_parameters": num_params
                }, Fout, indent=4)
            dis_res %= args.save_every

    with open(args.output_file, 'w') as Fout:
        json.dump({
            'rxn2gt': rxn2gt,
            'answer': prediction_results,
            'remap': remap,
            'idx2name': idx2name,
            'args': args.__dict__,
            "num_parameters": num_params
        }, Fout, indent=4)
