import os
import torch
import pickle
import argparse
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.chemistry_parse import canonical_rxn
from utils.inference import beam_search_500mt
from utils.Dataset import seq_inf_fn
from utils.data_utils import load_uspto_mt500_inference_json, count_parameters, smi_tokenizer

from model import (
    TranDec, USPTO500MTModel, PositionalEncoding,
    RAlignEncoder, DualGATEncoder
)


def ensure_folder_exists(file_path):
    absolute_path = os.path.abspath(file_path)
    folder_path = os.path.dirname(absolute_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', '-l', type=str, default='log/uspto/RAlign_clean_results/500mt/1751389561.5205266',
        help='the directory for logging'
    )
    parser.add_argument(
        '--data_path', '-d', type=str, required=True,
        help='data path of the dataset'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='the device id for traiing, negative for cpu'
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
        '--batch_size', type=int, default=128,
        help='the batch size for inference'
    )
    parser.add_argument(
        '--first-label-hint', '-flh', action='store_true',
        help='use first label as hint'
    )
    infer_args = parser.parse_args()
    logdir = infer_args.logdir
    with open(os.path.join(logdir, 'log.json'), 'r') as fin:
        log_info = json.load(fin)

    args = argparse.Namespace(**log_info['args'])    

    if torch.cuda.is_available() and infer_args.device >= 0:
        device = torch.device(f'cuda:{infer_args.device}')
    else:
        device = torch.device('cpu')

    output_file = os.path.join(logdir, 'first_label_hint_' * infer_args.first_label_hint + 'pred_' + os.path.basename(infer_args.data_path))

    with open(os.path.join(logdir, 'token.pkl'), 'rb') as Fin:
        remap = pickle.load(Fin)

    if args.remove_align:
        encoder = DualGATEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=0, update_last_edge=False,
            negative_slope=args.negative_slope
        )
    else:
        encoder = RAlignEncoder(
            emb_dim=args.dim, n_layer=args.n_layer, heads=args.heads,
            edge_dim=args.dim, dropout=0, update_last_edge=False,
            negative_slope=args.negative_slope
        )
    decoder = TranDec(
        n_layers=args.n_layer, emb_dim=args.dim, heads=args.heads,
        dropout=0, dim_ff=args.dim << 1
    )
    pos_env = PositionalEncoding(args.dim, 0, maxlen=2000)

    model = USPTO500MTModel(
        encoder=encoder, decoder=decoder, pe=pos_env,
        n_words=len(remap), dim=args.dim
    ).to(device)
    model_weight = torch.load(os.path.join(logdir, 'model.pth'), map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()

    test_set = load_uspto_mt500_inference_json(infer_args.data_path)
    loader = DataLoader(
        test_set, batch_size=infer_args.batch_size, shuffle=False,
        num_workers=4, collate_fn=seq_inf_fn
    )

    num_params = count_parameters(model)
    prediction_results, save_res, rxn2gt = [], 0, {}

    local_heads = (args.heads >> 1) * args.local_global if hasattr(args, 'local_global') else args.local_heads
    print('local_heads: ', local_heads)
    for reac, prod, raw_info, labels in tqdm(loader):
        if infer_args.first_label_hint:
            prefix = [(smi_tokenizer(label[0]) + ['`']) if len(label) > 0 else [] for label in labels]
        else:
            prefix = None
        answers = beam_search_500mt(
            model=model, reac=reac, prod=prod, device=device, toker=remap,
            begin_token='<CLS>',  max_len=infer_args.max_len, beams=infer_args.beam_size,
            total_heads=args.heads, local_heads=local_heads,
            end_token='<END>', pad_token='<PAD>', prefix=prefix
        )
        for idx, rxn in enumerate(raw_info):
            key = canonical_rxn(rxn)
            
            prediction_results.append({
                'query': rxn, 'query_key': key,
                'labels': labels[idx],
                'prob_answer': [item[1] for item in answers[idx]],
                'prob_scores': [item[0] for item in answers[idx]],
            })

        save_res += len(answers)

    with open(output_file, 'w') as Fout:
        outx = {
            'answer': prediction_results,
            'args': args.__dict__, "num_parameters": num_params
        }
        json.dump(outx, Fout, indent=4)

    print('prediction saved in ', output_file)