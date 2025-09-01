import argparse
import json
import torch
import pickle
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.chemistry_parse import canonical_rxn, canonical_smiles
from utils.inference import beam_search_condition
from utils.data_utils import load_uspto_condition_inference_json, count_parameters
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
        '--logdir', '-l', type=str, default='log/uspto/RAlign_clean_results/condition/1751684873.4615476',
        help='the directory for logging'
    )
    parser.add_argument(
        '--data_path', '-d', type=str, required=True,
        help='data path of the dataset'
    )
    parser.add_argument(
        '--has-label', action='store_true',
        help='whether the dataset has labels'
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
        '--batch_size', type=int, default=128,
        help='the batch size for inference'
    )
    parser.add_argument(
        '--decoder_layer', type=int, default=-1,
        help='the number of decoder layers'
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

    output_file = os.path.join(logdir, 'pred_' + os.path.basename(infer_args.data_path))


    with open(os.path.join(logdir, 'token.pkl'), 'rb') as Fin:
        remap = pickle.load(Fin)
    idx2name = {v: k for k, v in remap.items()}
    begin_idx = remap['<CLS>']

    testset = load_uspto_condition_inference_json(infer_args.data_path, remap, has_label=infer_args.has_label)

    loader = DataLoader(
        testset, batch_size=infer_args.batch_size, shuffle=False,
        collate_fn=seq_inf_fn, num_workers=4
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
    dec_layer = args.n_layer if infer_args.decoder_layer <= 0 else infer_args.decoder_layer
    decoder = TranDec(
        n_layers=dec_layer, emb_dim=args.dim, heads=args.heads,
        dropout=0, dim_ff=args.dim << 1
    )
    pos_env = PositionalEncoding(args.dim, 0, maxlen=50)

    model = USPTOConditionModel(
        encoder=encoder, decoder=decoder, pe=pos_env,
        n_words=len(remap), dim=args.dim
    ).to(device)

    model_weight = torch.load(os.path.join(logdir, 'model.pth'), map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()
    num_params = count_parameters(model)

    prediction_results, rxn2gt, dis_res = [], {}, 0

    local_heads = (args.heads >> 1) * args.local_global if hasattr(args, 'local_global') else args.local_heads

    for batch_data in tqdm(loader):
        reac, prod, raw_info, labels = batch_data
        
        answers = beam_search_condition(
            model=model, reac=reac, prod=prod, device=device,
            begin_idx=begin_idx, beams=infer_args.beam_size,
            total_heads=args.heads, local_heads=local_heads
        )
        for idx, rxn in enumerate(raw_info):
            key = canonical_rxn(rxn)
            if infer_args.has_label:
                true_ans = permute_five(*labels[idx])
            else:
                true_ans = [canonical_smiles(l) for l in labels[idx]]
            answer = answers[idx]
            pred = []
            for ans in answer:
                pred.append(
                    {
                        'score': ans[0],
                        'pred_smiles': [idx2name[_i] for _i in ans[1]],
                    }
                )
            prediction_results.append({
                'query': rxn, 'query_key': key,
                'label': true_ans,
                'prob_answer': [[idx2name[_i] for _i in ans[1]] for ans in answer],
                'prob_scores': [ans[0] for ans in answer],
            })
        dis_res += len(answers)

    with open(output_file, 'w') as Fout:
        json.dump({
            'answer': prediction_results,
            'idx2name': idx2name,
            'args': args.__dict__,
            "num_parameters": num_params
        }, Fout, indent=4)

    print('pred saved in ', output_file)