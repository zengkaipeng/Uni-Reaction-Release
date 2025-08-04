import torch
import argparse
import json
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path', required=True, type=str,
        help='the path containing results'
    )
    parser.add_argument(
        '--beams', type=int, default=10,
        help='the number of beams for beam search'
    )
    args = parser.parse_args()

    results = {
        'catalyst': [], 'reagent': [],
        'solvent': []
    }
    to_display = [1, 3, 5, 10, 20, 30, 50]

    with open(args.file_path) as Fin:
        INFO = json.load(Fin)

    real_answer = {
        'reagent': {}, 'catalyst': {},
        'solvent': {}
    }
    for k, v in INFO['rxn2gt'].items():
        real_answer['catalyst'][k] = set(x[0] for x in v)
        real_answer['solvent'][k] = set((x[1], x[2]) for x in v)
        real_answer['reagent'][k] = set((x[3], x[4]) for x in v)

    for line in tqdm(INFO['answer']):
        catalyst_line = np.zeros(args.beams)
        reagent_line = np.zeros(args.beams)
        solvent_line = np.zeros(args.beams)
        for idx, (prob, res) in enumerate(line['prob_answer']):
            catalyst = res[0]
            solvent = (res[1], res[2])
            reagent = (res[3], res[4])

            if catalyst in real_answer['catalyst'][line['query_key']]:
                catalyst_line[idx:] = 1
            if reagent in real_answer['reagent'][line['query_key']]:
                reagent_line[idx:] = 1
            if solvent in real_answer['solvent'][line['query_key']]:
                solvent_line[idx:] = 1

        results['catalyst'].append(catalyst_line)
        results['reagent'].append(reagent_line)
        results['solvent'].append(solvent_line)

    results = {
        k: np.mean(np.stack(v, axis=0), axis=0)
        for k, v in results.items()
    }
    print('[Model Config]')
    print(INFO['args'])

    for k, v in results.items():
        print(f'[{k}]')
        for p in to_display:
            if p <= args.beams:
                print(f'[top-{p}]', float(v[p - 1]))
            else:
                break
