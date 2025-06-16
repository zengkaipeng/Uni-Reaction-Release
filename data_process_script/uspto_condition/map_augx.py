from rxnmapper import RXNMapper, BatchedMapper
import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--bs', type=int, default=32)
    args = parser.parse_args()
    model = BatchedMapper(batch_size=args.bs)

    with open(os.path.join(args.dir, args.input)) as Fin:
        INFO = json.load(Fin)
    idxx, ress = [], []
    for k, v in INFO.items():
        idxx.append(k)
        ress.append(v)
    results = list(model.map_reactions_with_info(ress))
    out_res = {idx: results[t] for t, idx in enumerate(idxx)}

    Fo = open(os.path.join(args.dir, f'out_{args.input}'), 'w')
    json.dump(out_res, Fo)
    Fo.close()
