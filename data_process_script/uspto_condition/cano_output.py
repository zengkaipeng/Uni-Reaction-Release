import rdkit
from rdkit import Chem
import os
import argparse
import pandas
import json
from tqdm import tqdm


def cano_x(x):
    mol = Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path', required=True, type=str,
        help='the path containing the uspto_condition_data'
    )
    parser.add_argument(
        '--vocab_path', type=str, default='',
        help='the path to store the condition_to_idx mapping'
    )
    args = parser.parse_args()

    if args.vocab_path == '':
        data_dir = os.path.dirname(os.path.abspath(args.file_path))
        args.vocab_path = os.path.join(data_dir, 'label_to_idx.json')

    raw_info = pandas.read_csv(args.file_path)
    raw_info = raw_info.fillna('')
    all_raw_mols, mapper, raw_to_cano = set(), dict(), dict()
    labels_list = ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']

    for idx, x in raw_info.iterrows():
        all_raw_mols.update([x[t] for t in labels_list])

    for x in tqdm(all_raw_mols):
        raw_to_cano[x] = canoed_x = cano_x(x)
        mapper[canoed_x] = mapper.get(canoed_x, len(mapper))

    raw_info[labels_list].replace(raw_to_cano)
    raw_info.to_csv(args.file_path, index=False)

    print('before_cano: {} after_cano: {}'.format(
        len(raw_to_cano), len(mapper)
    ))

    with open(args.vocab_path, 'w') as Fout:
        json.dump(mapper, Fout, indent=4)
