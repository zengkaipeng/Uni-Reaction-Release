import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import heapq

import argparse
import os


def generate_rxn_infos(df):
    fwd_template = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"
    methylaniline = "Cc1ccc(N)cc1"
    # pd_catalyst = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
    # pd_catalyst = "O=S(=O)(O[Pd]1Nc2ccccc2-c2ccccc21)C(F)(F)F"
    # pd_catalyst = 'O=S(=O)(O[Pd]1c2ccccc2-c2ccccc2N~1)C(F)(F)F'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles(
        'O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'
    ))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)

    curr_amap = 1
    for idx, p in enumerate(methylaniline_mol.GetAtoms()):
        p.SetAtomMapNum(curr_amap)
        curr_amap += 1

    reas, prds, mprxns, cata = [], [], [], []
    for i, row in df.iterrows():
        cata.append(pd_catalyst)
        reac = Chem.MolFromSmiles(row["Aryl halide"])
        for idx, p in enumerate(reac.GetAtoms()):
            p.SetAtomMapNum(curr_amap + idx)

        neis_u = {}
        for p in methylaniline_mol.GetAtoms():
            amp = p.GetAtomMapNum()
            neis_u[amp] = set()
            for x in p.GetNeighbors():
                neis_u[amp].add((x.GetSymbol(), x.GetAtomMapNum()))

        for p in reac.GetAtoms():
            amp = p.GetAtomMapNum()
            neis_u[amp] = set()
            for x in p.GetNeighbors():
                neis_u[amp].add((x.GetSymbol(), x.GetAtomMapNum()))

        rxn_products = rxn.RunReactants((reac, methylaniline_mol))
        rxn_products_smiles = set([
            Chem.MolToSmiles(mol[0]) for mol in rxn_products
        ])
        assert len(rxn_products_smiles) == 1

        prod_mol = Chem.MolFromSmiles(list(rxn_products_smiles)[0])

        while True:
            am2idx, unk_n = {}, {}
            for p in prod_mol.GetAtoms():
                amp = p.GetAtomMapNum()
                if amp == 0:
                    continue
                unk_n[amp] = set()
                am2idx[amp] = p.GetIdx()
                for x in p.GetNeighbors():
                    x_am = x.GetAtomMapNum()
                    if x_am != 0:
                        tkey = (x.GetSymbol(), x_am)
                        if tkey in neis_u[amp]:
                            neis_u[amp].remove(tkey)
                    else:
                        unk_n[amp].add(x.GetIdx())

            for am, v in unk_n.items():
                if len(v) != 1:
                    continue
                assert len(neis_u[am]) == 1, 'Failed to Match UnMapped' + \
                    f" Atoms, uncertain atom mapping: {am}, {neis_u[am]}"
                un_atom = prod_mol.GetAtomWithIdx(list(unk_n[am])[0])
                un_sym, un_am = list(neis_u[am])[0]
                if un_sym == un_atom.GetSymbol():
                    # Matched!
                    un_atom.SetAtomMapNum(un_am)

            flag = True
            for p in prod_mol.GetAtoms():
                if p.GetAtomMapNum() == 0:
                    flag = False
            if flag:
                break

        product = Chem.MolToSmiles(prod_mol)
        for k in prod_mol.GetAtoms():
            k.ClearProp('molAtomMapNumber')

        cano_prod = Chem.MolToSmiles(prod_mol)
        cano_reac = f'{row["Aryl halide"]}.{methylaniline}'

        mapprx = '{}.{}>>{}'.format(
            Chem.MolToSmiles(methylaniline_mol),
            Chem.MolToSmiles(reac),
            product
        )

        reas.append(cano_reac)
        prds.append(cano_prod)
        mprxns.append(mapprx)

    df['catalyst'] = cata
    df['reactants'] = reas
    df['product'] = prds
    df['mapped_rxn'] = mprxns

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file', type=str, required=True,
        help='the path of original excel'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='the data folder containing datas of different split'
    )
    args = parser.parse_args()

    sheet_names = [
        "FullCV_01",
        "FullCV_02",
        "FullCV_03",
        "FullCV_04",
        "FullCV_05",
        "FullCV_06",
        "FullCV_07",
        "FullCV_08",
        "FullCV_09",
        "FullCV_10",
        "Test1",
        "Test2",
        "Test3",
        "Test4"
    ]

    test_size = [1187] * 10 + [898, 900, 897, 900]

    for idx, name in enumerate(sheet_names):
        org_df = pd.read_excel(args.input_file, sheet_name=name)
        new_data = generate_rxn_infos(org_df)
        outx = os.path.join(args.output_dir, name)
        if not os.path.exists(outx):
            os.makedirs(outx)

        ts = test_size[idx]

        new_data.to_csv(os.path.join(outx, f'{name}.csv'))
        new_data.iloc[-ts:, :].to_csv(os.path.join(outx, 'test.csv'))

        train_data = new_data.iloc[:-ts, :]
        train_data.to_csv(os.path.join(outx, 'train_src.csv'))
        val_data = train_data.sample(frac=0.1, random_state=42)
        val_data.to_csv(os.path.join(outx, 'val.csv'))
        train_data = train_data.drop(val_data.index)
        train_data.to_csv(os.path.join(outx, 'train.csv'))
