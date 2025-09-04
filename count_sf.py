import rdkit
from rdkit import Chem
import pandas
from tqdm import tqdm


def is_sulfonamide_bond_formed(reaction_smiles):
    """
    验证反应是否生成了磺酰胺键
    
    参数:
    reaction_smiles (str): 带有atom mapping的反应SMILES字符串，格式为"reactant>>product"
    
    返回:
    dict: 包含验证结果和详细信息的字典
    """
    try:
        # 分割反应物和产物
        if ">>" not in reaction_smiles:
            return {"valid": False, "error": "Invalid reaction format. Expected 'reactant>>product'"}
        
        reactants_smiles, products_smiles = reaction_smiles.split(">>")
        
        # 解析反应物和产物
        reactants = Chem.MolFromSmiles(reactants_smiles)
        products = Chem.MolFromSmiles(products_smiles)
        
        if reactants is None:
            return {"valid": False, "error": "Failed to parse reactants SMILES"}
        if products is None:
            return {"valid": False, "error": "Failed to parse products SMILES"}
        
        # 在产物中查找磺酰胺键 (S(=O)(=O)-N)
        sulfonamide_bond_found = False
        sulfonamide_atoms = []
        
        # 遍历产物中的所有原子
        for atom in products.GetAtoms():
            # 寻找硫原子 (S)
            if atom.GetSymbol() == "S":
                # 检查硫原子是否连接两个氧原子(双键)和一个氮原子(单键)
                o_double_count = 0
                n_single = None
                
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    bond_type = bond.GetBondType()
                    
                    if other_atom.GetSymbol() == "O" and bond_type == Chem.BondType.DOUBLE:
                        o_double_count += 1
                    elif other_atom.GetSymbol() == "N" and bond_type == Chem.BondType.SINGLE:
                        n_single = other_atom
                
                # 如果找到S(=O)(=O)-N结构，则确认磺酰胺键
                if o_double_count >= 2 and n_single is not None:
                    sulfonamide_bond_found = True
                    # 记录磺酰胺键涉及的原子映射编号
                    s_map = atom.GetProp("molAtomMapNumber") if atom.HasProp("molAtomMapNumber") else "?"
                    n_map = n_single.GetProp("molAtomMapNumber") if n_single.HasProp("molAtomMapNumber") else "?"
                    sulfonamide_atoms.append({"S": s_map, "N": n_map})
        
        # 检查磺酰胺键是否是新形成的
        new_bond_formed = False
        if sulfonamide_bond_found:
            # 在反应物中检查S和N原子是否已经成键
            for sulfonamide in sulfonamide_atoms:
                s_map = sulfonamide["S"]
                n_map = sulfonamide["N"]
                
                # 查找反应物中具有相同映射编号的原子
                s_reactant = None
                n_reactant = None
                
                for atom in reactants.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        map_num = atom.GetAtomMapNum()
                        if map_num == s_map:
                            s_reactant = atom
                        elif map_num == n_map:
                            n_reactant = atom
                
                # 如果找到了S和N原子，检查它们在反应物中是否已经成键
                if s_reactant and n_reactant:
                    # 获取两个原子之间的键
                    bond = reactants.GetBondBetweenAtoms(s_reactant.GetIdx(), n_reactant.GetIdx())
                    if bond is None:
                        new_bond_formed = True
                        break
                else:
                    # 如果找不到映射的原子，假设是新形成的键
                    new_bond_formed = True
                    break
        
        return {
            "valid": True,
            "is_sulfonamide_reaction": sulfonamide_bond_found and new_bond_formed,
            "sulfonamide_bond_detected": sulfonamide_bond_found,
            "new_bond_formed": new_bond_formed,
            "sulfonamide_atoms": sulfonamide_atoms
        }
    
    except Exception as e:
        return {"valid": False, "error": str(e)}


if __name__ == '__main__':
    data_path = '../RAlign/data/USPTO/uspto_condition_mapped.csv'
    raw_info = pandas.read_csv(data_path)
    raw_info = raw_info.fillna('')
    raw_info = raw_info.to_dict('records')

    all_datas = []

    for i, element in enumerate(tqdm(raw_info)):
        rxn_type = element['dataset']
        if rxn_type != 'train':
            continue
        all_datas.append(element['mapped_rxn'])


    answer, cnt = 0, 0
    for x in tqdm(all_datas):
        result = is_sulfonamide_bond_formed(x)
        if result['is_sulfonamide_reaction']:
            answer += 1
        cnt += 1

    print(answer, cnt, 1. * answer / cnt)





