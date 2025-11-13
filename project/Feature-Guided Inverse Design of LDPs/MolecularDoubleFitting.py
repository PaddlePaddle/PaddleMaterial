#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem import MolSurf
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import copy
import warnings
from rdkit import RDLogger
from mordred import Calculator, descriptors

# 关闭 RDKit 日志
RDLogger.DisableLog('rdApp.*')


# In[35]:


df = pd.read_excel("NewMolecules.xlsx") 
original_smiles = df["SMILES"]

hetero_atoms = ["N", "O", "F"]
replacement_pool = ["C"] * 6 + ["N"] * 2 + ["O"] * 1 + ["F"] * 1  # C占60%

valid_new_smiles = []
original_to_new = {}


# # 1. 替换为X，然后随机替换X生成新SMILES

# In[36]:


from rdkit.Chem import rdchem

def mutate_bonds_to_double(mol, max_changes=2):
    bond_indices = [b.GetIdx() for b in mol.GetBonds() if b.GetBondType() == rdchem.BondType.SINGLE]
    if not bond_indices:
        return []
    mutated_smiles = []
    tried_combos = set()
    for _ in range(10):
        selected = tuple(sorted(random.sample(bond_indices, min(max_changes, len(bond_indices)))) )
        if selected in tried_combos:
            continue
        tried_combos.add(selected)
        mol_copy = copy.deepcopy(mol)
        for idx in selected:
            mol_copy.GetBondWithIdx(idx).SetBondType(rdchem.BondType.DOUBLE)
        try:
            Chem.SanitizeMol(mol_copy)
            smi = Chem.MolToSmiles(mol_copy, isomericSmiles=True)
            mutated_smiles.append(smi)
        except:
            continue
    return mutated_smiles

for smiles in original_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in hetero_atoms]
    seen = set()

    if not atom_indices:
        mutated = mutate_bonds_to_double(mol)
        for smi in mutated:
            if smi not in seen:
                seen.add(smi)
                valid_new_smiles.append(smi)
                original_to_new.setdefault(smiles, []).append(smi)
        continue

    for replacements in product(replacement_pool, repeat=len(atom_indices)):
        new_mol = copy.deepcopy(mol)
        for idx, new_symbol in zip(atom_indices, replacements):
            new_mol.GetAtomWithIdx(idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
        Chem.SanitizeMol(new_mol, catchErrors=True)
        new_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)
        if new_smiles in seen:
            continue
        seen.add(new_smiles)
        valid_new_smiles.append(new_smiles)
        original_to_new.setdefault(smiles, []).append(new_smiles)

        # 在替换基础上进一步加双键突变
        mutated = mutate_bonds_to_double(new_mol)
        for smi in mutated:
            if smi not in seen:
                seen.add(smi)
                valid_new_smiles.append(smi)
                original_to_new.setdefault(smiles, []).append(smi)


# # 2. 计算新SMILES的描述符

# In[37]:


# 描述符计算
valid_list = []
atsc1pe_list = []
mats2c_list = []
slogp_vsa2_list = []

calc = Calculator([descriptors.Autocorrelation])
valid_mols = []
for smi in valid_new_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        valid_mols.append(mol)
        valid_list.append(smi)

if valid_mols:
    mordred_df = calc.pandas(valid_mols)
    atsc1pe_list = mordred_df.get("ATSC1pe", [None] * len(valid_mols)).tolist()
    mats2c_list = mordred_df.get("MATS2c", [None] * len(valid_mols)).tolist()

for mol in valid_mols:
    slogp_vsa2_list.append(MolSurf.SlogP_VSA2(mol))

original_map = {}
for orig, new_list in original_to_new.items():
    for new_smi in new_list:
        original_map[new_smi] = orig

descriptor_df = pd.DataFrame({
    "SMILES": valid_list,
    "Original_SMILES": [original_map[s] for s in valid_list],
    "O_ATSC1pe": atsc1pe_list,
    "O_MATS2c": mats2c_list,
    "O_SlogP_VSA2": slogp_vsa2_list
})


# # 3. 比较相似性

# In[38]:


def calc_similarity(a, b):
    if a == 0 and b == 0:
        return 1.0
    elif a == 0 or b == 0:
        return 0.0
    else:
        return 1 - abs(a - b) / max(abs(a), abs(b))


# In[39]:


merged_df = descriptor_df.merge(
    df[["SMILES", "P_ATSC1pe", "P_MATS2c", "P_SlogP_VSA2"]],
    left_on="Original_SMILES",
    right_on="SMILES",
    how="left"
)

def row_similarity(row):
    sim1 = calc_similarity(row["O_ATSC1pe"], row["P_ATSC1pe"])
    sim2 = calc_similarity(row["O_MATS2c"], row["P_MATS2c"])
    sim3 = calc_similarity(row["O_SlogP_VSA2"], row["P_SlogP_VSA2"])
    return (sim1 + sim2 + sim3) / 3


# In[40]:


merged_df["AvgSimilarity"] = merged_df.apply(row_similarity, axis=1)


# In[43]:


threshold = 0.8
top_similar = merged_df[merged_df["AvgSimilarity"] >= threshold]

top_similar.to_csv("80similar_newSMILES_0411.csv", index=False)


# In[42]:


descriptor_df.to_csv("ReplaceMolecular_0411.csv", index=False)


# # 可视化

# In[33]:


import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from IPython.display import display

# === 配置 ===
file_path = '80similar_newSMILES_2.csv'
smiles_column = 'SMILES_x'
mols_per_row = 4
mols_per_page = 16
output_pdf_path = 'molecules.pdf'

# === 读取 SMILES ===
df = pd.read_csv(file_path)
smiles_list = df[smiles_column].dropna().tolist()

# === SMILES 转换为分子对象 ===
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]

# === 分页绘图 ===
pages = []
for i in range(0, len(mols), mols_per_page):
    mol_chunk = mols[i:i + mols_per_page]
    legends = smiles_list[i:i + len(mol_chunk)]

    img = Draw.MolsToGridImage(
        mol_chunk,
        molsPerRow=mols_per_row,
        subImgSize=(200, 200),
        legends=legends,
        returnPNG=False  # 返回 PIL.Image
    )
    pages.append(img.convert("RGB"))  # PDF 需要 RGB 格式

# === 保存为 PDF ===
if pages:
    pages[0].save(
        output_pdf_path,
        save_all=True,
        append_images=pages[1:]
    )
    print(f"已成功保存为 PDF：{output_pdf_path}")
else:
    print("没有有效的分子可视化，未生成 PDF。")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




