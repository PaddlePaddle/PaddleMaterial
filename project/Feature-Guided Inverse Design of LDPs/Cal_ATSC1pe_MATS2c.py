#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from tqdm import tqdm
import gc

calc = Calculator([descriptors.Autocorrelation])

def process_chunk(chunk):
    valid_mols, valid_smiles = [], []
    invalid = 0
    
    for smiles in chunk['SMILES']: 
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_mols.append(mol)
            valid_smiles.append(smiles)
        else:
            invalid += 1
    
    if valid_mols:
        results = calc.pandas(valid_mols)
        results.insert(0, 'SMILES', valid_smiles)
        target_cols = ['SMILES']
        for col in ['ATSC1pe', 'MATS2c']:
            if col in results.columns:
                target_cols.append(col)
        results = results[target_cols]
        return results, invalid
    return None, invalid

chunk_size = 200000
total_valid = total_invalid = 0

for chunk_number, chunk in enumerate(pd.read_csv("11.csv", chunksize=chunk_size)):
    df_result, invalid = process_chunk(chunk)
    total_invalid += invalid
    
    if df_result is not None:
        total_valid += len(df_result)
        df_result.to_csv(
            "2.csv",
            mode='a',
            header=(chunk_number == 0),
            index=False
        )
    
    del chunk, df_result
    gc.collect()

print(f"处理完成：有效 {total_valid}，无效 {total_invalid}")


# # 合并文件

# In[4]:


# 读取文件1的前两列（假设列名为col1、col2）
df1 = pd.read_csv('2.csv', usecols=[0, 1, 2])  # 按索引选取
# 或根据列名选取：usecols=['col1', 'col2']

# 读取文件2的第二、第三列（假设索引为1、2）
df2 = pd.read_csv('1.csv', usecols=[1])  # 按索引选取
# 或根据列名选取：usecols=['colB', 'colC']


# In[5]:


combined_df = pd.concat([df1, df2], axis=1)

df1_reset = df1.reset_index(drop=True)
df2_reset = df2.reset_index(drop=True)
combined_df = pd.concat([df1_reset, df2_reset], axis=1)


# In[7]:


combined_df.to_csv('111.csv', index=False)


# In[ ]:




