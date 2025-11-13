#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolSurf
from tqdm import tqdm

df = pd.read_csv('11.csv')

smiles_column = 'SMILES'

def calculate_slogp_vsa2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return MolSurf.SlogP_VSA2(mol)
    else:
        return None

tqdm.pandas(desc="Processing SMILES")
df['SlogP_VSA2'] = df[smiles_column].progress_apply(calculate_slogp_vsa2)

df.to_csv('1.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




