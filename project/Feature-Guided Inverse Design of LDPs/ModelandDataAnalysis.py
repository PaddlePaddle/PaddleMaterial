#!/usr/bin/env python
# coding: utf-8

# # 特征生成率分析

# ## 生成集（Double Fit）

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ReplaceMolecular.csv")

columns = ["O_ATSC1pe", "O_MATS2c", "O_SlogP_VSA2"]
titles = {
    "O_ATSC1pe": "ATSC1pe",
    "O_MATS2c": "MATS2c",
    "O_SlogP_VSA2": "SlogP_VSA2"
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for ax, col in zip(axes, columns):
    sns.violinplot(y=df[col], ax=ax, color='teal', inner="box")
    ax.set_title(titles[col], fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# ## 训练集

# In[4]:


df = pd.read_csv("Modeldata0406.csv")

columns = ["ATSC1pe", "MATS2c", "SlogP_VSA2"]
titles = {
    "ATSC1pe": "ATSC1pe",
    "MATS2c": "MATS2c",
    "SlogP_VSA2": "SlogP_VSA2"
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for ax, col in zip(axes, columns):
    sns.violinplot(y=df[col], ax=ax, color='salmon', inner="box")
    ax.set_title(titles[col], fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()


# # Double Fit效果分析

# In[15]:


import matplotlib.pyplot as plt

features = ['ATSC1pe', 'MATS2c', 'SlogP_VSA2']
total_mse = [0.1678359, 0.0645482, 92.414153]
optimized_mse = [0.0807061, 0.0594437, 4.3848465]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
colors = ['teal', 'violet']

for i, ax in enumerate(axes):
    mse_values = [total_mse[i], optimized_mse[i]]
    bars = ax.bar(['Total MSE', 'Optimized MSE'], mse_values, color=colors)


    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(mse_values), f'{yval:.4f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_title(features[i])
    ax.set_ylabel('MSE')
    ax.set_ylim(0, max(total_mse[i] * 1.2, optimized_mse[i] * 1.2))  

plt.suptitle('MSE Comparison Before and After Optimization', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig('combined_mse_comparison.png')  
plt.show()


# # t-SNE

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("Modeldata0406.csv")
gen_df = pd.read_excel("NewMolecules0408.xlsx")
opt_df = pd.read_csv("80similar_newSMILES_0411.csv")

train_df["Set"] = "Training"
gen_df["Set"] = "Generated"
opt_df["Set"] = "Optimized"

all_data = pd.concat([
    train_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]],
    gen_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]],
    opt_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]]
], ignore_index=True)

sampled_train = all_data[all_data["Set"] == "Training"].sample(n=10000, random_state=42)
others = all_data[all_data["Set"] != "Training"]
combined_df = pd.concat([sampled_train, others], ignore_index=True)

print(combined_df["Set"].value_counts())

combined_df = combined_df.dropna(subset=["ATSC1pe", "MATS2c", "SlogP_VSA2"])

features = ["ATSC1pe", "MATS2c", "SlogP_VSA2"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_df[features])

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

combined_df["x"] = X_tsne[:, 0]
combined_df["y"] = X_tsne[:, 1]

plt.figure(figsize=(10, 7))
colors = {"Training": "tomato", "Generated": "teal", "Optimized": "violet"}
sizes = {"Training": 10, "Generated": 40, "Optimized": 40}

for label in combined_df["Set"].unique():
    subset = combined_df[combined_df["Set"] == label]
    plt.scatter(subset["x"], subset["y"], label=label, 
                c=colors[label], alpha=0.6, s=sizes[label])

plt.title("t-SNE visualization of molecular descriptors")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # UMAP

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("Modeldata0406.csv")
gen_df = pd.read_excel("NewMolecules0408.xlsx")
opt_df = pd.read_csv("80similar_newSMILES_0411.csv")

train_df["Set"] = "Training"
gen_df["Set"] = "Generated"
opt_df["Set"] = "Optimized"

all_data = pd.concat([
    train_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]],
    gen_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]],
    opt_df[["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "Set"]]
], ignore_index=True)

sampled_train = all_data[all_data["Set"] == "Training"].sample(n=10000, random_state=42)
others = all_data[all_data["Set"] != "Training"]
combined_df = pd.concat([sampled_train, others], ignore_index=True)

combined_df = combined_df.dropna(subset=["ATSC1pe", "MATS2c", "SlogP_VSA2"])

features = ["ATSC1pe", "MATS2c", "SlogP_VSA2"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_df[features])

# UMAP 降维
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

combined_df["x"] = X_umap[:, 0]
combined_df["y"] = X_umap[:, 1]

plt.figure(figsize=(10, 7))
colors = {"Training": "tomato", "Generated": "teal", "Optimized": "violet"}
sizes = {"Training": 10, "Generated": 40, "Optimized": 40}

for label in combined_df["Set"].unique():
    subset = combined_df[combined_df["Set"] == label]
    plt.scatter(subset["x"], subset["y"], label=label,
                c=colors[label], alpha=0.6, s=sizes[label])

plt.title("UMAP visualization of molecular descriptors")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # PCA

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("Modeldata0406.csv")
gen_df = pd.read_excel("NewMolecules0408.xlsx")
opt_df = pd.read_csv("80similar_newSMILES_0411.csv")

train_df["label"] = "Training"
gen_df["label"] = "Generated"
opt_df["label"] = "Optimized"

columns = ["SMILES", "ATSC1pe", "MATS2c", "SlogP_VSA2", "label"]
combined_df = pd.concat([
    train_df[columns],
    gen_df[columns],
    opt_df[columns]
], ignore_index=True)

combined_df = combined_df.dropna(subset=["ATSC1pe", "MATS2c", "SlogP_VSA2"])

X = combined_df[["ATSC1pe", "MATS2c", "SlogP_VSA2"]]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

combined_df["PC1"] = X_pca[:, 0]
combined_df["PC2"] = X_pca[:, 1]

color_map = {"Training": "darksalmon", "Generated": "teal", "Optimized": "blueviolet"}

plt.figure(figsize=(8, 6))
for label in combined_df["label"].unique():
    subset = combined_df[combined_df["label"] == label]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        label=label,
        alpha=0.7, s=40,
        color=color_map[label]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Molecule Descriptor Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # Tanimoto

# In[13]:


import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

tqdm.pandas()  

train_df = pd.read_csv("Modeldata0406.csv")
gen_df = pd.read_excel("NewMolecules0408.xlsx")
opt_df = pd.read_csv("80similar_newSMILES_0411.csv")

# 随机抽样训练集 5 万个
train_smiles = train_df["SMILES"].dropna().unique().tolist()
train_sample = random.sample(train_smiles, 100000)

gen_smiles = gen_df["SMILES"].dropna().unique().tolist()
opt_smiles = opt_df["SMILES"].dropna().unique().tolist()

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def get_fp(mol, fp_type="rdkit"):
    if mol is None:
        return None
    if fp_type == "rdkit":
        return Chem.RDKFingerprint(mol)
    elif fp_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        raise ValueError("Unknown fingerprint type")

ref_mols = [mol_from_smiles(smi) for smi in train_sample]
ref_rdkit_fps = [get_fp(mol, "rdkit") for mol in ref_mols if mol is not None]
ref_morgan_fps = [get_fp(mol, "morgan") for mol in ref_mols if mol is not None]

def calc_max_tc(smi_list, ref_fps, fp_type):
    results = []
    for smi in tqdm(smi_list):
        mol = mol_from_smiles(smi)
        if mol is None:
            continue
        fp = get_fp(mol, fp_type)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        results.append(max(sims))
    return results

gen_rdkit = calc_max_tc(gen_smiles, ref_rdkit_fps, "rdkit")
opt_rdkit = calc_max_tc(opt_smiles, ref_rdkit_fps, "rdkit")

gen_morgan = calc_max_tc(gen_smiles, ref_morgan_fps, "morgan")
opt_morgan = calc_max_tc(opt_smiles, ref_morgan_fps, "morgan")

def plot_histogram(data_dict, threshold, fp_type):
    import numpy as np
    plt.figure(figsize=(8, 6))

    bins = np.linspace(0, 1.0, 31) 
    colors = {"Optimized": "violet", "Generated": "teal"}

    for label, data in data_dict.items():
        plt.hist(data, bins=bins, alpha=0.7, label=label, color=colors.get(label, "gray"), edgecolor="white")

    plt.axvline(x=threshold, linestyle="--", color="black", linewidth=1.5, label=f"Threshold: {threshold}")
    
    plt.xlabel(f"Maximum Tanimoto Coefficient ({fp_type})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(np.linspace(0, 1.0, 11))
    plt.title(f"Max Tanimoto Distribution ({fp_type.capitalize()} Fingerprint)", fontsize=14, pad=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_histogram(
    {"Optimized": opt_rdkit, "Generated": gen_rdkit},
    threshold=0.7, fp_type="rdkit"
)

plot_histogram(
    {"Optimized": opt_morgan, "Generated": gen_morgan},
    threshold=0.5, fp_type="morgan"
)


# In[ ]:




