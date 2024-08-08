import pickle

import numpy as np
import paddle

train_file = "./data/2d_structure/2d_structure_csv_ehull_200/train_ori.pkl"

with open(train_file, "rb") as f:
    cached_data = pickle.load(f)

num_atoms = []
for data_dict in cached_data:
    num_atoms.append(data_dict["graph_arrays"][-1])

train_dist = [0 for i in range(max(num_atoms) + 1)]
for i in num_atoms:
    train_dist[i] += 1
train_dist = [i / len(num_atoms) for i in train_dist]
print(train_dist)
