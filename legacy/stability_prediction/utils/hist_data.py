from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm


def load_dataset_from_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to energy file")
    parser.add_argument("--save_path", type=str, help="Path to save file")
    args = parser.parse_args()

    data = load_dataset_from_pickle(args.path)
    new_data = []
    for d in data:
        if abs(d) > 50:
            print(d)
        else:
            new_data.append(d)
    data = new_data

    data = np.asarray(data)
    n, bins, patch = plt.hist(data, bins=100)
    plt.savefig(args.save_path)
