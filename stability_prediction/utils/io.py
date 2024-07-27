from __future__ import annotations

import pickle

import numpy as np


def load_from_pickle(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data
