# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import numpy as np


def read_xyz(file):
    all_coords = []
    try:
        while True:
            n_atom = int(file.readline())
            file.readline()
            coords = []
            for _ in range(n_atom):
                coords.append([float(x) for x in file.readline().split()[1:4]])
            all_coords.append(coords)
    except (StopIteration, ValueError):
        all_coords = np.array(all_coords, dtype=float)
        print(all_coords.shape)
        return all_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--out", type=str, default="./data")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    for mol in ["ethane", "malonaldehyde"]:
        den = np.loadtxt(root / f"{mol}_300K/densities.txt")
        train_dir = root / f"{mol}/{mol}_train/"
        os.makedirs(train_dir, exist_ok=True)
        np.save(train_dir / "dft_densities.npy", den)
        with open(root / f"{mol}_300K/structures.xyz") as f:
            np.save(train_dir / "structures.npy", read_xyz(f))
        den = np.loadtxt(root / f"{mol}_300K-test/densities.txt")
        test_dir = root / f"{mol}/{mol}_test/"
        os.makedirs(test_dir, exist_ok=True)
        np.save(test_dir / "dft_densities.npy", den)
        with open(root / f"{mol}_300K-test/structures.xyz") as f:
            np.save(test_dir / "structures.npy", read_xyz(f))
    print("Done")
