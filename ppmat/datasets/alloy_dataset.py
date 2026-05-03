# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AlloyDataset — tabular dataset for metallic glass alloy compositions.

Loads Alloy_train.csv produced by tools/prepare_alloy_data.py.
Each sample is a 66-dimensional float vector:
    columns  0-39: element composition fractions (40 elements)
    columns 40-42: Tg, Tx, Tl (thermal transition temperatures in K)
    columns 43-65: 23 GFA criteria (derived from Tg/Tx/Tl)

The "source" column is dropped on load (same as original AlloyGAN).
"""

import logging

import numpy as np
import paddle
from paddle.io import Dataset

_log = logging.getLogger("alloygan")


class AlloyDataset(Dataset):
    """Tabular dataset for AlloyGAN training.

    Args:
        path: Path to Alloy_train.csv.
        categories: Optional list of dominant-element categories to filter
            (e.g., ["Cu", "Fe", "Ti", "Zr"]). Default uses all entries.
        normalize: Whether to normalize composition fractions to [0, 1]
            by dividing by 100. Conditions are left raw (matching original
            AlloyGAN code which feeds raw Tg/Tx/Tl/GFA into the network).
    """

    # Top 40 elements in order (matches CSV columns 0-39)
    ELEMENTS = [
        "Cu", "Zr", "Al", "Ni", "Ti", "Ag", "Fe", "Mg", "B", "Si",
        "Nb", "Y", "Ca", "La", "Co", "Be", "C", "Mo", "Pd", "P",
        "Sn", "Cr", "Hf", "Zn", "Gd", "Ce", "Er", "Ga", "Au", "Nd",
        "Dy", "W", "Pr", "Ta", "Sc", "Li", "Sm", "S", "Pt", "Mn",
    ]

    def __init__(self, path, categories=None, normalize=False):
        super().__init__()
        import pandas as pd

        df = pd.read_csv(path)

        # Drop the "source" column if present (same as original code)
        if "source" in df.columns:
            df = df.drop(columns=["source"])

        # Optional category filtering by dominant element
        if categories is not None:
            elem_cols = df.columns[:40]
            dominant = df[elem_cols].idxmax(axis=1)
            mask = dominant.isin(categories)
            df = df[mask].reset_index(drop=True)
            _log.info(
                f"Filtered to categories {categories}: "
                f"{len(df)} entries"
            )

        data = df.values.astype(np.float32)

        # Normalize compositions (cols 0-39) to [0,1] by dividing by 100.
        # Conditions (cols 40+) are MinMax-scaled to [0,1].
        # G's Sigmoid output range [0,1] matches normalized comp fractions.
        # Original GAN version uses sklearn MinMaxScaler on full data;
        # CGAN version's CSV was likely pre-normalized.
        if normalize:
            data[:, :40] = data[:, :40] / 100.0
            # MinMax normalize conditions for training stability
            cond = data[:, 40:]
            cond_min = cond.min(axis=0, keepdims=True)
            cond_max = cond.max(axis=0, keepdims=True)
            denom = cond_max - cond_min
            denom[denom == 0] = 1.0  # constant columns stay 0
            data[:, 40:] = (cond - cond_min) / denom

        self.data = data

        _log.info(
            f"Loaded AlloyDataset: {len(self.data)} samples, "
            f"{self.data.shape[1]} features from {path}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx]}
