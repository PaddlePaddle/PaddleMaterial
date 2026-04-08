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

from __future__ import annotations

import csv
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from paddle.io import Dataset

class PolymerChempropDataset(Dataset):
    """Dataset for polymer-chemprop model.

    Loads CSV data with SMILES and target columns, converts to MolGraph.
    """

    def __init__(
        self,
        path: str,
        smiles_columns: List[str] = None,
        target_columns: List[str] = None,
        featurization_config: Dict = None,
        max_data_size: int = None,
    ):
        super().__init__()
        self.path = path

        # Build featurization config (lazy import to avoid hard dependency at module load)
        from ppmat.models.polymer_chemprop.featurization import FeaturizationConfig

        if featurization_config is not None:
            self.feat_config = FeaturizationConfig(**featurization_config)
        else:
            self.feat_config = FeaturizationConfig()

        # Read CSV
        with open(path) as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

            # Default: first column is SMILES
            if smiles_columns is None:
                smiles_columns = [header[0]]
            self.smiles_columns = smiles_columns

            # Default: all non-SMILES columns are targets
            if target_columns is None:
                target_columns = [c for c in header if c not in smiles_columns]
            self.target_columns = target_columns

            rows = list(reader)

        if max_data_size is not None:
            rows = rows[:max_data_size]

        self.smiles_list = []
        self.targets_list = []
        self.target_masks = []

        for row in rows:
            smiles = [row[col] for col in self.smiles_columns]
            targets = []
            mask = []
            for col in self.target_columns:
                val = row.get(col, '')
                if val == '' or val is None:
                    targets.append(0.0)
                    mask.append(0.0)
                else:
                    targets.append(float(val))
                    mask.append(1.0)
            self.smiles_list.append(smiles)
            self.targets_list.append(np.array(targets, dtype=np.float32))
            self.target_masks.append(np.array(mask, dtype=np.float32))

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        from ppmat.models.polymer_chemprop.featurization import MolGraph

        # Build MolGraph for each molecule
        mol_graphs = []
        for smi in smiles:
            mol_graph = MolGraph(smi, config=self.feat_config)
            mol_graphs.append(mol_graph)

        return {
            "mol_graphs": mol_graphs,
            "targets": self.targets_list[idx],
            "target_mask": self.target_masks[idx],
            "features": None,
            "smiles": smiles,
        }
