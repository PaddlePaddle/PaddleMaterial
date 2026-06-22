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

from __future__ import absolute_import
from __future__ import annotations

from typing import Callable
from typing import Optional

import pandas as pd
import paddle
from paddle.io import Dataset

from ppmat.utils import logger


class TransPolymerCsvDataset(Dataset):
    """TransPolymer CSV Dataset Handler

    This class loads polymer SMILES strings and regression targets from CSV files.
    Tokenization and model input construction are handled by the TransPolymer model.

    **Data Format**
    The dataset is stored in CSV format. The first column is the polymer sequence
    and the target property can be selected by column name or by position.

    **Example Row:**
    ```csv
    smiles,Conductivity [S/cm]
    [*]CC[*],-3.12
    ```

    Args:
        path (str, optional): Path to the CSV file. Defaults to None.
        property_names (Optional[list[str]], optional): Target property names. Only
            one target is supported currently. Defaults to None.
        smiles_key (str, optional): Key used for returning polymer sequences.
            Defaults to "smiles".
        transforms (Optional[Callable], optional): Preprocess transforms for each
            sample. Defaults to None.
        file_path (str, optional): Deprecated alias of `path`. Defaults to None.
        label_name (str, optional): Deprecated alias used when `property_names` is
            not set. Defaults to None.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        property_names: Optional[list[str]] = None,
        smiles_key: str = "smiles",
        transforms: Optional[Callable] = None,
        file_path: Optional[str] = None,
        label_name: Optional[str] = None,
        **kwargs,  # for compatibility
    ):
        super().__init__()
        if path is None:
            path = file_path
        if path is None:
            raise ValueError("`path` must be specified for TransPolymerCsvDataset.")

        if isinstance(property_names, str):
            property_names = [property_names]
        if property_names is not None and len(property_names) != 1:
            raise NotImplementedError(
                "TransPolymerCsvDataset currently supports single-target regression."
            )

        self.path = path
        self.data = pd.read_csv(path)
        self.smiles_key = smiles_key
        self.property_names = property_names or [label_name or self.data.columns[1]]
        self.transforms = transforms
        logger.info(f"Load {len(self.data)} samples from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row.iloc[0])
        property_name = self.property_names[0]
        label = (
            float(row[property_name]) if property_name in row else float(row.iloc[1])
        )
        sample = {
            self.smiles_key: seq,
            property_name: paddle.to_tensor([label], dtype="float32"),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
