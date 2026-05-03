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
CIF Token Dataset for CrystalLLM.

Memory-mapped dataset of pre-tokenized CIF files stored as uint16 numpy arrays.
Supports CIF-aware sampling via optional starts.pkl for selecting CIF boundaries.
"""

import os
import pickle

import numpy as np
from paddle.io import Dataset


class CIFTokenDataset(Dataset):
    """Memory-mapped dataset of pre-tokenized CIF sequences.

    Data format:
        - ``train.bin`` / ``val.bin``: numpy memmap files of dtype uint16
        - ``train_starts.pkl`` / ``val_starts.pkl`` (optional): pickle files
          containing lists of starting indices for each CIF entry, enabling
          CIF-boundary-aware sampling.

    Each sample returns a dict with:
        - ``input_ids``: (block_size,) int64 array of token indices
        - ``target_ids``: (block_size,) int64 array shifted by 1

    Args:
        data_path: Path to the ``.bin`` memory-mapped file.
        block_size: Sequence length for each sample.
        starts_path: Optional path to a pickle file with CIF start indices.
            If provided, samples are drawn from CIF boundaries instead of
            random positions. This improves training quality.
    """

    def __init__(
        self,
        data_path: str,
        block_size: int = 1024,
        starts_path: str = None,
    ):
        super().__init__()
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

        self.starts = None
        if starts_path is not None and os.path.exists(starts_path):
            with open(starts_path, "rb") as f:
                self.starts = pickle.load(f)

        # Compute effective length
        if self.starts is not None:
            self._len = len(self.starts)
        else:
            self._len = max(1, len(self.data) - block_size)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.starts is not None:
            # CIF-aware sampling: pick from boundary starts
            i = self.starts[idx % len(self.starts)]
        else:
            # Random window sampling
            i = idx % (len(self.data) - self.block_size)

        chunk = self.data[i : i + self.block_size + 1].astype(np.int64)
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return {"input_ids": input_ids, "target_ids": target_ids}
