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

"""SMILES-based dataset for polymer property prediction with TrinityLLM."""

import csv
import os

import paddle
from paddle.io import Dataset


class SMILESDataset(Dataset):
    """Dataset for SMILES-based property prediction.

    Reads CSV files with SMILES strings and numerical property columns.
    Tokenizes SMILES using a regex-based tokenizer compatible with TrinityLLM.

    Args:
        data_path: Path to CSV file.
        smiles_col: Column name for SMILES strings (default: "smiles").
        label_cols: List of property column names.
        max_length: Maximum token sequence length (default: 512).
        tokenizer: Optional pre-built tokenizer. If None, uses default SMILES tokenizer.
    """

    def __init__(self, data_path, smiles_col="smiles", label_cols=None,
                 max_length=512, tokenizer=None):
        super().__init__()
        self.max_length = max_length
        self.smiles_col = smiles_col
        self.label_cols = label_cols or []

        # Import tokenizer from trinityllm module
        if tokenizer is None:
            from ppmat.models.trinityllm.trinityllm import SMILESTokenizer
            self.tokenizer = SMILESTokenizer()
        else:
            self.tokenizer = tokenizer

        # Read CSV
        self.data = []
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        smiles = row[self.smiles_col]

        # Tokenize
        tokens = self.tokenizer.encode(smiles)

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_id = self.tokenizer.pad_id
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))

        result = {
            "input_ids": paddle.to_tensor(tokens, dtype="int64"),
        }

        # Add labels
        for col in self.label_cols:
            if col in row:
                result[col] = paddle.to_tensor([float(row[col])], dtype="float32")

        return result
