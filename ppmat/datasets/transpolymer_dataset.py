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

from ppmat.models.transpolymer.tokenizer import PolymerSmilesTokenizer
from ppmat.utils import logger


class TransPolymerCsvDataset(Dataset):
    """TransPolymer CSV Dataset Handler

    This class loads polymer SMILES strings and regression targets from CSV files
    and tokenizes the strings with the TransPolymer tokenizer.

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
        tokenizer_name_or_path (str, optional): Tokenizer checkpoint or local path.
            Defaults to "roberta-base".
        vocab_sup_file (Optional[str], optional): Supplementary vocabulary CSV path.
            Defaults to None.
        blocksize (int, optional): Maximum token sequence length. Defaults to 411.
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
        tokenizer=None,
        tokenizer_name_or_path: str = "roberta-base",
        vocab_sup_file: Optional[str] = None,
        blocksize: int = 411,
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
        self.tokenizer = tokenizer or PolymerSmilesTokenizer.from_pretrained(
            tokenizer_name_or_path, max_len=blocksize
        )
        if vocab_sup_file is not None:
            vocab_sup = pd.read_csv(vocab_sup_file, header=None).values.flatten()
            self.tokenizer.add_tokens(vocab_sup.tolist())
        self.blocksize = blocksize
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
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        sample = {
            "input_ids": paddle.to_tensor(encoding["input_ids"], dtype="int64"),
            "attention_mask": paddle.to_tensor(
                encoding["attention_mask"], dtype="int64"
            ),
            property_name: paddle.to_tensor([label], dtype="float32"),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
