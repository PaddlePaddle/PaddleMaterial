# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Structure Dataset for OMatG.

This module is migrated from OMatG (Open Materials Generation).
Supports reading crystalline structures from LMDB, CSV, and Parquet files.
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import lmdb
import numpy as np
import paddle
import pandas as pd

from ppmat.datasets.omatg.structure import Structure


class StructureDataset(paddle.io.Dataset):
    """
    Dataset for reading crystalline structures from several file formats.

    This dataset optionally allows for lazy reading of the structures from LMDB files.

    :param file_path:
        Path to the file containing the structures.
        Supported formats are .lmdb, .csv, and .parquet.
    :type file_path: str
    :param property_keys:
        An optional sequence of property keys that should be read from the file.
        Defaults to None.
    :type property_keys: Optional[Sequence[str]]
    :param lazy_storage:
        Whether to read the structures lazily from a LMDB file when they are requested.
        Defaults to True.
    :type lazy_storage: bool
    :param convert_to_fractional:
        Whether to convert the atomic positions to fractional coordinates.
        Defaults to True.
    :type convert_to_fractional: bool
    :param niggli_reduce:
        Whether to apply a Niggli reduction to the returned structures.
        Defaults to False.
    :type niggli_reduce: bool
    """

    def __init__(
        self,
        file_path: str,
        property_keys: Optional[Sequence[str]] = None,
        lazy_storage: bool = True,
        convert_to_fractional: bool = True,
        niggli_reduce: bool = False,
    ) -> None:
        """Constructor for the OMATGStructureDataset class."""
        self.file_path = file_path
        self.property_keys = property_keys if property_keys is not None else []
        self.lazy_storage = lazy_storage
        self.convert_to_fractional = convert_to_fractional
        self.niggli_reduce = niggli_reduce
        self._env = None  # Will be initialized lazily

        # Determine file format from extension
        self.file_format = Path(file_path).suffix.lower()

        # Initialize dataset based on format
        if self.file_format == ".lmdb":
            self._init_from_lmdb()
        elif self.file_format == ".csv":
            self._init_from_csv()
        elif self.file_format == ".parquet":
            self._init_from_parquet()
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    @property
    def env(self):
        """Lazy initialization of LMDB environment for multiprocessing safety."""
        if self._env is None:
            self._env = lmdb.open(
                self.file_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def _init_from_lmdb(self) -> None:
        """
        Initialize dataset from LMDB file.
        """
        # Open LMDB temporarily to read keys
        temp_env = lmdb.open(
            self.file_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with temp_env.begin() as txn:
            # Convert keys to list of strings for proper pickling in multiprocessing
            all_keys = [
                k.decode("ascii") if isinstance(k, bytes) else k
                for k in txn.cursor().iternext(values=False)
            ]
            # Filter out special keys like '__len__'
            self.keys = [k for k in all_keys if not k.startswith("__")]

        # Close temporary environment
        temp_env.close()

        if not self.lazy_storage:
            # Load all structures into memory
            # Need to initialize env for this
            with self.env.begin() as txn:
                self.structures = []
                for key in self.keys:
                    # Convert key back to bytes for LMDB lookup
                    key_bytes = key.encode("ascii") if isinstance(key, str) else key
                    data = pickle.loads(txn.get(key_bytes))
                    structure = self._create_structure(data)
                    self.structures.append(structure)

    def _init_from_csv(self) -> None:
        """
        Initialize dataset from CSV file.
        The CSV file is expected to contain a "cif" column with CIF strings.
        """
        from pymatgen.core import Structure as PymatgenStructure

        df = pd.read_csv(self.file_path)
        if "cif" not in df.columns:
            raise KeyError(
                f"CSV file does not contain 'cif' column. "
                f"Available columns: {list(df.columns)}"
            )

        self.structures = []
        for _, row in df.iterrows():
            pmg = PymatgenStructure.from_str(row["cif"], fmt="cif")

            cell = paddle.to_tensor(pmg.lattice.matrix, dtype="float32")
            atomic_numbers = paddle.to_tensor(pmg.atomic_numbers, dtype="int64")
            pos = paddle.to_tensor(pmg.cart_coords, dtype="float32")

            property_dict = {}
            for key in self.property_keys:
                if key in df.columns:
                    val = row[key]
                    if isinstance(val, (int, float, list)):
                        property_dict[key] = paddle.to_tensor(val, dtype="float32")
                    elif isinstance(val, np.ndarray):
                        property_dict[key] = paddle.to_tensor(val, dtype="float32")

            structure = Structure(
                cell=cell,
                atomic_numbers=atomic_numbers,
                pos=pos,
                property_dict=property_dict if property_dict else None,
                pos_is_fractional=False,
            )

            if self.convert_to_fractional:
                structure.convert_to_fractional()
            if self.niggli_reduce:
                structure.niggli_reduce()

            self.structures.append(structure)

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _init_from_parquet(self) -> None:
        """
        Initialize dataset from Parquet file.
        Expected columns: "positions" (Nx3), "cell" (3x3), "atomic_numbers" (N,).
        """
        df = pd.read_parquet(self.file_path)
        required_cols = ["positions", "cell", "atomic_numbers"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Parquet file does not contain '{col}' column. "
                    f"Available columns: {list(df.columns)}"
                )

        self.structures = []
        for _, row in df.iterrows():
            atomic_numbers = paddle.to_tensor(
                np.asarray(row["atomic_numbers"], dtype=np.int64), dtype="int64"
            )
            pos = paddle.to_tensor(np.stack(row["positions"]), dtype="float32")
            cell = paddle.to_tensor(np.stack(row["cell"]), dtype="float32")

            property_dict = {}
            for key in self.property_keys:
                if key in df.columns:
                    val = row[key]
                    if isinstance(val, (int, float, list)):
                        property_dict[key] = paddle.to_tensor(val, dtype="float32")
                    elif isinstance(val, np.ndarray):
                        property_dict[key] = paddle.to_tensor(val, dtype="float32")

            structure = Structure(
                cell=cell,
                atomic_numbers=atomic_numbers,
                pos=pos,
                property_dict=property_dict if property_dict else None,
                pos_is_fractional=False,
            )

            if self.convert_to_fractional:
                structure.convert_to_fractional()
            if self.niggli_reduce:
                structure.niggli_reduce()

            self.structures.append(structure)

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _create_structure(self, data: dict[str, Any]) -> Structure:
        """
        Create a Structure object from data dictionary.
        :param data:
            Dictionary containing structure data.
        :return:
            Structure object.
        """
        # Extract required fields
        cell = paddle.to_tensor(data["cell"], dtype="float32")
        atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
        pos = paddle.to_tensor(data["pos"], dtype="float32")

        # Extract properties
        property_dict = {}
        for key in self.property_keys:
            if key in data:
                property_dict[key] = paddle.to_tensor(data[key], dtype="float32")

        # Create structure
        structure = Structure(
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            property_dict=property_dict if property_dict else None,
            pos_is_fractional=False,  # LMDB stores Cartesian coordinates
        )

        # Apply transformations
        if self.convert_to_fractional:
            structure.convert_to_fractional()

        if self.niggli_reduce:
            structure.niggli_reduce()

        return structure

    def __len__(self) -> int:
        """Get the number of structures in the dataset."""
        if self.lazy_storage:
            return len(self.keys)
        else:
            return len(self.structures)

    def __getitem__(self, idx: int) -> Structure:
        """
        Get a structure from the dataset.

        :param idx:
            Index of the structure to retrieve.
        :return:
            Structure object.
        """
        if self.lazy_storage:
            # Lazy loading from LMDB
            with self.env.begin() as txn:
                key = self.keys[idx]
                # Convert string key to bytes for LMDB lookup
                key_bytes = key.encode("ascii") if isinstance(key, str) else key

                value = txn.get(key_bytes)
                if value is None:
                    raise KeyError(f"Key {key} not found in LMDB")

                data = pickle.loads(value)

            return self._create_structure(data)
        else:
            # Loading from memory
            return self.structures[idx]

    def __del__(self) -> None:
        """Clean up LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
