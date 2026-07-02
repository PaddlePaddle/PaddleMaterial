# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the specific language governing permissions and
# limitations under the License.

"""
Structure Dataset for OMatG (Open Materials Generation).

Combines Structure, OMATGData, and StructureDataset into a single module.
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import lmdb
import numpy as np
import paddle
import pandas as pd
from ase import Atoms
from ase.symbols import Symbols

from ppmat.datasets.geometric_data_type import Data


class Structure:
    """Storage for crystalline structure with cell, atomic numbers, coordinates, properties, and metadata.
    Supports Cartesian/fractional coordinate conversion and Niggli reduction.
    """

    def __init__(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        property_dict: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        pos_is_fractional: bool = False,
    ) -> None:
        if cell.shape != (3, 3):
            raise ValueError(f"cell must be 3x3, got {cell.shape}")
        if atomic_numbers.dim() != 1:
            raise ValueError(f"atomic_numbers must be 1D, got {atomic_numbers.dim()}D")
        if pos.shape[0] != len(atomic_numbers) or pos.shape[1] != 3:
            raise ValueError(f"pos must be (N, 3), got {pos.shape}")

        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._pos = pos
        self._property_dict = property_dict if property_dict is not None else {}
        self._metadata = metadata if metadata is not None else {}
        self._fractional = pos_is_fractional

    @property
    def cell(self) -> paddle.Tensor:
        return self._cell

    @property
    def atomic_numbers(self) -> paddle.Tensor:
        return self._atomic_numbers

    @property
    def symbols(self) -> list[str]:
        return list(Symbols(self._atomic_numbers.numpy()))

    @property
    def pos(self) -> paddle.Tensor:
        return self._pos

    @property
    def pos_is_fractional(self) -> bool:
        return self._fractional

    @property
    def property_dict(self) -> dict[str, Any]:
        return self._property_dict

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def to(self, floating_point_precision: str) -> None:
        valid_precisions = ["float32", "float64", "float16", "bfloat16"]
        if floating_point_precision not in valid_precisions:
            raise ValueError(
                f"Unsupported floating point precision: {floating_point_precision}. "
                f"Supported precisions are {valid_precisions}."
            )
        self._cell = self._cell.cast(floating_point_precision)
        self._pos = self._pos.cast(floating_point_precision)
        for key, value in self._property_dict.items():
            if paddle.is_tensor(value) and value.dtype in [
                paddle.float32,
                paddle.float64,
                paddle.float16,
                paddle.bfloat16,
            ]:
                self._property_dict[key] = value.cast(floating_point_precision)

    def get_ase_atoms(self) -> Atoms:
        if self._fractional:
            return Atoms(
                numbers=self.atomic_numbers.tolist(),
                scaled_positions=self.pos.numpy(),
                cell=self.cell.numpy(),
                pbc=True,
                info=self.property_dict | self.metadata,
            )
        else:
            return Atoms(
                numbers=self.atomic_numbers.tolist(),
                positions=self.pos.numpy(),
                cell=self.cell.numpy(),
                pbc=True,
                info=self.property_dict | self.metadata,
            )

    def niggli_reduce(self) -> None:
        from pymatgen.core import Structure as PmgStructure, Element

        species = [Element.from_Z(int(z)).symbol for z in self._atomic_numbers.numpy()]
        pmg = PmgStructure(
            lattice=self._cell.numpy(),
            species=species,
            coords=self._pos.numpy(),
            coords_are_cartesian=not self._fractional,
        )
        reduced = pmg.get_reduced_structure(reduction_algo="niggli")
        self._cell = paddle.to_tensor(reduced.lattice.matrix, dtype=self._cell.dtype)
        self._pos = paddle.to_tensor(reduced.frac_coords, dtype=self._pos.dtype)
        self._fractional = True

    def convert_to_fractional(self) -> None:
        if not self._fractional:
            with paddle.no_grad():
                self._pos = paddle.remainder(
                    paddle.linalg.solve(self._cell, self._pos.T).T,
                    1.0,
                )
            self._fractional = True

    def convert_to_cartesian(self) -> None:
        if self._fractional:
            with paddle.no_grad():
                self._pos = paddle.matmul(self._pos, self._cell)
            self._fractional = False

    def __repr__(self) -> str:
        return (
            f"Structure(cell={self._cell.shape}, "
            f"atomic_numbers={self._atomic_numbers.shape}, "
            f"pos={self._pos.shape}, "
            f"fractional={self._fractional})"
        )


class OMATGData(Data):
    """Representation of single/batch crystal structures.
    Batch format: n_atoms(batch_size,), species(total_atoms,), cell(batch,3,3),
    pos(total_atoms,3), pos_is_fractional(batch_size,), ptr(batch+1,).
    """

    _FIELD_NAMES = ("n_atoms", "species", "cell", "pos", "pos_is_fractional", "batch", "ptr")

    def __init__(self, structure: Optional[Structure] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if structure is None:
            self.n_atoms = None
            self.species = None
            self.cell = None
            self.pos = None
            self.pos_is_fractional = None
            self.property_dict = None
            self.batch = None
            self.ptr = None
        else:
            self._from_structure(structure)

    def _from_structure(self, structure: Structure) -> None:
        n = len(structure.atomic_numbers)
        self.n_atoms = paddle.to_tensor([n], dtype="int64")
        self.species = structure.atomic_numbers
        self.cell = structure.cell.unsqueeze(0)
        self.pos = structure.pos
        self.pos_is_fractional = paddle.to_tensor(
            [structure.pos_is_fractional], dtype="bool"
        )
        self.property_dict = [structure.property_dict]
        self.batch = paddle.zeros([n], dtype="int64")
        self.ptr = paddle.to_tensor([0, n], dtype="int64")

    @classmethod
    def from_batch(
        cls, structures: list[Structure], concatenate: bool = True
    ) -> "OMATGData":
        if not concatenate:
            return [cls(s) for s in structures]

        if len(structures) == 0:
            return cls()

        if len(structures) == 1:
            return cls(structures[0])

        cells = []
        species_list = []
        pos_list = []
        pos_is_fractional_list = []
        n_atoms_list = []
        batch_indices = []
        properties_list = []

        for i, struct in enumerate(structures):
            n_atoms_list.append(len(struct.atomic_numbers))
            species_list.append(struct.atomic_numbers)
            cells.append(struct.cell)
            pos_list.append(struct.pos)
            pos_is_fractional_list.append(
                paddle.to_tensor([struct.pos_is_fractional], dtype="bool")
            )
            batch_indices.append(
                paddle.full([len(struct.atomic_numbers)], i, dtype="int64")
            )
            properties_list.append(struct.property_dict)

        data = cls()
        data.n_atoms = paddle.to_tensor(n_atoms_list, dtype="int64")
        data.species = paddle.concat(species_list)
        data.cell = paddle.stack(cells)
        data.pos = paddle.concat(pos_list)
        data.pos_is_fractional = paddle.concat(pos_is_fractional_list)
        data.batch = paddle.concat(batch_indices)
        data.ptr = paddle.concat(
            [
                paddle.to_tensor([0], dtype="int64"),
                paddle.cumsum(data.n_atoms, axis=0).cast("int64"),
            ]
        )
        data.property_dict = properties_list

        return data

    @classmethod
    def from_collate_dict(cls, data: dict) -> "OMATGData":
        num_atoms = data["num_atoms"]
        batch = data["node2graph"]
        ptr = paddle.concat([
            paddle.to_tensor([0], dtype="int64"),
            paddle.cumsum(num_atoms, axis=0).cast("int64"),
        ])
        d = cls()
        d.n_atoms = num_atoms
        d.species = data["atom_types"]
        d.cell = data["lattices"]
        d.pos = data["frac_coords"]
        d.pos_is_fractional = paddle.ones_like(num_atoms, dtype="bool")
        d.batch = batch
        d.ptr = ptr
        d.property_dict = {}
        return d

    @property
    def num_graphs(self) -> int:
        if self.n_atoms is None:
            return 0
        return len(self.n_atoms)

    @property
    def num_atoms(self) -> int:
        if self.species is None:
            return 0
        return len(self.species)

    def get_graph(self, idx: int) -> Structure:
        if idx < 0 or idx >= self.num_graphs:
            raise IndexError(f"Index {idx} out of range for batch of {self.num_graphs}")

        start = int(self.ptr[idx])
        end = int(self.ptr[idx + 1])

        species = self.species[start:end]
        pos = self.pos[start:end]
        cell = self.cell[idx]
        pos_is_fractional = bool(self.pos_is_fractional[idx])

        if self.property_dict and idx < len(self.property_dict):
            prop = self.property_dict[idx]
        else:
            prop = {}

        return Structure(
            cell=cell,
            atomic_numbers=species,
            pos=pos,
            property_dict=prop,
            metadata={},
            pos_is_fractional=pos_is_fractional,
        )

    def slice(self, idx: int) -> slice:
        start = int(self.ptr[idx])
        end = int(self.ptr[idx + 1])
        return slice(start, end)

    @classmethod
    def _from_dict(cls, data_dict: dict[str, Any]) -> "OMATGData":
        data = cls()
        for name in cls._FIELD_NAMES:
            setattr(data, name, data_dict.get(name))
        data.property_dict = data_dict.get("property_dict")
        return data

    def set_field(self, field_name: str, value: paddle.Tensor) -> None:
        if field_name not in self._FIELD_NAMES:
            raise ValueError(f"Unknown field: {field_name}")
        setattr(self, field_name, value)

    def get_field(self, field_name: str) -> paddle.Tensor:
        if field_name not in self._FIELD_NAMES:
            raise ValueError(f"Unknown field: {field_name}")
        return getattr(self, field_name)

    def __repr__(self) -> str:
        return (
            f"OMATGData(num_graphs={self.num_graphs}, "
            f"num_atoms={self.num_atoms}, "
            f"cell={self.cell.shape if self.cell is not None else None})"
        )


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
        self.file_path = file_path
        self.property_keys = property_keys if property_keys is not None else []
        self.lazy_storage = lazy_storage
        self.convert_to_fractional = convert_to_fractional
        self.niggli_reduce = niggli_reduce
        self._env = None

        file_format = Path(file_path).suffix.lower()

        if file_format == ".lmdb":
            self._init_from_lmdb()
        elif file_format == ".csv":
            self._init_from_csv()
        elif file_format == ".parquet":
            self._init_from_parquet()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    @property
    def env(self):
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
        temp_env = lmdb.open(
            self.file_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        try:
            with temp_env.begin() as txn:
                all_keys = [
                    k.decode("ascii") if isinstance(k, bytes) else k
                    for k in txn.cursor().iternext(values=False)
                ]
                self.keys = [k for k in all_keys if not k.startswith("__")]

            if not self.lazy_storage:
                self.structures = []
                with temp_env.begin() as txn:
                    for key in self.keys:
                        key_bytes = (
                            key.encode("ascii") if isinstance(key, str) else key
                        )
                        data = pickle.loads(txn.get(key_bytes))
                        structure = self._create_structure(data)
                        self.structures.append(structure)
        finally:
            temp_env.close()

    def _extract_property_dict(self, data) -> dict[str, Any]:
        property_dict = {}
        for key in self.property_keys:
            if key in data:
                val = data[key]
                if isinstance(val, (int, float, list, np.ndarray)):
                    property_dict[key] = paddle.to_tensor(val, dtype="float32")
        return property_dict

    def _apply_transforms(self, structure: Structure) -> None:
        if self.convert_to_fractional:
            structure.convert_to_fractional()
        if self.niggli_reduce:
            structure.niggli_reduce()

    def _init_from_csv(self) -> None:
        from ppmat.datasets.build_structure import BuildStructure

        df = pd.read_csv(self.file_path)
        if "cif" not in df.columns:
            raise KeyError(
                f"CSV file does not contain 'cif' column. "
                f"Available columns: {list(df.columns)}"
            )

        self.structures = []
        for _, row in df.iterrows():
            pmg = BuildStructure.build_one(row["cif"], "cif_str")

            cell = paddle.to_tensor(pmg.lattice.matrix, dtype="float32")
            atomic_numbers = paddle.to_tensor(pmg.atomic_numbers, dtype="int64")
            pos = paddle.to_tensor(pmg.cart_coords, dtype="float32")

            property_dict = self._extract_property_dict(row)
            structure = Structure(
                cell=cell,
                atomic_numbers=atomic_numbers,
                pos=pos,
                property_dict=property_dict if property_dict else None,
                pos_is_fractional=False,
            )
            self._apply_transforms(structure)
            self.structures.append(OMATGData(structure))

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _init_from_parquet(self) -> None:
        df = pd.read_parquet(self.file_path)
        required_cols = ["positions", "cell", "atomic_numbers"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(
                    f"Parquet file missing '{col}'. "
                    f"Available columns: {list(df.columns)}"
                )

        self.structures = []
        for _, row in df.iterrows():
            atomic_numbers = paddle.to_tensor(
                np.asarray(row["atomic_numbers"], dtype=np.int64), dtype="int64"
            )
            pos = paddle.to_tensor(np.stack(row["positions"]), dtype="float32")
            cell = paddle.to_tensor(np.stack(row["cell"]), dtype="float32")

            property_dict = self._extract_property_dict(row)
            structure = Structure(
                cell=cell,
                atomic_numbers=atomic_numbers,
                pos=pos,
                property_dict=property_dict if property_dict else None,
                pos_is_fractional=False,
            )
            self._apply_transforms(structure)
            self.structures.append(OMATGData(structure))

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _create_structure(self, data: dict[str, Any]) -> OMATGData:
        cell = paddle.to_tensor(data["cell"], dtype="float32")
        atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
        pos = paddle.to_tensor(data["pos"], dtype="float32")

        property_dict = self._extract_property_dict(data)
        structure = Structure(
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            property_dict=property_dict if property_dict else None,
            pos_is_fractional=False,
        )
        self._apply_transforms(structure)
        return OMATGData(structure)

    def __len__(self) -> int:
        if self.lazy_storage:
            return len(self.keys)
        else:
            return len(self.structures)

    def __getitem__(self, idx: int) -> OMATGData:
        if self.lazy_storage:
            with self.env.begin() as txn:
                key = self.keys[idx]
                key_bytes = key.encode("ascii") if isinstance(key, str) else key

                value = txn.get(key_bytes)
                if value is None:
                    raise KeyError(f"Key {key} not found in LMDB")

                data = pickle.loads(value)

            return self._create_structure(data)
        else:
            return self.structures[idx]

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


__all__ = [
    "OMATGData",
]
