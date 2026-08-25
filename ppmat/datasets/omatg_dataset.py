# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMatG structure dataset: reads LMDB/CSV/parquet into plain dict samples."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import pandas as pd
from ase import Atoms
from ase.symbols import Symbols

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.utils.lmdb_utils import lmdb_get
from ppmat.utils.lmdb_utils import lmdb_keys
from ppmat.utils.lmdb_utils import open_lmdb

# Lattice log-length mean/std per dataset (upstream FlowMM lattice_params_stats).
LATTICE_PARAMS = {
    "carbon_24": {
        "means": [0.9852757453918457, 1.3865314722061157, 1.7068126201629639],
        "stds": [0.14957907795906067, 0.20431114733219147, 0.2403733879327774],
    },
    "mp_20": {
        "means": [1.575442910194397, 1.7017393112182617, 1.9781638383865356],
        "stds": [0.24437622725963593, 0.26526379585266113, 0.3535512685775757],
    },
    "mpts_52": {
        "means": [1.6565313339233398, 1.8407557010650635, 2.1225264072418213],
        "stds": [0.2952289581298828, 0.3340013027191162, 0.41885802149772644],
    },
    "perov_5": {
        "means": [1.419227957725525, 1.419227957725525, 1.419227957725525],
        "stds": [0.07268335670232773, 0.07268335670232773, 0.07268335670232773],
    },
    "alex_mp_20": {
        "means": [1.5808929163076058, 1.74672046352959, 2.065243388307474],
        "stds": [0.27284015410437057, 0.2944785731740152, 0.30899526911753017],
    },
}


def sample_lattice_cell(lattice_means, lattice_stds):
    """Sample a cell from log-normal lengths and uniform angles."""
    from ase.geometry.cell import cellpar_to_cell

    lengths = paddle.exp(
        paddle.randn([3]) * paddle.to_tensor(lattice_stds)
        + paddle.to_tensor(lattice_means)
    )
    angles = paddle.rand([3]) * 60.0 + 60.0
    return cellpar_to_cell(paddle.concat((lengths, angles)).numpy())


class _OMATGStructure:
    """Single-crystal container with coordinate conversion, used to build samples."""

    def __init__(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        property_dict: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
    def symbols(self) -> list:
        return list(Symbols(self._atomic_numbers.numpy()))

    @property
    def pos(self) -> paddle.Tensor:
        return self._pos

    @property
    def pos_is_fractional(self) -> bool:
        return self._fractional

    @property
    def property_dict(self) -> Dict[str, Any]:
        return self._property_dict

    @property
    def metadata(self) -> Dict[str, Any]:
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
        from pymatgen.core import Element
        from pymatgen.core import Structure as PmgStructure

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


class OMATGStructureDataset(paddle.io.Dataset):
    """Read crystal structures from LMDB/CSV/parquet; lazy LMDB supported."""

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
            self._env = open_lmdb(self.file_path)
        return self._env

    def _init_from_lmdb(self) -> None:
        temp_env = open_lmdb(self.file_path)

        try:
            self.keys = lmdb_keys(temp_env)

            if not self.lazy_storage:
                self.structures = []
                for key in self.keys:
                    data = lmdb_get(temp_env, key)
                    self.structures.append(self._create_structure(data))
        finally:
            temp_env.close()

    def _extract_property_dict(self, data) -> Dict[str, Any]:
        property_dict = {}
        for key in self.property_keys:
            if key in data:
                val = data[key]
                if isinstance(val, (int, float, list, np.ndarray)):
                    property_dict[key] = paddle.to_tensor(val, dtype="float32")
        return property_dict

    def _apply_transforms(self, structure: _OMATGStructure) -> None:
        if self.convert_to_fractional:
            structure.convert_to_fractional()
        if self.niggli_reduce:
            structure.niggli_reduce()

    def _build_sample(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        property_dict: Optional[dict] = None,
    ) -> Dict[str, Any]:
        structure = _OMATGStructure(
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            property_dict=property_dict if property_dict else None,
            pos_is_fractional=False,
        )
        self._apply_transforms(structure)

        # Wrap fields in ConcatData so DefaultCollator concatenates atom-level
        # fields and stacks sample-level fields (same pattern as MP20Dataset).
        n = len(structure.atomic_numbers)
        return {
            "n_atoms": ConcatData(np.array([n], dtype="int64")),
            "species": ConcatData(structure.atomic_numbers.numpy()),
            "cell": ConcatData(structure.cell.numpy().reshape(1, 3, 3)),
            "pos": ConcatData(structure.pos.numpy()),
            "pos_is_fractional": ConcatData(
                np.array([structure.pos_is_fractional], dtype="bool")
            ),
        }

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
            pmg = BuildStructure.build_one(
                row["cif"], "cif_str", niggli=False, canocial=False
            )

            cell = paddle.to_tensor(pmg.lattice.matrix, dtype="float32")
            atomic_numbers = paddle.to_tensor(pmg.atomic_numbers, dtype="int64")
            pos = paddle.to_tensor(pmg.frac_coords, dtype="float32")

            property_dict = self._extract_property_dict(row)
            self.structures.append(
                self._build_sample(cell, atomic_numbers, pos, property_dict)
            )

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
            self.structures.append(
                self._build_sample(cell, atomic_numbers, pos, property_dict)
            )

        self.keys = list(range(len(self.structures)))
        self.lazy_storage = False

    def _create_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cell = paddle.to_tensor(data["cell"], dtype="float32")
        atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
        pos = paddle.to_tensor(data["pos"], dtype="float32")

        property_dict = self._extract_property_dict(data)
        return self._build_sample(cell, atomic_numbers, pos, property_dict)

    def __len__(self) -> int:
        if self.lazy_storage:
            return len(self.keys)
        else:
            return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._get_sample(idx)

    def _get_sample(self, idx: int) -> Dict[str, Any]:
        if self.lazy_storage:
            data = lmdb_get(self.env, self.keys[idx])
            return self._create_structure(data)
        else:
            return self.structures[idx]

    def __del__(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


__all__ = [
    "OMATGStructureDataset",
]
