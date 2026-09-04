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

from __future__ import annotations

import glob
import json
import os.path as osp
import zlib
from typing import Any

import lmdb
import numpy as np
from ase import Atoms
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger

UMA_DATASET_URL = (
    "https://paddle-org.bj.bcebos.com/paddlematerials/" "datasets/UMA/uma_datasets.zip"
)


class _UMAAseLMDBDataset(Dataset):
    """Read one UMA split from compressed ASE-LMDB files."""

    url: str
    dataset_subdir: str

    def __init__(
        self,
        build_graph_cfg: dict[str, Any],
        split: str,
        path: str | None = None,
        property_names: str | list[str] | tuple[str, ...] = ("energy", "forces"),
        build_structure_cfg: dict[str, Any] | None = None,
        pattern: str = "*.aselmdb",
        offset: int = 0,
        limit: int | None = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        transforms=None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}.")
        if path is None or not osp.exists(path):
            logger.message(
                f"The UMA {self.dataset_subdir} {split} split is not found. "
                "Downloading it now."
            )
            root_path = download.get_datasets_path_from_url(self.url)
            path = self._get_downloaded_split_path(root_path, split)

        self.paths = self._get_database_paths(path, pattern)
        self.databases = [
            lmdb.open(
                database_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            for database_path in self.paths
        ]
        self.indices = self._build_indices()
        self.indices = self.indices[int(offset) :]
        if limit is not None:
            self.indices = self.indices[: int(limit)]
        if not self.indices:
            raise ValueError(f"No records found in {path}.")

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "ase_atoms",
                "primitive": False,
                "niggli": False,
                "canocial": False,
                "num_cpus": 1,
            }
        self.build_structure = BuildStructure(**build_structure_cfg)
        self.graph_converter = build_graph_converter(build_graph_cfg)
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = tuple(property_names)
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.transforms = transforms

    def _get_downloaded_split_path(self, root_path: str, split: str) -> str:
        relative_path = self.get_split_relative_path(split)
        candidates = (
            osp.join(root_path, relative_path),
            osp.join(root_path, "uma_datasets", relative_path),
        )
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Cannot find UMA {self.dataset_subdir} {split} split under "
            f"{root_path}. Expected {relative_path}."
        )

    def get_split_relative_path(self, split: str) -> str:
        return osp.join(self.dataset_subdir, split)

    @staticmethod
    def _get_database_paths(path: str, pattern: str) -> list[str]:
        paths = (
            [path] if osp.isfile(path) else sorted(glob.glob(osp.join(path, pattern)))
        )
        if not paths:
            raise FileNotFoundError(f"No ASE-LMDB file found in {path}.")
        return paths

    def _build_indices(self) -> list[tuple[int, int]]:
        indices = []
        for database_index, database in enumerate(self.databases):
            with database.begin() as transaction:
                length = transaction.get(b"length")
                if length is not None:
                    row_ids = range(1, int(length.decode()) + 1)
                else:
                    row_ids = sorted(
                        int(key.decode())
                        for key, _ in transaction.cursor()
                        if key.isdigit()
                    )
            indices.extend((database_index, row_id) for row_id in row_ids)
        return indices

    @classmethod
    def _decode(cls, value):
        if isinstance(value, dict):
            if "__ndarray__" in value:
                shape, dtype, data = value["__ndarray__"][:3]
                return np.asarray(data, dtype=dtype).reshape(shape)
            return {key: cls._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._decode(item) for item in value]
        return value

    def read_data(self, database_index: int, row_id: int) -> dict[str, Any]:
        with self.databases[database_index].begin() as transaction:
            raw = transaction.get(str(row_id).encode())
        if raw is None:
            raise KeyError(f"Missing row {row_id} in {self.paths[database_index]}.")
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        return self._decode(json.loads(raw.decode()))

    @staticmethod
    def _get_property(record: dict[str, Any], key: str):
        value = record.get(key)
        if value is None and isinstance(record.get("data"), dict):
            value = record["data"].get(key)
        return value

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        database_index, row_id = self.indices[index]
        record = self.read_data(database_index, row_id)
        atoms = Atoms(
            numbers=record["numbers"],
            positions=record["positions"],
            cell=record.get("cell"),
            pbc=record.get("pbc", True),
            tags=record.get("tags"),
        )
        graph = self.graph_converter(self.build_structure(atoms))
        if graph is None:
            raise ValueError(f"Failed to build graph for sample {index}.")

        data: dict[str, Any] = {"graph": graph}
        if "energy" in self.property_names:
            energy = self._get_property(record, self.energy_key)
            data["energy"] = np.asarray(
                [np.nan if energy is None else energy], dtype="float32"
            )
        if "forces" in self.property_names:
            forces = self._get_property(record, self.forces_key)
            if forces is None:
                forces = np.full((len(atoms), 3), np.nan, dtype="float32")
            data["forces"] = ConcatNumpyWarper(forces).astype("float32")

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __del__(self):
        for database in getattr(self, "databases", []):
            database.close()


class UMAOC20Dataset(_UMAAseLMDBDataset):
    """Prepared OC20 S2EF data used by the UMA configuration."""

    url = UMA_DATASET_URL
    dataset_subdir = osp.join("oc20", "uma_aselmdb")


class UMAOMat24Dataset(_UMAAseLMDBDataset):
    """OMat24 rattled structures used by the UMA configuration."""

    url = UMA_DATASET_URL
    dataset_subdir = "omat24"

    def get_split_relative_path(self, split: str) -> str:
        return osp.join(self.dataset_subdir, split, "rattled-500")
