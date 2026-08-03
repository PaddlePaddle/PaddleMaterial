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

import numpy as np
from paddle.io import Dataset
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger


def _find_property(atoms, key: str):
    if key in atoms.info:
        return atoms.info[key]
    if key in atoms.arrays:
        return atoms.arrays[key]
    results = getattr(getattr(atoms, "calc", None), "results", {})
    return results.get(key) if isinstance(results, dict) else None


class UMAAseDBDataset(Dataset):
    """Single-task atomistic dataset backed by one or more ASE databases."""

    url: str | None = None
    md5: str | None = None

    def __init__(
        self,
        path: str,
        build_graph_cfg: dict[str, Any],
        property_names: list[str] | tuple[str, ...] = ("energy", "forces"),
        pattern: str = "*.aselmdb",
        select_args: dict[str, Any] | None = None,
        url: str | None = None,
        md5: str | None = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        transforms=None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if not osp.exists(path):
            dataset_url = url or self.url
            if dataset_url is None:
                raise FileNotFoundError(
                    f"Dataset path does not exist: {path}. Please configure a "
                    "download URL or prepare the dataset first."
                )
            logger.message("The UMA dataset is not found. Downloading it now.")
            downloaded = download.get_datasets_path_from_url(
                dataset_url, md5 or self.md5
            )
            candidate = osp.join(downloaded, osp.basename(path))
            path = candidate if osp.exists(candidate) else downloaded

        self.paths = self._collect_paths(path, pattern)
        self.property_names = tuple(property_names)
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.transforms = transforms
        self.graph_converter = build_graph_converter(build_graph_cfg)

        select_args = dict(select_args or {})
        offset = int(select_args.pop("offset", 0))
        limit = select_args.pop("limit", None)
        limit = None if limit is None else int(limit)

        self.databases = [self._connect(path) for path in self.paths]
        self.indices: list[tuple[int, int]] = []
        for db_index, database in enumerate(self.databases):
            row_ids = self._row_ids(database, select_args)
            self.indices.extend((db_index, row_id) for row_id in row_ids)
        self.indices = self.indices[offset:]
        if limit is not None:
            self.indices = self.indices[:limit]
        if not self.indices:
            raise ValueError(f"No records found in {path}.")

    @staticmethod
    def _collect_paths(path: str, pattern: str) -> list[str]:
        if osp.isfile(path):
            return [path]
        paths = sorted(glob.glob(osp.join(path, "**", pattern), recursive=True))
        if not paths:
            paths = sorted(glob.glob(path))
        if not paths:
            raise FileNotFoundError(f"No ASE database found from {path}.")
        return paths

    @staticmethod
    def _connect(path: str):
        import ase.db

        try:
            return ase.db.connect(path, readonly=True, use_lock_file=False)
        except (TypeError, ValueError):
            try:
                return ase.db.connect(path)
            except ValueError:
                import lmdb

                env = lmdb.open(
                    path,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                return {"env": env, "path": path}

    @staticmethod
    def _row_ids(database, select_args: dict[str, Any]) -> list[int]:
        if isinstance(database, dict):
            with database["env"].begin() as txn:
                length = txn.get(b"length")
                if length is not None:
                    return list(range(1, int(length.decode()) + 1))
                return sorted(
                    int(key.decode())
                    for key, _ in txn.cursor()
                    if key.decode().isdigit()
                )
        if hasattr(database, "ids") and not select_args:
            return [int(row_id) for row_id in database.ids]
        return [int(row.id) for row in database.select(**select_args)]

    @staticmethod
    def _decode(value):
        if isinstance(value, dict):
            if "__ndarray__" in value:
                shape, dtype, data = value["__ndarray__"][:3]
                return np.asarray(data, dtype=dtype).reshape(shape)
            return {key: UMAAseDBDataset._decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [UMAAseDBDataset._decode(item) for item in value]
        return value

    def _read(self, database, row_id: int):
        if not isinstance(database, dict):
            row = database._get_row(row_id)
            atoms = row.toatoms()
            if isinstance(row.data, dict):
                atoms.info.update(row.data)
            return atoms, row

        with database["env"].begin() as txn:
            raw = txn.get(str(row_id).encode())
        if raw is None:
            raise KeyError(f"Missing row {row_id} in {database['path']}.")
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        record = self._decode(json.loads(raw.decode()))

        from ase import Atoms

        atoms = Atoms(
            numbers=record["numbers"],
            positions=record["positions"],
            cell=record.get("cell"),
            pbc=record.get("pbc", True),
            tags=record.get("tags"),
        )
        if isinstance(record.get("data"), dict):
            atoms.info.update(record["data"])
        return atoms, record

    @staticmethod
    def _row_property(row, key: str):
        if isinstance(row, dict):
            value = row.get(key)
            if value is None and isinstance(row.get("data"), dict):
                value = row["data"].get(key)
            return value
        try:
            return row.get(key)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        database_index, row_id = self.indices[index]
        atoms, row = self._read(self.databases[database_index], row_id)
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.graph_converter(structure)
        if graph is None:
            raise ValueError(f"Failed to build graph for sample {index}.")

        data: dict[str, Any] = {"graph": graph}
        if "energy" in self.property_names:
            energy = self._row_property(row, self.energy_key)
            if energy is None:
                energy = _find_property(atoms, self.energy_key)
            data["energy"] = np.asarray(
                [np.nan if energy is None else energy], dtype="float32"
            )
        if "forces" in self.property_names:
            forces = self._row_property(row, self.forces_key)
            if forces is None:
                forces = _find_property(atoms, self.forces_key)
            if forces is None:
                forces = np.full((len(atoms), 3), np.nan, dtype="float32")
            data["forces"] = ConcatNumpyWarper(forces).astype("float32")

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __del__(self):
        for database in getattr(self, "databases", []):
            target = database.get("env") if isinstance(database, dict) else database
            if hasattr(target, "close"):
                target.close()
