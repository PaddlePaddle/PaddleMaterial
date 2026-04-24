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

import os
import os.path as osp
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from paddle.io import Dataset
from pymatgen.core import Structure

from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger


class ISO17Dataset(Dataset):
    """ISO17 small-molecule dataset loader for SchNet generalization benchmark."""
    name = "iso17"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/ISO17/iso17.tar.gz"
    md5 = None

    def __init__(
        self,
        path: str,
        subset: str = "train",
        property_names: Optional[Sequence[str]] = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        split_file: Optional[str] = None,
        seed: int = 42,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()
        self.path = path
        self.url = url if url is not None else self.url
        self.md5 = md5 if md5 is not None else self.md5
        self.auto_download = bool(auto_download)
        self.subset = subset
        self.seed = int(seed)
        self.transforms = transforms
        self.property_names = list(property_names or ["energy"])
        self.split_file = split_file
        if self.split_file is not None and not osp.isabs(self.split_file):
            self.split_file = osp.join(path, self.split_file)
        self.graph_converter = (
            build_graph_converter(build_graph_cfg) if build_graph_cfg is not None else None
        )

        positions, atomic_numbers, energies, isomer_ids = self._load_raw_data(path)
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.energies = energies
        self.isomer_ids = isomer_ids
        self.indices = self._split_indices(
            n_total=positions.shape[0],
            isomer_ids=isomer_ids,
            subset=subset,
            split_file=self.split_file,
            seed=self.seed,
        )

    @staticmethod
    def _find_npz(paths: Sequence[str], names: Sequence[str]) -> Optional[str]:
        for path in paths:
            if path is None or not osp.exists(path):
                continue
            for name in names:
                p = osp.join(path, name)
                if osp.exists(p):
                    return p
            for root, _, files in os.walk(path):
                for name in names:
                    if name in files:
                        return osp.join(root, name)
        return None

    @staticmethod
    def _unique_paths(paths: Sequence[str]) -> list[str]:
        uniq = []
        for path in paths:
            if path is None:
                continue
            norm_path = osp.normpath(path)
            if norm_path not in uniq:
                uniq.append(norm_path)
        return uniq

    def _ensure_raw_data(self, path: str) -> list[str]:
        names = ["iso17.npz", "ISO17.npz"]
        search_roots = [path]
        found = self._find_npz(search_roots, names)
        if found is not None:
            return self._unique_paths(search_roots)

        if not self.auto_download:
            return self._unique_paths(search_roots)
        if not self.url:
            return self._unique_paths(search_roots)

        logger.message(f"ISO17 data not found under {path}, downloading from {self.url}.")
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        if osp.isfile(downloaded_root):
            downloaded_root = osp.dirname(downloaded_root)
        search_roots.append(downloaded_root)
        search_roots.append(osp.join(downloaded_root, self.name))
        return self._unique_paths(search_roots)

    def _load_raw_data(self, path: str):
        os.makedirs(path, exist_ok=True)
        names = ["iso17.npz", "ISO17.npz"]
        search_roots = [path]
        npz_path = self._find_npz(search_roots, names)
        if npz_path is None:
            search_roots = self._ensure_raw_data(path)
            npz_path = self._find_npz(search_roots, names)
        if npz_path is None:
            raise FileNotFoundError(
                f"Cannot find ISO17 data in {path}. "
                f"Tried names: {list(names)}, searched roots: {search_roots}"
            )

        raw = np.load(npz_path)
        positions = self._pick_key(raw, ["R", "positions"]).astype(np.float32)
        atomic_numbers = self._pick_key(raw, ["z", "Z", "atomic_numbers"]).astype(np.int64)
        energies = self._pick_key(raw, ["E", "energies", "energy"]).astype(np.float32)
        if energies.ndim == 1:
            energies = energies[:, None]
        if atomic_numbers.ndim == 1:
            if atomic_numbers.shape[0] != positions.shape[1]:
                raise ValueError(
                    "ISO17 atomic numbers shape mismatch: "
                    f"z.shape={atomic_numbers.shape}, R.shape={positions.shape}"
                )
        elif atomic_numbers.ndim == 2:
            if atomic_numbers.shape[0] != positions.shape[0] or atomic_numbers.shape[1] != positions.shape[1]:
                raise ValueError(
                    "ISO17 per-sample atomic numbers shape mismatch: "
                    f"z.shape={atomic_numbers.shape}, R.shape={positions.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported ISO17 atomic number shape: {atomic_numbers.shape}"
            )
        if "isomer_ids" in raw:
            isomer_ids = raw["isomer_ids"].astype(np.int64)
        else:
            rng = np.random.default_rng(self.seed)
            isomer_ids = rng.integers(0, 600, size=(positions.shape[0],), dtype=np.int64)

        return positions, atomic_numbers, energies, isomer_ids

    @staticmethod
    def _pick_key(raw_obj, keys):
        for key in keys:
            if key in raw_obj:
                return raw_obj[key]
        raise KeyError(f"None of keys {keys} found in ISO17 file.")

    @staticmethod
    def _split_indices(
        n_total: int,
        isomer_ids: np.ndarray,
        subset: str,
        split_file: Optional[str],
        seed: int,
    ):
        if subset in ("all", None):
            return np.arange(n_total, dtype=np.int64)

        if split_file is not None and osp.exists(split_file):
            split = np.load(split_file)
            train_idx = split["train_idx"].astype(np.int64)
            val_idx = split["val_idx"].astype(np.int64)
            test_idx = split["test_idx"].astype(np.int64)
        else:
            rng = np.random.default_rng(seed)
            uniq = np.unique(isomer_ids)
            rng.shuffle(uniq)

            n_isomer = uniq.shape[0]
            n_train = int(0.7 * n_isomer)
            n_val = int(0.15 * n_isomer)

            train_set = set(uniq[:n_train].tolist())
            val_set = set(uniq[n_train:n_train + n_val].tolist())
            test_set = set(uniq[n_train + n_val:].tolist())

            train_idx = np.where(
                np.array([i in train_set for i in isomer_ids], dtype=bool)
            )[0].astype(np.int64)
            val_idx = np.where(
                np.array([i in val_set for i in isomer_ids], dtype=bool)
            )[0].astype(np.int64)
            test_idx = np.where(
                np.array([i in test_set for i in isomer_ids], dtype=bool)
            )[0].astype(np.int64)

            if split_file is not None:
                os.makedirs(osp.dirname(split_file), exist_ok=True)
                np.savez(
                    split_file,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                )

        train_idx = train_idx[(train_idx >= 0) & (train_idx < n_total)]
        val_idx = val_idx[(val_idx >= 0) & (val_idx < n_total)]
        test_idx = test_idx[(test_idx >= 0) & (test_idx < n_total)]

        if subset == "train":
            return train_idx
        if subset in ("val", "validation"):
            return val_idx
        if subset == "test":
            return test_idx
        raise ValueError(f"Unsupported subset: {subset}")

    def __len__(self):
        return len(self.indices)

    def _to_structure(self, pos: np.ndarray, atomic_numbers: np.ndarray) -> Structure:
        lattice = np.eye(3, dtype=np.float32) * 40.0
        species = [int(z) for z in atomic_numbers]
        return Structure(lattice, species, pos, coords_are_cartesian=True)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        pos = self.positions[real_idx]
        energy = self.energies[real_idx]
        isomer_id = self.isomer_ids[real_idx]
        if self.atomic_numbers.ndim == 1:
            z = self.atomic_numbers
        else:
            z = self.atomic_numbers[real_idx]

        data = {}
        if self.graph_converter is not None:
            structure = self._to_structure(pos, z)
            data["graph"] = self.graph_converter(structure)
        else:
            data["cart_coords"] = pos.astype(np.float32)
            data["atomic_numbers"] = z.astype(np.int64)

        for name in self.property_names:
            if name in ("energy", "energy_per_atom"):
                data[name] = energy.astype(np.float32)
            else:
                raise KeyError(
                    f"Unsupported property '{name}' for ISO17Dataset. "
                    "Use one of: energy, energy_per_atom."
                )
        data["isomer_id"] = np.array([isomer_id], dtype=np.int64)

        if self.transforms is not None:
            data = self.transforms(data)
        return data
