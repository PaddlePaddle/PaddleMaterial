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

MD17_MOLECULES = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic_acid",
    "toluene",
    "uracil",
]


class MD17Dataset(Dataset):
    """MD17/rMD17 small-molecule dataset loader for SchNet-style training."""
    name = "md17"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
    md5 = None

    def __init__(
        self,
        path: str,
        molecule: str,
        subset: str = "train",
        property_names: Optional[Sequence[str]] = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = None,
        seed: int = 42,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()
        if molecule not in MD17_MOLECULES:
            raise ValueError(
                f"Unknown molecule '{molecule}', expected one of {MD17_MOLECULES}."
            )
        self.path = path
        self.molecule = molecule
        self.url = url if url is not None else self.url
        self.md5 = md5 if md5 is not None else self.md5
        self.auto_download = bool(auto_download)
        self.subset = subset
        self.seed = int(seed)
        self.transforms = transforms
        self.property_names = list(property_names or ["energy", "force"])
        self.split_file = split_file
        if self.split_file is not None and not osp.isabs(self.split_file):
            self.split_file = osp.join(path, self.split_file)
        self.graph_converter = (
            build_graph_converter(build_graph_cfg) if build_graph_cfg is not None else None
        )

        positions, atomic_numbers, energies, forces = self._load_raw_data(path, molecule)
        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.energies = energies
        self.forces = forces

        self.indices = self._split_indices(
            n_total=positions.shape[0],
            subset=subset,
            train_size=num_train if num_train is not None else train_size,
            val_size=num_val if num_val is not None else val_size,
            test_size=num_test,
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

    def _ensure_raw_data(self, path: str, molecule: str) -> list[str]:
        names = [
            f"{molecule}.npz",
            f"rmd17_{molecule}.npz",
            f"md17_{molecule}.npz",
        ]
        search_roots = [path]
        found = self._find_npz(search_roots, names)
        if found is not None:
            return self._unique_paths(search_roots)

        if not self.auto_download:
            return self._unique_paths(search_roots)
        if not self.url:
            return self._unique_paths(search_roots)

        logger.message(f"MD17 data not found under {path}, downloading from {self.url}.")
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        if osp.isfile(downloaded_root):
            downloaded_root = osp.dirname(downloaded_root)
        search_roots.append(downloaded_root)
        search_roots.append(osp.join(downloaded_root, self.name))
        return self._unique_paths(search_roots)

    def _load_raw_data(self, path: str, molecule: str):
        os.makedirs(path, exist_ok=True)
        names = [
            f"{molecule}.npz",
            f"rmd17_{molecule}.npz",
            f"md17_{molecule}.npz",
        ]
        search_roots = [path]
        npz_path = self._find_npz(search_roots, names)
        if npz_path is None:
            search_roots = self._ensure_raw_data(path, molecule)
            npz_path = self._find_npz(search_roots, names)
        if npz_path is None:
            raise FileNotFoundError(
                f"Cannot find MD17 file for '{molecule}' in {path}. "
                f"Tried names: {list(names)}, searched roots: {search_roots}"
            )

        raw = np.load(npz_path)
        positions = self._pick_key(raw, ["R", "positions"]).astype(np.float32)
        atomic_numbers = self._pick_key(raw, ["z", "Z", "atomic_numbers"]).astype(np.int64)
        energies = self._pick_key(raw, ["E", "energies", "energy"]).astype(np.float32)
        forces = self._pick_key(raw, ["F", "forces"]).astype(np.float32)

        if energies.ndim == 1:
            energies = energies[:, None]
        return positions, atomic_numbers, energies, forces

    @staticmethod
    def _pick_key(raw_obj, keys):
        for key in keys:
            if key in raw_obj:
                return raw_obj[key]
        raise KeyError(f"None of keys {keys} found in MD17 file.")

    @staticmethod
    def _split_indices(
        n_total: int,
        subset: str,
        train_size: Optional[int],
        val_size: Optional[int],
        test_size: Optional[int],
        split_file: Optional[str],
        seed: int,
    ):
        if split_file is not None and osp.exists(split_file):
            split = np.load(split_file)
            train_idx = split["train_idx"].astype(np.int64)
            val_idx = split["val_idx"].astype(np.int64)
            test_idx = split["test_idx"].astype(np.int64)
        else:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n_total).astype(np.int64)

            if train_size is None:
                train_size = int(0.8 * n_total)
            if val_size is None:
                val_size = int(0.1 * n_total)
            train_size = max(0, min(int(train_size), n_total))
            val_size = max(0, min(int(val_size), n_total - train_size))
            if test_size is None:
                test_size = n_total - train_size - val_size
            else:
                test_size = max(0, min(int(test_size), n_total - train_size - val_size))

            train_end = train_size
            val_end = train_size + val_size
            test_end = val_end + test_size
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:test_end]
            if test_end < n_total:
                test_idx = np.concatenate([test_idx, indices[test_end:]], axis=0)

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
        if subset in ("all", None):
            return np.arange(n_total, dtype=np.int64)
        raise ValueError(f"Unsupported subset: {subset}")

    def __len__(self):
        return len(self.indices)

    def _to_structure(self, pos: np.ndarray) -> Structure:
        lattice = np.eye(3, dtype=np.float32) * 40.0
        species = [int(z) for z in self.atomic_numbers]
        return Structure(lattice, species, pos, coords_are_cartesian=True)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        pos = self.positions[real_idx]
        energy = self.energies[real_idx]
        force = self.forces[real_idx]

        data = {}
        if self.graph_converter is not None:
            structure = self._to_structure(pos)
            data["graph"] = self.graph_converter(structure)
        else:
            data["cart_coords"] = pos.astype(np.float32)
            data["atomic_numbers"] = self.atomic_numbers.astype(np.int64)

        for name in self.property_names:
            if name in ("energy", "energy_per_atom"):
                data[name] = energy.astype(np.float32)
            elif name in ("force", "forces"):
                data[name] = force.astype(np.float32)
            else:
                raise KeyError(
                    f"Unsupported property '{name}' for MD17Dataset. "
                    "Use one of: energy, energy_per_atom, force, forces."
                )

        if self.transforms is not None:
            data = self.transforms(data)
        return data
