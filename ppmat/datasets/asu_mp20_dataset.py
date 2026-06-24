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

"""Asymmetric Unit (ASU) MP-20 dataset."""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import paddle
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.asu_crystal import ASUCrystal, ImmutableASUCrystal


def _get_data_directory() -> Path:
    """Return data root directory: check env var first, fallback to project data/."""
    import os
    env = os.environ.get("SGEQUIDIFF_DATA_DIR", None)
    if env:
        return Path(env)
    # try project data directory
    candidates = [
        Path(__file__).resolve().parents[3] / "data",
        Path("~").expanduser() / ".sgequidiff_data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find data directory. Set SGEQUIDIFF_DATA_DIR or ensure data/ exists."
    )

class AsymmetricUnitDataset(Dataset):
    """ASU-representation Materials Project dataset (MP-20 / MPTS-52)."""

    def __init__(
        self,
        name: str = "mp_20",
        split: str = "train",
        data_directory: Optional[Path] = None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"unknown split: {split}"
        assert name in ["mp_20", "mp_20_assumeP1", "mpts_52"], f"unknown name: {name}"

        if name in ("mp_20", "mp_20_assumeP1"):
            self.max_atoms = 20
            self.max_elements = 7
        elif name == "mpts_52":
            self.max_atoms = 52
            self.max_elements = 7

        self.name = name
        self.split = split

        if data_directory is None:
            data_directory = _get_data_directory()

        data_path = Path(data_directory) / name / f"{split}.npz"
        properties_path = Path(data_directory) / name / f"{split}_properties.pkl"

        npz: dict = np.load(data_path)
        properties_df = pd.read_pickle(properties_path)  # reserved for future use

        self.indices_arr: np.ndarray = npz["indices"]
        self.packed: np.ndarray = npz["packed"]
        flat_crystals: List[np.ndarray] = np.split(self.packed, self.indices_arr)
        num_crystals = len(flat_crystals)

        self.data: List[ImmutableASUCrystal] = []
        _space_group_indices = []
        _composition_spaces = []
        _lattice_lengths = []
        _lattice_angles = []
        _n_atoms_per_asu = []

        self.padded_element_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_wyckoff_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_wyckoff_shape_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_frac_coords = -1.0 * paddle.ones(
            [num_crystals, self.max_atoms, 3], dtype=paddle.float32
        )
        self.atoms_mask = paddle.zeros([num_crystals, self.max_atoms], dtype=paddle.bool)

        for i, flat in enumerate(flat_crystals):
            crystal: ASUCrystal = ASUCrystal.from_flat(flat)
            num_atoms: int = crystal.num_atoms
            self.data.append(crystal.to_ImmutableASUCrystal())

            _space_group_indices.append(crystal.space_group_number - 1)
            _composition_spaces.append(crystal.composition_space)
            _lattice_lengths.append(crystal.conventional_lattice_lengths)
            _lattice_angles.append(crystal.conventional_lattice_angles)
            _n_atoms_per_asu.append(num_atoms)

            # sort by wyckoff_index, element_index lexicographically
            sorting_indices = paddle.to_tensor(
                sorted(
                    range(num_atoms),
                    key=lambda j: (
                        int(crystal.wyckoff_indices[j].item()),
                        int(crystal.element_indices[j].item()),
                    ),
                ),
                dtype=paddle.int64,
            )

            self.padded_element_indices[i, :num_atoms] = crystal.element_indices[sorting_indices]
            self.padded_wyckoff_indices[i, :num_atoms] = crystal.wyckoff_indices[sorting_indices]
            self.padded_wyckoff_shape_indices[i, :num_atoms] = crystal.wyckoff_shape_indices[sorting_indices]
            self.padded_frac_coords[i, :num_atoms] = crystal.conventional_frac_coords[sorting_indices]
            self.atoms_mask[i, :num_atoms] = True

        self.space_group_indices = paddle.to_tensor(_space_group_indices, dtype=paddle.int64)
        self.n_atoms_per_asu = paddle.to_tensor(_n_atoms_per_asu, dtype=paddle.int64)
        self.composition_spaces = paddle.stack(
            [paddle.to_tensor(c, dtype=paddle.float32) for c in _composition_spaces], axis=0
        )
        self.lattice_lengths = paddle.stack(
            [paddle.to_tensor(l, dtype=paddle.float32) for l in _lattice_lengths], axis=0
        )
        self.lattice_angles = paddle.stack(
            [paddle.to_tensor(a, dtype=paddle.float32) for a in _lattice_angles], axis=0
        )

    def __len__(self) -> int:
        return int(self.space_group_indices.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        """Return single sample dict compatible with DefaultCollator."""
        if not isinstance(index, int):
            raise TypeError(
                f"Expected int index, got {type(index)}. "
                "DataLoader with BatchSampler calls __getitem__ with int."
            )

        return self._get_single_item(index)

    @paddle.no_grad()
    def _get_single_item(self, index: int) -> dict:
        """Return single sample dict, all tensors without batch dim."""
        return {
            "space_group_indices": self.space_group_indices[index],          # scalar
            "batch_chemistries": self.composition_spaces[index],             # (chem_dim,)
            "lattice_lengths": self.lattice_lengths[index],                  # (3,)
            "lattice_angles": self.lattice_angles[index],                    # (3,)
            "n_atoms_per_asu": self.n_atoms_per_asu[index],                  # scalar
            "element_indices": self.padded_element_indices[index],            # (max_atoms,)
            "wyckoff_indices": self.padded_wyckoff_indices[index],            # (max_atoms,)
            "wyckoff_shape_indices": self.padded_wyckoff_shape_indices[index],# (max_atoms,)
            "frac_coords": self.padded_frac_coords[index],                   # (max_atoms, 3)
            "atoms_mask": self.atoms_mask[index],                             # (max_atoms,)
        }