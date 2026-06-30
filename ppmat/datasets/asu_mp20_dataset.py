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

"""Asymmetric Unit (ASU) dataset. Pure data loader, no model-specific imports."""
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.sgequidiff.constants import NUM_ELEMENTS as _NUM_ELEMENTS


def _parse_flat(flat: np.ndarray):
    """Parse flat NPZ crystal array into fields. Format:
       [0] num_atoms | [1] sg | [2:2+NE] composition | [2+NE:5+NE] lengths |
       [5+NE:8+NE] angles | [8+NE:8+NE+N] elements | [+N:+2N] wyckoffs |
       [+2N:+5N] frac_coords | [+5N:+6N] wyckoff_shape (optional)."""
    NE = _NUM_ELEMENTS
    n = int(flat[0])
    sg = int(flat[1]) - 1
    comp = flat[2:2 + NE].astype(np.float32)
    lengths = flat[2 + NE:5 + NE].astype(np.float32)
    angles = flat[5 + NE:8 + NE].astype(np.float32)
    elems = flat[8 + NE:8 + NE + n].astype(np.int64)
    wycks = flat[8 + NE + n:8 + NE + 2 * n].astype(np.int64)
    fracs = flat[8 + NE + 2 * n:8 + NE + 5 * n].reshape(n, 3).astype(np.float32)
    wsi = None
    if len(flat) > 8 + NE + 5 * n:
        wsi = flat[8 + NE + 5 * n:8 + NE + 6 * n].astype(np.int64)
    return sg, comp, lengths, angles, n, elems, wycks, fracs, wsi


class AsymmetricUnitDataset(Dataset):
    """ASU-representation dataset (MP-20 / MPTS-52) with ConcatData for batch collation."""

    def __init__(
        self,
        name: str = "mp_20",
        split: str = "train",
        data_directory: Optional[Path] = None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"unknown split: {split}"
        assert name in ["mp_20", "mp_20_assumeP1", "mpts_52"], f"unknown name: {name}"

        if data_directory is None:
            env = os.environ.get("SGEQUI_DATA_DIR", None)
            if env:
                data_directory = Path(env)
            else:
                candidates = [
                    Path(__file__).resolve().parents[2] / "data" / "data",
                    Path(__file__).resolve().parents[3] / "data",
                    Path("~").expanduser() / ".sgequidiff_data",
                ]
                for c in candidates:
                    if c.exists():
                        data_directory = c
                        break
                if data_directory is None:
                    raise FileNotFoundError(
                        "Cannot find data directory. Set SGEQUI_DATA_DIR or ensure data/ exists."
                    )

        npz = np.load(Path(data_directory) / name / f"{split}.npz")
        flat_crystals = np.split(npz["packed"], npz["indices"])
        num_crystals = len(flat_crystals)

        self._sg = np.empty([num_crystals], dtype=np.int64)
        self._comp = np.empty([num_crystals, _NUM_ELEMENTS], dtype=np.float32)
        self._lengths = np.empty([num_crystals, 3], dtype=np.float32)
        self._angles = np.empty([num_crystals, 3], dtype=np.float32)
        self._n_atoms = np.empty([num_crystals], dtype=np.int64)
        self._elems: List[np.ndarray] = []
        self._wycks: List[np.ndarray] = []
        self._wsi: List[np.ndarray] = []
        self._fracs: List[np.ndarray] = []

        for i, flat in enumerate(flat_crystals):
            sg, comp, lengths, angles, n, elems, wycks, fracs, wsi = _parse_flat(flat)
            order = sorted(range(n), key=lambda j: (wycks[j], elems[j]))
            self._sg[i] = sg
            self._comp[i] = comp
            self._lengths[i] = lengths
            self._angles[i] = angles
            self._n_atoms[i] = n
            self._elems.append(elems[order])
            self._wycks.append(wycks[order])
            self._fracs.append(fracs[order])
            self._wsi.append(wsi[order] if wsi is not None else np.zeros([n], dtype=np.int64))

    def __len__(self) -> int:
        return self._sg.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {
            "space_group_indices": self._sg[index],
            "batch_chemistries": self._comp[index],
            "lattice_lengths": self._lengths[index],
            "lattice_angles": self._angles[index],
            "n_atoms_per_asu": self._n_atoms[index],
            "element_indices": ConcatData(self._elems[index]),
            "wyckoff_indices": ConcatData(self._wycks[index]),
            "wyckoff_shape_indices": ConcatData(self._wsi[index]),
            "frac_coords": ConcatData(self._fracs[index]),
        }
