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

"""Asymmetric Unit (ASU) dataset. """

import os
from pathlib import Path
from typing import Optional

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE as _NUM_ELEMENTS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Supported ASU dataset names. Each dataset stores preprocessed asymmetric-unit
# data as NPZ archives under ``<data_dir>/<name>/{train,val,test}.npz``.
SUPPORTED_DATASETS = ("mp_20", "mp_20_assumeP1", "mpts_52")

_CANDIDATE_DIRS = [
    _PROJECT_ROOT / "data/data",
    Path("~").expanduser() / ".asu_data",
]


def resolve_asu_data_dir() -> Path:
    """Locate the ASU data directory.

    Priority: $ASU_DATA_DIR > project data dirs > ~/.asu_data.
    A directory is accepted if it contains any of the supported ASU
    dataset directories (npz files). Model-specific resources (e.g.
    ``ppmat.models.sgequidiff.vocabs``) live in their own module and
    resolve via their own resolvers rather than this directory.
    """
    env = os.getenv("ASU_DATA_DIR")
    if env:
        return Path(env)

    for candidate in _CANDIDATE_DIRS:
        if any((candidate / d).is_dir() for d in SUPPORTED_DATASETS):
            return candidate

    _candidate_paths = "\n".join([f"  * {d}" for d in _CANDIDATE_DIRS])
    raise FileNotFoundError(
        f"Cannot find ASU data directory.\n"
        f"\nTried paths:\n{_candidate_paths}\n"
        f"\nSolutions:\n"
        f"  1. Copy data to: {str(_CANDIDATE_DIRS[0])}\n"
        f"  2. Set environment variable: export ASU_DATA_DIR=/your/data/path\n"
        f"\nRequired: at least one of dataset dirs {list(SUPPORTED_DATASETS)}"
    )

# Supported data splits for ASU datasets (each has a corresponding NPZ archive).
SUPPORTED_SPLITS = ("train", "val", "test")

# Flat packed-array layout shared by all supported datasets
# (mp_20 / mp_20_assumeP1 / mpts_52). Each crystal is a 1-D float array:
#   [0]                : num_atoms (n)
#   [1]                : space group number (1-indexed)
#   [2:2+NE]           : composition (one-hot over NE elements)
#   [2+NE:5+NE]        : conventional lattice lengths (a, b, c)
#   [5+NE:8+NE]        : conventional lattice angles (alpha, beta, gamma)
#   [8+NE:8+NE+n]      : element indices
#   [8+NE+n:8+NE+2n]   : wyckoff indices
#   [8+NE+2n:8+NE+5n]  : fractional coords (n*3)
#   [8+NE+5n:8+NE+6n]  : wyckoff shape indices (optional)
_IDX_SG = 1
_IDX_COMP = 2
_IDX_LENGTHS = 2 + _NUM_ELEMENTS
_IDX_ANGLES = 5 + _NUM_ELEMENTS
_IDX_ATOMS = 8 + _NUM_ELEMENTS


def _parse_flat(flat: np.ndarray):
    """Parse one flat NPZ crystal array into its fields (see layout above)."""
    n = int(flat[0])
    sg = int(flat[_IDX_SG]) - 1
    comp = flat[_IDX_COMP:_IDX_LENGTHS].astype(np.float32)
    lengths = flat[_IDX_LENGTHS:_IDX_ANGLES].astype(np.float32)
    angles = flat[_IDX_ANGLES:_IDX_ATOMS].astype(np.float32)
    elems = flat[_IDX_ATOMS : _IDX_ATOMS + n].astype(np.int64)
    wycks = flat[_IDX_ATOMS + n : _IDX_ATOMS + 2 * n].astype(np.int64)
    fracs = (
        flat[_IDX_ATOMS + 2 * n : _IDX_ATOMS + 5 * n].reshape(n, 3).astype(np.float32)
    )
    wsi = None
    if len(flat) > _IDX_ATOMS + 5 * n:
        wsi = flat[_IDX_ATOMS + 5 * n : _IDX_ATOMS + 6 * n].astype(np.int64)
    return sg, comp, lengths, angles, n, elems, wycks, fracs, wsi


class AsymmetricUnitDataset(Dataset):
    """ASU-representation dataset (MP-20 / MP-20 assumeP1 / MPTS-52).

    Crystals are parsed lazily on first access so that building the dataset
    (and its DataLoader) stays cheap regardless of dataset size.
    """

    def __init__(
        self,
        name: str = "mp_20",
        split: str = "train",
        data_directory: Optional[Path] = None,
    ):
        super().__init__()
        assert split in SUPPORTED_SPLITS, f"unknown split: {split}"
        assert name in SUPPORTED_DATASETS, f"unknown name: {name}"

        if data_directory is None:
            data_directory = resolve_asu_data_dir()

        npz = np.load(Path(data_directory) / name / f"{split}.npz")
        self._flat_crystals = np.split(npz["packed"], npz["indices"])

    def __len__(self) -> int:
        return len(self._flat_crystals)

    def __getitem__(self, index: int) -> dict:
        sg, comp, lengths, angles, n, elems, wycks, fracs, wsi = _parse_flat(
            self._flat_crystals[index]
        )
        order = sorted(range(n), key=lambda j: (wycks[j], elems[j]))
        return {
            "space_group_indices": sg,
            "batch_chemistries": comp,
            "lattice_lengths": lengths,
            "lattice_angles": angles,
            "n_atoms_per_asu": n,
            "element_indices": ConcatData(elems[order]),
            "wyckoff_indices": ConcatData(wycks[order]),
            "wyckoff_shape_indices": ConcatData(
                wsi[order] if wsi is not None else np.zeros([n], dtype=np.int64)
            ),
            "frac_coords": ConcatData(fracs[order]),
        }
