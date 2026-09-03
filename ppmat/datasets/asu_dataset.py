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

import os.path as osp

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.utils import download
from ppmat.utils import logger

# Download metadata of the ASU dataset archives (bcebos release addresses).
_ASU_DATASETS = {
    "mp_20": {
        "url": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/asu/mp_20_asu.zip",
        "md5": "c8dc162555808bf8dc0183b840209f6a",
    },
    "mpts_52": {
        "url": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/asu/mpts_52_asu.zip",
        "md5": "bdbfdad0352bbf32afb1ee6561cea97b",
    },
}


class AsymmetricUnitDataset(Dataset):
    """ASU-representation dataset (MP-20).

    Crystals are stored as flat packed arrays in NPZ archives
    (``<split>.npz``) and parsed lazily on first access so that building
    the dataset (and its DataLoader) stays cheap regardless of dataset size.

    **Data Format**

    Each crystal is a 1-D float array in the packed layout (NE = element
    encoding size):

    - ``[0]``: num_atoms (n)
    - ``[1]``: space group number (1-indexed)
    - ``[2:2+NE]``: composition (one-hot over NE elements)
    - ``[2+NE:5+NE]``: conventional lattice lengths (a, b, c)
    - ``[5+NE:8+NE]``: conventional lattice angles (alpha, beta, gamma)
    - ``[8+NE:8+NE+n]``: element indices
    - ``[8+NE+n:8+NE+2n]``: wyckoff indices
    - ``[8+NE+2n:8+NE+5n]``: fractional coords (n*3)
    - ``[8+NE+5n:8+NE+6n]``: wyckoff shape indices (optional)

    Args:
        path (str, optional): The path of the dataset, if path is not exists,
            it will be downloaded. Defaults to "./data/mp_20/train.npz".
    """

    name = "mp_20"
    url = _ASU_DATASETS[name]["url"]
    md5 = _ASU_DATASETS[name]["md5"]

    # Packed-layout offsets that depend on the element encoding size
    # (see Data Format in the class docstring).
    _IDX_LENGTHS = 2 + ELEMENT_ENCODING_SIZE
    _IDX_ANGLES = 5 + ELEMENT_ENCODING_SIZE
    _IDX_ATOMS = 8 + ELEMENT_ENCODING_SIZE

    def __init__(self, path: str = "./data/mp_20/train.npz", **kwargs):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        self._flat_crystals, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

    def read_data(self, path: str):
        """Read the packed NPZ archive and split it into per-crystal arrays."""
        npz = np.load(path)
        flat_crystals = np.split(npz["packed"], npz["indices"])
        return flat_crystals, len(flat_crystals)

    def parse_flat_crystal(self, flat: np.ndarray) -> dict:
        """Parse one packed crystal array into a dict of its fields.

        Archives without the optional trailing wyckoff-shape segment are
        normalized with zero shape indices (no shape decomposition).
        """
        n = int(flat[0])
        idx_atoms = self._IDX_ATOMS
        if len(flat) > idx_atoms + 5 * n:
            shape_indices = flat[idx_atoms + 5 * n : idx_atoms + 6 * n].astype(np.int64)
        else:
            shape_indices = np.zeros(n, dtype=np.int64)
        return {
            "num_atoms": n,
            "space_group_index": int(flat[1]) - 1,
            "batch_chemistries": flat[2 : self._IDX_LENGTHS].astype(np.float32),
            "lattice_lengths": flat[self._IDX_LENGTHS : self._IDX_ANGLES].astype(
                np.float32
            ),
            "lattice_angles": flat[self._IDX_ANGLES : idx_atoms].astype(np.float32),
            "element_indices": flat[idx_atoms : idx_atoms + n].astype(np.int64),
            "wyckoff_indices": flat[idx_atoms + n : idx_atoms + 2 * n].astype(np.int64),
            "frac_coords": flat[idx_atoms + 2 * n : idx_atoms + 5 * n]
            .reshape(n, 3)
            .astype(np.float32),
            "wyckoff_shape_indices": shape_indices,
        }

    def __getitem__(self, index: int) -> dict:
        fields = self.parse_flat_crystal(self._flat_crystals[index])
        # Stable sort by (wyckoff index, element index); np.lexsort takes the
        # keys in reverse order, so the last key is the primary one.
        order = np.lexsort((fields["element_indices"], fields["wyckoff_indices"]))
        return {
            "space_group_indices": fields["space_group_index"],
            "batch_chemistries": fields["batch_chemistries"],
            "lattice_lengths": fields["lattice_lengths"],
            "lattice_angles": fields["lattice_angles"],
            "n_atoms_per_asu": fields["num_atoms"],
            "element_indices": ConcatData(fields["element_indices"][order]),
            "wyckoff_indices": ConcatData(fields["wyckoff_indices"][order]),
            "wyckoff_shape_indices": ConcatData(fields["wyckoff_shape_indices"][order]),
            "frac_coords": ConcatData(fields["frac_coords"][order]),
        }

    def __len__(self) -> int:
        return self.num_samples


class MPTS52ASUDataset(AsymmetricUnitDataset):
    """ASU-representation dataset (MPTS-52)."""

    name = "mpts_52"
    url = _ASU_DATASETS[name]["url"]
    md5 = _ASU_DATASETS[name]["md5"]
