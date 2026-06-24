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
"""MD17 molecular dynamics dataset.

Each molecule has a single .npz file containing:
    - E: energies (N,)
    - F: forces  (N, num_atoms, 3)
    - R: positions (N, num_atoms, 3)
    - z: atomic numbers (num_atoms,)

Available molecules (8 total):
    aspirin, benzene_old, ethanol, malonaldehyde,
    naphthalene, salicylic, toluene, uracil
"""

import os
import os.path as osp

import numpy as np
import paddle
from paddle.io import Dataset

from ppmat.utils import logger
from ppmat.utils.download import get_datasets_path_from_url

# bcebos mirror (fast download within mainland China)
BCEBOS_URL = (
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
)
# MD5 extracted from bcebos ETag
BCEBOS_MD5 = "5d5d97a14ccef9e938e500f5f4601a59"

# Fallback individual molecule URLs (used when bcebos is unavailable)
MD17_MOLECULES = {
    "aspirin": "http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz",
    "benzene_old": "http://quantum-machine.org/gdml/data/npz/benzene_old_dft.npz",
    "ethanol": "http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
    "malonaldehyde": "http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz",
    "naphthalene": "http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz",
    "salicylic": "http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz",
    "toluene": "http://quantum-machine.org/gdml/data/npz/toluene_dft.npz",
    "uracil": "http://quantum-machine.org/gdml/data/npz/uracil_dft.npz",
}

# Train/val/test split sizes (same as DIG SphereNet defaults)
DEFAULT_SPLITS = {
    "aspirin": (1000, 500, 1000),
    "benzene_old": (1000, 500, 1000),
    "ethanol": (1000, 500, 1000),
    "malonaldehyde": (1000, 500, 1000),
    "naphthalene": (1000, 500, 1000),
    "salicylic": (1000, 500, 1000),
    "toluene": (1000, 500, 1000),
    "uracil": (1000, 500, 1000),
}


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset for energy and force prediction.

    Each sample contains atomic numbers, 3D positions, total energy,
    and atomic forces from DFT-based molecular dynamics trajectories.
    Downloads from the bcebos mirror by default, with fallback to
    individual molecule URLs from quantum-machine.org.

    Args:
        path: Root directory for storing raw and processed data.
        name: Molecule name (one of the 8 supported molecules).
        split: One of ``None`` (all data), ``'train'``, ``'val'``, or
            ``'test'``.
        train_size: Number of training samples.
        val_size: Number of validation samples.
        test_size: Number of test samples.
        force_key: Key name for forces in the output dict.
            Default: ``'force'``.
    """

    def __init__(
        self,
        path: str,
        name: str = "benzene_old",
        split=None,
        train_size=None,
        val_size=None,
        test_size=None,
        force_key="force",
    ):
        super().__init__()

        if name not in MD17_MOLECULES:
            raise ValueError(
                f"Unknown MD17 molecule '{name}'. "
                f"Supported: {list(MD17_MOLECULES.keys())}"
            )
        self.name = name
        self.force_key = force_key

        os.makedirs(path, exist_ok=True)
        self.root = path

        # Download raw data — prefer bcebos bundle, fall back to single-file URL
        raw_path = self._ensure_raw_data()

        # Load npz data
        data = np.load(raw_path)
        all_z = data["z"]  # (num_atoms,)
        all_pos = data["R"]  # (N, num_atoms, 3)
        all_energy = data["E"]  # (N,)
        all_forces = data["F"]  # (N, num_atoms, 3)

        # Build indices
        num_samples = all_pos.shape[0]
        indices = np.arange(num_samples)

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        # Apply split
        ts = train_size or DEFAULT_SPLITS[name][0]
        vs = val_size or DEFAULT_SPLITS[name][1]
        tes = test_size or DEFAULT_SPLITS[name][2]

        if split == "train":
            self._indices = indices[:ts]
        elif split == "val":
            self._indices = indices[ts : ts + vs]
        elif split == "test":
            self._indices = indices[ts + vs : ts + vs + tes]
        else:
            self._indices = indices

        self._z = paddle.to_tensor(all_z, dtype=paddle.int64)
        self._pos = paddle.to_tensor(all_pos, dtype=paddle.get_default_dtype())
        self._energy = paddle.to_tensor(all_energy, dtype=paddle.get_default_dtype())
        self._forces = paddle.to_tensor(all_forces, dtype=paddle.get_default_dtype())

        self.num_samples = len(self._indices)
        logger.info(
            f"MD17Dataset ({name}) ready: {self.num_samples} samples "
            f"(split={split})"
        )

    def _ensure_raw_data(self):
        """Download raw npz file to local cache if not already present.

        Tries the bcebos bundle (all 8 molecules in a single tar.gz) first.
        Falls back to the individual molecule URL when the bcebos download
        is unavailable or the extracted npz is missing.
        """
        raw_dir = osp.join(self.root, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Check if raw npz already exists from a previous download
        individual_path = osp.join(raw_dir, f"{self.name}_dft.npz")
        if osp.exists(individual_path):
            return individual_path

        # Try bcebos bundle download (distributed-safe via get_datasets_path_from_url)
        try:
            extract_dir = get_datasets_path_from_url(BCEBOS_URL, BCEBOS_MD5)
            bundle_npz_path = osp.join(extract_dir, f"{self.name}_dft.npz")
            if osp.exists(bundle_npz_path):
                return bundle_npz_path
        except Exception as e:
            logger.warning(
                f"bcebos download failed for MD17/{self.name}: {e}. "
                "Falling back to individual URL."
            )

        # Fallback to individual molecule URL
        import urllib.request

        url = MD17_MOLECULES[self.name]
        logger.info(f"Downloading MD17/{self.name} from {url} ...")
        try:
            urllib.request.urlretrieve(url, individual_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MD17/{self.name} from {url}: {e}. "
                "The bcebos mirror may also be unavailable."
            )
        return individual_path

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        return {
            "z": self._z.numpy(),
            "pos": self._pos[real_idx].numpy(),
            "energy": np.array([float(self._energy[real_idx])], dtype=np.float32),
            self.force_key: self._forces[real_idx].numpy(),
        }

    def __len__(self):
        return self.num_samples
