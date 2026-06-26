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
"""MD17 molecular dynamics dataset for energy and force prediction.

Each molecule trajectory is stored as a single .npz file containing:
    - E: energies (N,)
    - F: forces  (N, num_atoms, 3)
    - R: positions (N, num_atoms, 3)
    - z: atomic numbers (num_atoms,)

Supported molecules (8 total):
    aspirin, benzene_old, ethanol, malonaldehyde,
    naphthalene, salicylic, toluene, uracil

**STATS:**
+----------------+----------+--------+-------+----------+-------+
| Molecule       | #samples | #atoms | #tasks| #targets | Split |
+================+==========+========+=======+==========+=======+
| Aspirin        | 211,762  | 21     | 2     | E + F    | 1k/1k/R |
| Benzene (old)  | 627,983  | 12     | 2     | E + F    | 1k/1k/R |
| Ethanol        | 555,092  | 9      | 2     | E + F    | 1k/1k/R |
| Malonaldehyde  | 993,237  | 9      | 2     | E + F    | 1k/1k/R |
| Naphthalene    | 326,250  | 10     | 2     | E + F    | 1k/1k/R |
| Salicylic      | 320,231  | 16     | 2     | E + F    | 1k/1k/R |
| Toluene        | 442,790  | 15     | 2     | E + F    | 1k/1k/R |
| Uracil         | 133,770  | 12     | 2     | E + F    | 1k/1k/R |
+----------------+----------+--------+-------+----------+-------+
"""

import os
import os.path as osp
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle
from paddle.io import Dataset

from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.download import get_datasets_path_from_url

# Fallback individual molecule URLs (used when bcebos is unavailable)
_MOLECULE_URLS = {
    "aspirin": "http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz",
    "benzene_old": "http://quantum-machine.org/gdml/data/npz/benzene_old_dft.npz",
    "ethanol": "http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
    "malonaldehyde": "http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz",
    "naphthalene": "http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz",
    "salicylic": "http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz",
    "toluene": "http://quantum-machine.org/gdml/data/npz/toluene_dft.npz",
    "uracil": "http://quantum-machine.org/gdml/data/npz/uracil_dft.npz",
}

# Default train/val split sizes (matching DIG SphereNet defaults).
# Train=1000, Val=1000; Test = remaining samples.
_DEFAULT_SPLITS = {
    "aspirin": (1000, 1000),
    "benzene_old": (1000, 1000),
    "ethanol": (1000, 1000),
    "malonaldehyde": (1000, 1000),
    "naphthalene": (1000, 1000),
    "salicylic": (1000, 1000),
    "toluene": (1000, 1000),
    "uracil": (1000, 1000),
}

# Mapping from molecule names to bundle npz filenames
_BUNDLE_NPZ_MAP = {
    "aspirin": "md17_aspirin.npz",
    "benzene_old": "md17_benzene2017.npz",
    "ethanol": "md17_ethanol.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "naphthalene": "md17_naphthalene.npz",
    "salicylic": "md17_salicylic.npz",
    "toluene": "md17_toluene.npz",
    "uracil": "md17_uracil.npz",
}


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset for energy and force prediction.

    Each sample contains atomic numbers, 3D positions, total energy,
    and atomic forces from DFT-based molecular dynamics trajectories.

    Downloads from the bcebos mirror by default, with fallback to
    individual molecule URLs from quantum-machine.org.

    Args:
        path: Root directory for storing raw and processed data.
        name: Molecule name. Supported: aspirin, benzene_old, ethanol,
            malonaldehyde, naphthalene, salicylic, toluene, uracil.
        split: One of ``None`` (all data), ``'train'``, ``'val'``, or
            ``'test'``.
        train_size: Number of training samples.
        val_size: Number of validation samples.
        test_size: Number of test samples.
        force_key: Key name for forces in the output dict.
            Default: ``'force'``.
        transforms: Optional transforms to apply to each sample.
            Defaults to None.
        **kwargs: Additional arguments (for compatibility).
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
    md5 = "634cc25cc8a3fb0d99bd14245eb8dabd"
    name = "md17"

    def __init__(
        self,
        path: str,
        name: str = "benzene_old",
        split=None,
        train_size=None,
        val_size=None,
        test_size=None,
        force_key="force",
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()

        if name not in _MOLECULE_URLS:
            raise ValueError(
                f"Unknown MD17 molecule '{name}'. "
                f"Supported: {list(_MOLECULE_URLS.keys())}"
            )
        self.mol_name = name
        self.force_key = force_key
        self.transforms = transforms

        os.makedirs(path, exist_ok=True)
        self.root = path

        # --- Inline download (MP20 pattern) ---
        raw_dir = osp.join(self.root, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        individual_path = osp.join(raw_dir, f"{self.mol_name}_dft.npz")
        if osp.exists(individual_path):
            raw_path = individual_path
        else:
            # Try bcebos bundle first
            try:
                extract_dir = get_datasets_path_from_url(self.url, self.md5)
                bundle_rel = _BUNDLE_NPZ_MAP[self.mol_name]
                bundle_found = None
                for sub in ["", "md17/"]:
                    candidate = osp.join(extract_dir, sub, bundle_rel)
                    if osp.exists(candidate):
                        bundle_found = candidate
                        break
                if bundle_found is not None:
                    raw_path = bundle_found
                else:
                    raw_path = None
            except Exception as e:
                logger.warning(
                    f"bcebos download failed for MD17/{self.mol_name}: {e}. "
                    "Falling back to individual URL."
                )
                raw_path = None

            if raw_path is None:
                # Fallback to individual molecule URL
                import urllib.request
                url = _MOLECULE_URLS[self.mol_name]
                logger.info(f"Downloading MD17/{self.mol_name} from {url} ...")
                try:
                    urllib.request.urlretrieve(url, individual_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download MD17/{self.mol_name} from {url}: {e}. "
                        "The bcebos mirror may also be unavailable."
                    )
                raw_path = individual_path

        # Instantiate graph converter if configured
        if build_graph_cfg is not None:
            self.graph_converter = build_graph_converter(build_graph_cfg)
        else:
            self.graph_converter = None

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

        # Apply split (DIG convention: train=1000, val=1000, test=rest)
        ts = train_size or _DEFAULT_SPLITS[name][0]
        vs = val_size or _DEFAULT_SPLITS[name][1]

        if split == "train":
            self._indices = indices[:ts]
        elif split == "val":
            self._indices = indices[ts : ts + vs]
        elif split == "test":
            self._indices = indices[ts + vs :]
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

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        sample = {
            "z": self._z.numpy(),
            "pos": self._pos[real_idx].numpy(),
            "energy": np.array([float(self._energy[real_idx])], dtype=np.float32),
            self.force_key: self._forces[real_idx].numpy(),
        }
        if self.graph_converter is not None:
            pos_t = paddle.to_tensor(sample["pos"])
            batch_t = paddle.zeros([sample["z"].shape[0]], dtype=paddle.int64)
            ei = self.graph_converter(pos_t, batch_t)
            sample["edge_index"] = ei.numpy()
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.num_samples
