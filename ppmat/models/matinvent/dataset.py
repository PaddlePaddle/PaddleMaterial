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

import os
from typing import List
from typing import Optional

import numpy as np
import paddle
import paddle.io as io
from pymatgen.core.structure import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.utils import logger as ppmat_logger


def is_valid_structure(struc: Structure, min_volume: float = 0.1,
                       max_lattice_param: float = 25.0,
                       min_interatomic_dist: float = 0.5) -> bool:
    if struc is None or struc.num_sites == 0 or struc.volume <= min_volume:
        return False
    if max(struc.lattice.abc) > max_lattice_param:
        return False
    dmat = struc.distance_matrix.copy()
    np.fill_diagonal(dmat, np.inf)
    if dmat.min() < min_interatomic_dist:
        return False
    return True


def save_structures(structures: List[Structure], save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    with open(out_path, "w"):
        for struc in structures:
            write(out_path, adaptor.get_atoms(struc), append=True)
    ppmat_logger.info(f"Saved {len(structures)} structures to {out_path}")
    return out_path


class RLDataset(io.Dataset):
    def __init__(self, structures: List[Structure], rewards: np.ndarray,
                 transform: Optional[callable] = None):
        assert len(structures) == len(rewards)
        self.structures = structures
        self.rewards = rewards
        self.transform = transform
        ppmat_logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        s = self.structures[idx]
        frac = np.array([site.frac_coords for site in s.sites], dtype=np.float32)
        return {"structure_array": {
            "frac_coords": ConcatNumpyWarper(frac),
            "atom_types": ConcatNumpyWarper(
                np.array([site.specie.Z for site in s.sites], dtype=np.int64)),
            "lattice": s.lattice.matrix.astype(np.float32),
            "num_atoms": len(s),
            "reward": np.float32(self.rewards[idx]),
        }}

    @staticmethod
    def collate_fn(batch):
        out = DefaultCollator()(batch)
        na = out["structure_array"]["num_atoms"]
        out["structure_array"]["batch"] = np.concatenate(
            [np.full(n, i, dtype=np.int64) for i, n in enumerate(na)])
        return {k: _to_tensor(v) for k, v in out.items()}


def _to_tensor(v):
    if isinstance(v, np.ndarray):
        return paddle.to_tensor(v)
    if isinstance(v, dict):
        return {k: _to_tensor(vv) for k, vv in v.items()}
    return v


def create_rl_dataloader(structures: List[Structure], rewards: np.ndarray,
                         batch_size: int = 8, shuffle: bool = True,
                         num_workers: int = 0) -> io.DataLoader:
    dl = io.DataLoader(dataset=RLDataset(structures, rewards), batch_size=batch_size,
                       shuffle=shuffle, num_workers=num_workers, collate_fn=RLDataset.collate_fn)
    ppmat_logger.info(f"DataLoader: {len(dl.dataset)} samples, batch_size={batch_size}")
    return dl
