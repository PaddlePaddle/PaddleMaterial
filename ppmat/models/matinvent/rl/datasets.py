# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional

import numpy as np
import paddle
import paddle.io as io
from pymatgen.core.structure import Structure

from ppmat.utils import logger as ppmat_logger


class RLDataset(io.Dataset):
    def __init__(
        self,
        structures: List[Structure],
        rewards: np.ndarray,
        transform: Optional[callable] = None,
    ):
        assert len(structures) == len(rewards)
        self.structures = structures
        self.rewards = rewards
        self.transform = transform
        ppmat_logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        reward = self.rewards[idx]
        frac_coords = np.array([site.frac_coords for site in structure.sites])
        lattice = structure.lattice.matrix
        atom_types = np.array([site.specie.Z for site in structure.sites])
        num_atoms = len(structure)
        sample = {
            "frac_coords": frac_coords.astype(np.float32),
            "lattice": lattice.astype(np.float32),
            "atom_types": atom_types.astype(np.int64),
            "num_atoms": num_atoms,
            "reward": reward.astype(np.float32),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    frac_coords_list = [item["frac_coords"] for item in batch]
    lattice_list = [item["lattice"] for item in batch]
    atom_types_list = [item["atom_types"] for item in batch]
    num_atoms_list = [item["num_atoms"] for item in batch]
    rewards = np.array([item["reward"] for item in batch], dtype=np.float32)

    frac_coords = np.concatenate(frac_coords_list, axis=0)
    atom_types = np.concatenate(atom_types_list, axis=0)
    lattice = np.stack(lattice_list, axis=0)
    num_atoms = np.array(num_atoms_list, dtype=np.int64)

    batch_idx = []
    for i, n_atoms in enumerate(num_atoms):
        batch_idx.extend([i] * n_atoms)
    batch_idx = np.array(batch_idx, dtype=np.int64)

    structure_array = {
        "frac_coords": paddle.to_tensor(frac_coords),
        "lattice": paddle.to_tensor(lattice),
        "atom_types": paddle.to_tensor(atom_types),
        "num_atoms": paddle.to_tensor(num_atoms),
        "batch": paddle.to_tensor(batch_idx),
        "reward": paddle.to_tensor(rewards),
    }
    return {"structure_array": structure_array}


def create_rl_dataloader(
    structures: List[Structure],
    rewards: np.ndarray,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> io.DataLoader:
    dataset = RLDataset(structures, rewards)
    dataloader = io.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    ppmat_logger.info(
        f"Created DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}"
    )
    return dataloader
