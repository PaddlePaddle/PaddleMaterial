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

"""
Data utilities for MatInvent RL module.

This module provides common data processing utilities
that reuse ppmat built-in functionality.
"""

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.io as io
from pymatgen.core.structure import Structure

from ppmat.utils import logger as ppmat_logger


class RLDataset(io.Dataset):
    """Dataset for RL fine-tuning of diffusion models.

    This class follows the same pattern as ppmat.datasets.* classes
    but is specialized for RL training with rewards.

    Args:
        structures: List of pymatgen Structure objects
        rewards: Array of reward values for each structure
        transform: Optional transform function
    """

    def __init__(
        self,
        structures: List[Structure],
        rewards: np.ndarray,
        transform: Optional[callable] = None,
    ):
        assert len(structures) == len(
            rewards
        ), "Structures and rewards must have same length"
        self.structures = structures
        self.rewards = rewards
        self.transform = transform

        ppmat_logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        reward = self.rewards[idx]

        # Convert pymatgen Structure to model input format
        # This follows the same pattern as ppmat.datasets.* classes
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
    """Collate function for RL dataset.

    This function follows ppmat's collate function pattern,
    creating a batched structure_array.

    Args:
        batch: List of samples from RLDataset

    Returns:
        Batched data with structure_array
    """
    # Extract all components
    frac_coords_list = [item["frac_coords"] for item in batch]
    lattice_list = [item["lattice"] for item in batch]
    atom_types_list = [item["atom_types"] for item in batch]
    num_atoms_list = [item["num_atoms"] for item in batch]
    rewards = np.array([item["reward"] for item in batch], dtype=np.float32)

    # Concatenate frac_coords and atom_types
    frac_coords = np.concatenate(frac_coords_list, axis=0)
    atom_types = np.concatenate(atom_types_list, axis=0)
    lattice = np.stack(lattice_list, axis=0)
    num_atoms = np.array(num_atoms_list, dtype=np.int64)

    # Create batch index (following ppmat's scatter pattern)
    batch_idx = []
    for i, n_atoms in enumerate(num_atoms):
        batch_idx.extend([i] * n_atoms)
    batch_idx = np.array(batch_idx, dtype=np.int64)

    # Create structure_array
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
    """Create a DataLoader for RL fine-tuning.

    This function follows ppmat.datasets.build_dataloader pattern.

    Args:
        structures: List of pymatgen Structure objects
        rewards: Array of reward values
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PaddlePaddle DataLoader
    """
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


def filter_by_reward(
    structures: List[Structure],
    rewards: np.ndarray,
    data_list: List,
    topk_ratio: float = 0.5,
) -> Tuple[List[Structure], np.ndarray, List]:
    """Filter samples by reward (top-k selection).

    Args:
        structures: List of pymatgen Structure objects
        rewards: Array of reward values
        data_list: List of data dictionaries
        topk_ratio: Ratio of top samples to keep (0.0 to 1.0)

    Returns:
        Tuple of (filtered_structures, filtered_rewards, filtered_data)
    """
    assert 0.0 < topk_ratio <= 1.0, "topk_ratio must be in (0, 1]"

    n_keep = max(1, int(len(structures) * topk_ratio))

    # Sort by reward (descending)
    indices = np.argsort(rewards)[::-1][:n_keep]

    filtered_structures = [structures[i] for i in indices]
    filtered_rewards = rewards[indices]
    filtered_data = [data_list[i] for i in indices]

    ppmat_logger.info(
        f"Filtered {len(structures)} -> {len(filtered_structures)} samples "
        f"(topk_ratio={topk_ratio}, "
        f"reward_range=[{filtered_rewards.min():.4f}, "
        f"{filtered_rewards.max():.4f}])"
    )

    return filtered_structures, filtered_rewards, filtered_data
