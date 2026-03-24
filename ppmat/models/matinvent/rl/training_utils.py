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
Training utilities for MatInvent RL module.

This module provides training-related utilities that reuse ppmat built-in functionality.
"""

import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
from pymatgen.core.structure import Structure

from ppmat.utils import logger as ppmat_logger


def is_valid_structure(
    struc: Structure,
    min_volume: float = 0.1,
    max_lattice_param: float = 25.0,
    min_interatomic_dist: float = 0.5,
) -> bool:
    """Check if a structure is valid.

      - volume > min_volume A^3
      - minimum interatomic distance > min_interatomic_dist A
      - max lattice parameter < max_lattice_param A

    Args:
        struc: pymatgen Structure
        min_volume: Minimum volume threshold (default: 0.1)
        max_lattice_param: Maximum lattice parameter (default: 25.0)
        min_interatomic_dist: Minimum interatomic distance (default: 0.5)

    Returns:
        True if structure is valid
    """
    try:
        if struc is None:
            return False
        if struc.num_sites == 0:
            return False
        if struc.volume <= min_volume:
            return False
        # max lattice parameter check
        if max(struc.lattice.abc) > max_lattice_param:
            return False
        # minimum interatomic distance check
        dmat = struc.distance_matrix.copy()
        np.fill_diagonal(dmat, np.inf)
        if dmat.min() < min_interatomic_dist:
            return False
        return True
    except Exception:
        return False


def save_structures(
    structures: List[Structure],
    save_dir: str,
    filename: str,
) -> str:
    """Save pymatgen structures to ASE extxyz format.

    Args:
        structures: List of pymatgen Structure objects
        save_dir: Directory to save the file
        filename: Output filename

    Returns:
        Path to the saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    # Convert pymatgen structures to ASE atoms and write
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()

    with open(out_path, "w"):
        for struc in structures:
            atoms = adaptor.get_atoms(struc)
            write(out_path, atoms, append=True)

    ppmat_logger.info(f"Saved {len(structures)} structures to {out_path}")
    return out_path


def filter_valid_structures(
    data_list: List,
    structures: List[Structure],
    validity_check: Optional[callable] = None,
) -> Tuple[List, List[Structure]]:
    """Filter out invalid structures.

    Args:
        data_list: List of data samples
        structures: List of pymatgen Structure objects
        validity_check: Optional custom validity check function

    Returns:
        Tuple of (filtered_data, filtered_structures)
    """
    if validity_check is None:
        validity_check = is_valid_structure

    valid_data = []
    valid_structures = []

    for data, struc in zip(data_list, structures):
        if validity_check(struc):
            valid_data.append(data)
            valid_structures.append(struc)

    if len(valid_structures) < len(structures):
        ppmat_logger.info(
            f"Filtered {len(structures) - len(valid_structures)} invalid structures, "
            f"{len(valid_structures)} valid structures remaining"
        )

    return valid_data, valid_structures


def log_training_step(
    step: int,
    loss_dict: Dict[str, float],
    reward_stats: Dict[str, float],
    prefix: str = "RL",
):
    """Log training step statistics.

    Args:
        step: Current step number
        loss_dict: Dictionary of loss values
        reward_stats: Dictionary of reward statistics
        prefix: Log prefix
    """
    log_parts = [f"[{prefix}] Step {step}"]

    # Add loss info
    if loss_dict:
        loss_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
        log_parts.append(f"loss: {loss_str}")

    # Add reward info
    if reward_stats:
        reward_str = ", ".join([f"{k}={v:.4f}" for k, v in reward_stats.items()])
        log_parts.append(f"reward: {reward_str}")

    ppmat_logger.info(" | ".join(log_parts))


def save_rl_model(
    model: nn.Layer,
    save_dir: str,
    filename: str = "rl_model.pdparams",
    epoch: Optional[int] = None,
    optimizer: Optional[paddle.optimizer.Optimizer] = None,
):
    """Save RL model checkpoint.

    Simplified wrapper for saving RL model checkpoints.
    Uses paddle.save directly for RL-specific checkpoint format.

    Args:
        model: Model to save
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        epoch: Optional epoch number
        optimizer: Optional optimizer state to save
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)

    checkpoint_dict = {
        "model": model.state_dict(),
    }

    if epoch is not None:
        checkpoint_dict["epoch"] = epoch

    if optimizer is not None:
        checkpoint_dict["optimizer"] = optimizer.state_dict()

    paddle.save(checkpoint_dict, checkpoint_path)
    ppmat_logger.info(f"Model saved to {checkpoint_path}")


def load_rl_model(
    model: nn.Layer,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[paddle.optimizer.Optimizer] = None,
) -> Dict:
    """Load RL model checkpoint.

    Simplified wrapper for loading RL model checkpoints.
    Uses paddle.load directly for RL-specific checkpoint format.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into (if load_optimizer=True)

    Returns:
        Dictionary with checkpoint information (epoch, etc.)
    """
    checkpoint = paddle.load(checkpoint_path)

    if "model" in checkpoint:
        model.set_state_dict(checkpoint["model"])
        ppmat_logger.info(f"Model loaded from {checkpoint_path}")

    result = {}
    if "epoch" in checkpoint:
        result["epoch"] = checkpoint["epoch"]

    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.set_state_dict(checkpoint["optimizer"])
        ppmat_logger.info("Optimizer state loaded")

    return result
