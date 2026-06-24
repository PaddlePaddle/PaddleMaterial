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
from typing import List

import numpy as np
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
        if max(struc.lattice.abc) > max_lattice_param:
            return False
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

    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()

    with open(out_path, "w"):
        for struc in structures:
            atoms = adaptor.get_atoms(struc)
            write(out_path, atoms, append=True)

    ppmat_logger.info(f"Saved {len(structures)} structures to {out_path}")
    return out_path

