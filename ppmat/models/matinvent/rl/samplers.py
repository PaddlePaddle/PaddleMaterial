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
Samplers for RL training.



Provides sampler classes for generating crystal structures during RL training.
"""

from typing import List
from typing import Tuple

import numpy as np
import paddle
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure


def structure_array_to_pymatgen_structures(
    frac_coords, atom_types, lattice, num_atoms
) -> List[Structure]:
    """Convert structure array components to pymatgen Structure objects.

    Args:
        frac_coords: Fractional coordinates array (N_atoms, 3)
        atom_types: Atomic numbers array (N_atoms,)
        lattice: Lattice matrix array (N_crystals, 3, 3)
        num_atoms: Number of atoms per crystal (N_crystals,)

    Returns:
        List of pymatgen Structure objects
    """
    structures = []
    start_idx = 0
    batch_size = len(num_atoms)

    for i in range(batch_size):
        n_atoms = num_atoms[i]
        end_idx = start_idx + n_atoms

        # Extract data for this crystal
        crystal_frac_coords = frac_coords[start_idx:end_idx]
        crystal_atom_types = atom_types[start_idx:end_idx].astype(int)
        crystal_lattice = lattice[i]

        # Create pymatgen Structure
        structure = Structure(
            lattice=Lattice(crystal_lattice),
            species=crystal_atom_types,
            coords=crystal_frac_coords,
            coords_are_cartesian=False,
        )

        structures.append(structure)
        start_idx = end_idx

    return structures


class BaseSampler:
    """Base class for RL samplers."""

    def __init__(
        self,
        batch_size: int = 16,
        num_batches: int = 4,
        num_inference_steps: int = 1000,
    ):
        """Initialize sampler.

        Args:
            batch_size: Number of samples per batch
            num_batches: Number of batches to generate
            num_inference_steps: Number of denoising steps
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps

    def generate(self, model, **kwargs):
        """Generate samples using the model.

        Args:
            model: Diffusion model (MatterGen or DiffCSP)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (data_list, structure_list)
        """
        raise NotImplementedError


class MatterGenSampler(BaseSampler):
    """Sampler for MatterGen models."""

    def __init__(
        self,
        batch_size: int = 16,
        num_batches: int = 4,
        num_inference_steps: int = 1000,
    ):
        super().__init__(batch_size, num_batches, num_inference_steps)

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        """Generate crystal structures using MatterGen model.

        Args:
            model: MatterGen model
            **kwargs: Additional parameters (num_atoms, conditions, etc.)

        Returns:
            Tuple of (data_list, structure_list)
        """
        all_data = []
        all_structures = []

        total_samples = self.num_batches * self.batch_size
        for sample_idx in range(total_samples):
            # MatterGen's graph builder can fail when different samples carry
            # inconsistent PBC metadata in one batch, so we sample one by one.
            if "num_atoms" in kwargs:
                num_atoms = kwargs["num_atoms"]
                if isinstance(num_atoms, int):
                    num_atoms = [max(2, int(num_atoms))]
                elif isinstance(num_atoms, (list, tuple, np.ndarray)):
                    num_atoms = [max(2, int(num_atoms[sample_idx % len(num_atoms)]))]
                else:
                    num_atoms = [max(2, int(num_atoms))]
                num_atoms = paddle.to_tensor(num_atoms, dtype="int64")
            else:
                num_atoms = paddle.randint(low=2, high=50, shape=[1])

            # Keep at least two structures in the batch to avoid single-item
            # PBC reduction instability in MatterGen internals.
            if int(num_atoms.shape[0]) == 1:
                num_atoms = paddle.concat([num_atoms, num_atoms], axis=0)

            batch_data = {"structure_array": {"num_atoms": num_atoms}}

            try:
                with paddle.no_grad():
                    output = model.sample(
                        batch_data, num_inference_steps=self.num_inference_steps
                    )
            except Exception:
                continue

            # MatterGen.sample() returns
            # {"result": [{num_atoms, atom_types, ...}, ...]}
            results = output["result"]
            for r in results:
                na = r["num_atoms"]
                frac = np.array(r["frac_coords"], dtype=np.float32)
                at = np.array(r["atom_types"], dtype=np.int32)
                lat = np.array(r["lattice"], dtype=np.float32)

                structure_data = {
                    "structure_array": {
                        "num_atoms": paddle.to_tensor([na], dtype="int64"),
                        "frac_coords": paddle.to_tensor(frac),
                        "atom_types": paddle.to_tensor(at),
                        "lattice": paddle.to_tensor(lat).unsqueeze(0),
                    }
                }
                try:
                    pmg_struct = structure_array_to_pymatgen_structures(
                        frac_coords=frac,
                        atom_types=at,
                        lattice=lat[np.newaxis],
                        num_atoms=[na],
                    )[0]
                except Exception:
                    pmg_struct = None

                all_data.append(structure_data)
                all_structures.append(pmg_struct)

        return all_data, all_structures


class DiffCSPSampler(BaseSampler):
    """Sampler for DiffCSP models."""

    def __init__(
        self,
        batch_size: int = 16,
        num_batches: int = 4,
        num_inference_steps: int = 1000,
    ):
        super().__init__(batch_size, num_batches, num_inference_steps)

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        """Generate crystal structures using DiffCSP model.

        Args:
            model: DiffCSP model
            **kwargs: Additional parameters

        Returns:
            Tuple of (data_list, structure_list)
        """
        all_data = []
        all_structures = []

        for batch_idx in range(self.num_batches):
            # Prepare batch data
            if "num_atoms" in kwargs:
                num_atoms = kwargs["num_atoms"]
                if isinstance(num_atoms, int):
                    num_atoms = [num_atoms] * self.batch_size
                num_atoms = paddle.to_tensor(num_atoms, dtype="int64")
            else:
                num_atoms = paddle.randint(low=1, high=50, shape=[self.batch_size])

            total_atoms = int(num_atoms.sum().item())
            # DiffCSP.sample() 需要 atom_types (1-100)，随机初始化
            atom_types = paddle.randint(
                low=1, high=101, shape=[total_atoms], dtype="int64"
            )

            batch_data = {
                "structure_array": {
                    "num_atoms": num_atoms,
                    "atom_types": atom_types,
                }
            }

            # Generate samples
            with paddle.no_grad():
                output = model.sample(
                    batch_data,
                    num_inference_steps=self.num_inference_steps,
                )

            # DiffCSP.sample() 返回
            # {"result": [{num_atoms, atom_types, ...}, ...]}
            results = output.get("result", [])
            for r in results:
                na = r["num_atoms"]
                frac = np.array(r["frac_coords"], dtype=np.float32)
                at = np.array(r["atom_types"], dtype=np.int32)
                lat = np.array(r["lattice"], dtype=np.float32)

                structure_data = {
                    "structure_array": {
                        "num_atoms": paddle.to_tensor([na], dtype="int64"),
                        "frac_coords": paddle.to_tensor(frac),
                        "atom_types": paddle.to_tensor(at),
                        "lattice": paddle.to_tensor(lat).unsqueeze(0),
                    }
                }
                try:
                    pmg_struct = structure_array_to_pymatgen_structures(
                        frac_coords=frac,
                        atom_types=at,
                        lattice=lat[np.newaxis],
                        num_atoms=[na],
                    )[0]
                except Exception:
                    pmg_struct = None

                all_data.append(structure_data)
                all_structures.append(pmg_struct)

        return all_data, all_structures
