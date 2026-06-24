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


def _process_result_dict(r: dict):
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
        pmg_struct = _structure_array_to_pymatgen(
            frac_coords=frac, atom_types=at, lattice=lat[np.newaxis], num_atoms=[na]
        )[0]
    except Exception:
        pmg_struct = None
    return structure_data, pmg_struct


def _structure_array_to_pymatgen(frac_coords, atom_types, lattice, num_atoms):
    structures = []
    start_idx = 0
    for i in range(len(num_atoms)):
        n_atoms = num_atoms[i]
        end_idx = start_idx + n_atoms
        structure = Structure(
            lattice=Lattice(lattice[i]),
            species=atom_types[start_idx:end_idx].astype(int),
            coords=frac_coords[start_idx:end_idx],
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

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        all_data = []
        all_structures = []
        total_samples = self.num_batches * self.batch_size

        for sample_idx in range(total_samples):
            num_atoms = self._resolve_num_atoms(kwargs, sample_idx, min_atoms=2)
            if int(num_atoms.shape[0]) == 1:
                num_atoms = paddle.concat([num_atoms, num_atoms], axis=0)
            batch_data = {"structure_array": {"num_atoms": num_atoms}}

            try:
                with paddle.no_grad():
                    output = model.sample(batch_data, num_inference_steps=self.num_inference_steps)
            except Exception:
                continue

            for r in output["result"]:
                sd, s = _process_result_dict(r)
                all_data.append(sd)
                all_structures.append(s)

        return all_data, all_structures

    @staticmethod
    def _resolve_num_atoms(kwargs, idx, min_atoms=1):
        if "num_atoms" not in kwargs:
            return paddle.randint(low=min_atoms, high=50, shape=[1])
        na = kwargs["num_atoms"]
        if isinstance(na, int):
            na = [max(min_atoms, na)]
        elif isinstance(na, (list, tuple, np.ndarray)):
            na = [max(min_atoms, int(na[idx % len(na)]))]
        else:
            na = [max(min_atoms, int(na))]
        return paddle.to_tensor(na, dtype="int64")


class DiffCSPSampler(BaseSampler):
    """Sampler for DiffCSP models."""

    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        all_data = []
        all_structures = []

        for _ in range(self.num_batches):
            if "num_atoms" in kwargs:
                na = kwargs["num_atoms"]
                num_atoms = paddle.to_tensor(
                    [na] * self.batch_size if isinstance(na, int) else na,
                    dtype="int64",
                )
            else:
                num_atoms = paddle.randint(low=1, high=50, shape=[self.batch_size])

            total_atoms = int(num_atoms.sum().item())
            atom_types = paddle.randint(low=1, high=101, shape=[total_atoms], dtype="int64")
            batch_data = {"structure_array": {"num_atoms": num_atoms, "atom_types": atom_types}}

            with paddle.no_grad():
                output = model.sample(batch_data, num_inference_steps=self.num_inference_steps)

            for r in output.get("result", []):
                sd, s = _process_result_dict(r)
                all_data.append(sd)
                all_structures.append(s)

        return all_data, all_structures
