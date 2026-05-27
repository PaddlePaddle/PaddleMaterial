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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle

from ppmat.datasets.custom_data_type import ConcatData


def _extract_concat_data(data):
    if data is None:
        return None
    if isinstance(data, ConcatData):
        return np.array(data.data)
    return data


class CrystalBatch:
    """Minimal batch object mimicking torch_geometric Batch interface."""

    def __init__(
        self,
        num_atoms: paddle.Tensor,
        atom_types: paddle.Tensor,
        batch: paddle.Tensor,
        frac_coords: Optional[paddle.Tensor] = None,
        lattice: Optional[paddle.Tensor] = None,
    ):
        self.num_atoms = num_atoms
        self.atom_types = atom_types
        self.batch = batch
        self.frac_coords = frac_coords
        self.lattice = lattice

    def cuda(self, device: str = "gpu"):
        place = paddle.CUDAPlace(0) if device == "gpu" else paddle.CPUPlace()
        self.num_atoms = self.num_atoms.to(place)
        self.atom_types = self.atom_types.to(place)
        self.batch = self.batch.to(place)
        if self.frac_coords is not None:
            self.frac_coords = self.frac_coords.to(place)
        if self.lattice is not None:
            self.lattice = self.lattice.to(place)
        return self


class MiADCollator:
    """Collate function for MiAD model.

    Converts MP20Dataset batch to CrystalGen format.
    """

    def __init__(
        self,
        mirage_max_atoms: Optional[int] = None,
        use_mirage: bool = False,
    ):
        self.mirage_max_atoms = mirage_max_atoms
        self.use_mirage = use_mirage
        self.N_m = mirage_max_atoms

    def __call__(self, batch_list: List[Dict]) -> Dict[str, Any]:
        if not batch_list:
            return {}
        batch_size = len(batch_list)
        all_frac_coords = []
        all_atom_types = []
        all_num_atoms = []
        all_lattices = []
        batch_idx_list = []

        for i, sample_i in enumerate(batch_list):
            if not isinstance(sample_i, dict):
                continue
            if "structure_array" in sample_i:
                sa = sample_i["structure_array"]
            elif "frac_coords" in sample_i:
                sa = sample_i
            else:
                continue

            frac_coords = _extract_concat_data(sa.get("frac_coords"))
            atom_types = _extract_concat_data(sa.get("atom_types"))
            lattice = _extract_concat_data(sa.get("lattice"))
            num_atoms = _extract_concat_data(sa.get("num_atoms"))

            if frac_coords is None or atom_types is None or lattice is None:
                continue

            if frac_coords.ndim >= 2:
                n_atoms = frac_coords.shape[0]
            elif num_atoms is not None:
                if isinstance(num_atoms, np.ndarray) and num_atoms.ndim > 0:
                    n_atoms = int(num_atoms.flatten()[0])
                else:
                    n_atoms = int(num_atoms)
            else:
                continue

            if frac_coords.ndim == 1:
                frac_coords = frac_coords.reshape(-1, 3)
            if atom_types.ndim > 1:
                atom_types = atom_types.flatten()

            if lattice.ndim == 2:
                lattice = lattice.reshape(1, 3, 3)
            elif lattice.ndim == 1:
                lattice = lattice.reshape(1, 3, 3)

            if self.N_m is not None and self.use_mirage and n_atoms < self.N_m:
                frac_coords, atom_types = self._pad_mirage(
                    frac_coords, atom_types, n_atoms, self.N_m
                )
                n_atoms = self.N_m

            all_frac_coords.append(frac_coords)
            all_atom_types.append(atom_types)
            all_lattices.append(lattice[0] if lattice.shape[0] == 1 else lattice)
            all_num_atoms.append(n_atoms)
            batch_idx_list.extend([i] * n_atoms)

        if not all_frac_coords:
            return {}

        frac_coords_cat = np.concatenate(all_frac_coords, axis=0).astype("float32")
        atom_types_cat = np.concatenate(all_atom_types, axis=0).astype("int64")
        lattices_cat = np.stack(all_lattices, axis=0).astype("float32")
        num_atoms_arr = np.array(all_num_atoms, dtype="int64")
        batch_idx = np.array(batch_idx_list, dtype="int64")

        frac_coords_t = paddle.to_tensor(frac_coords_cat)
        atom_types_t = paddle.to_tensor(atom_types_cat)
        lattices_t = paddle.to_tensor(lattices_cat)
        num_atoms_t = paddle.to_tensor(num_atoms_arr)
        batch_idx_t = paddle.to_tensor(batch_idx)

        crystal_batch = CrystalBatch(
            num_atoms=num_atoms_t,
            atom_types=atom_types_t,
            batch=batch_idx_t,
            frac_coords=frac_coords_t,
            lattice=lattices_t,
        )

        return {
            "x0": [lattices_t, frac_coords_t, atom_types_t],
            "batch_size": batch_size,
            "num_atoms": num_atoms_t,
            "batch_idx": batch_idx_t,
            "atom_types": atom_types_t,
            "batch": crystal_batch,
        }

    @staticmethod
    def _pad_mirage(frac_coords, atom_types, n_atoms, N_m):
        pad_n = N_m - n_atoms
        if pad_n > 0:
            pad_fc = np.random.rand(pad_n, 3).astype("float32")
            pad_at = np.zeros(pad_n, dtype="int64")
            frac_coords = np.concatenate([frac_coords, pad_fc], axis=0)
            atom_types = np.concatenate([atom_types, pad_at], axis=0)
        return frac_coords, atom_types


def create_miad_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    mirage_max_atoms: Optional[int] = None,
    use_mirage: bool = False,
    **kwargs,
) -> paddle.io.DataLoader:
    collate_fn = MiADCollator(
        mirage_max_atoms=mirage_max_atoms,
        use_mirage=use_mirage,
    )
    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )


def create_sampling_batch(
    batch_size: int,
    num_atoms: Optional[paddle.Tensor] = None,
    mirage_max_atoms: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    if mirage_max_atoms is not None:
        N_m = mirage_max_atoms
        num_atoms = paddle.full([batch_size], N_m, dtype="int64")
        total_atoms = batch_size * N_m
    elif num_atoms is not None:
        total_atoms = int(num_atoms.sum())
    else:
        num_atoms = paddle.randint(5, 30, [batch_size], dtype="int64")
        total_atoms = int(num_atoms.sum())

    node2graph = paddle.repeat_interleave(
        paddle.arange(batch_size, dtype="int64"), num_atoms
    )

    crystal_batch = CrystalBatch(
        num_atoms=num_atoms,
        atom_types=paddle.zeros([total_atoms], dtype="int64"),
        batch=node2graph,
        frac_coords=paddle.zeros([total_atoms, 3], dtype="float32"),
        lattice=paddle.zeros([batch_size, 3, 3], dtype="float32"),
    )

    return {
        "x0": [None, None, None],
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "batch": crystal_batch,
    }
