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

from __future__ import annotations

import glob
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import ase.io
import numpy as np
import paddle
from ase.neighborlist import primitive_neighbor_list
from paddle.io import Dataset

from ppmat.datasets.geometric_data_type.data import Data


def _expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(paths, str):
        paths = [paths]

    expanded = []
    for path in paths:
        matches = sorted(glob.glob(path))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(path)
    return expanded


def _as_pbc_tuple(pbc) -> tuple[bool, bool, bool]:
    arr = np.asarray(pbc, dtype=bool).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    return tuple(bool(x) for x in arr[:3])


def ase_atoms_to_dm2_data(
    atoms,
    species_to_index: Dict[int, int],
    cutoff: float,
    cooling_rate: Optional[float] = None,
) -> Data:
    """Convert an ASE ``Atoms`` object to PaddleMaterials geometric ``Data``."""

    atomic_numbers = np.asarray(atoms.numbers, dtype=np.int64)
    x = np.asarray([species_to_index[int(z)] for z in atomic_numbers], dtype=np.int64)
    pbc = _as_pbc_tuple(atoms.pbc)
    cell = np.asarray(atoms.cell.array, dtype=np.float32)
    positions = np.asarray(atoms.positions, dtype=np.float32)

    src, dst, disp = primitive_neighbor_list(
        "ijD",
        cutoff=cutoff,
        pbc=pbc,
        cell=cell,
        positions=positions,
        numbers=atomic_numbers,
    )
    data = Data(
        x=paddle.to_tensor(x, dtype="int64"),
        pos=paddle.to_tensor(positions, dtype="float32"),
        edge_index=paddle.to_tensor(np.stack([src, dst]), dtype="int64"),
        edge_attr=paddle.to_tensor(disp, dtype="float32"),
        atomic_numbers=paddle.to_tensor(atomic_numbers, dtype="int64"),
        lattice=paddle.to_tensor(cell[None, :, :], dtype="float32"),
        pbc=paddle.to_tensor(np.asarray(pbc, dtype=bool)[None, :], dtype="bool"),
        num_nodes=len(atomic_numbers),
    )
    if cooling_rate is not None:
        data.cooling_rate = paddle.to_tensor([cooling_rate], dtype="float32")
    return data


class DM2StructureDataset(Dataset):
    """ASE-backed dataset for DM2 disordered-material denoising.

    Args:
        paths (str|Sequence[str]): Structure files or glob patterns. ASE is used for
            reading, so formats such as ``lammps-data``, ``extxyz`` and ``cif`` are
            supported through ``file_format``.
        file_format (Optional[str]): ASE format string. Leave ``None`` to let ASE
            infer the format.
        cutoff (float): Large neighbor cutoff used before rattle/downselect.
        species (Optional[Sequence[int]]): Ordered atomic numbers for species
            encoding. If omitted, the dataset infers a sorted species list.
        duplicate (int): Repeat each loaded structure this many times per epoch.
        cooling_rates (Optional[Sequence[float]]): Per-structure conditioning values.
            DM2 conditional training conventionally uses ``log10(cooling_rate)``;
            set ``log10_cooling_rate=True`` to apply that transform here.
        log10_cooling_rate (bool): Whether to store log10-transformed cooling rates.
    """

    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        file_format: Optional[str] = None,
        cutoff: float = 10.0,
        species: Optional[Sequence[int]] = None,
        duplicate: int = 128,
        cooling_rates: Optional[Sequence[float]] = None,
        log10_cooling_rate: bool = True,
    ):
        super().__init__()
        self.paths = _expand_paths(paths)
        if len(self.paths) == 0:
            raise ValueError("DM2StructureDataset received no structure files.")
        self.file_format = file_format
        self.cutoff = cutoff
        self.duplicate = int(duplicate)
        if self.duplicate <= 0:
            raise ValueError("duplicate must be a positive integer.")

        atoms_list = [
            ase.io.read(path, format=file_format)
            for path in self.paths
        ]

        if species is None:
            unique_species = sorted(
                {
                    int(number)
                    for atoms in atoms_list
                    for number in np.asarray(atoms.numbers).tolist()
                }
            )
        else:
            unique_species = [int(number) for number in species]
        self.species = unique_species
        self.species_to_index = {z: idx for idx, z in enumerate(unique_species)}

        if cooling_rates is not None and len(cooling_rates) != len(atoms_list):
            raise ValueError(
                "cooling_rates must be omitted or have the same length as paths."
            )

        self.graphs = []
        for idx, atoms in enumerate(atoms_list):
            cooling_rate = None
            if cooling_rates is not None:
                cooling_rate = float(cooling_rates[idx])
                if log10_cooling_rate:
                    cooling_rate = float(np.log10(cooling_rate))
            self.graphs.append(
                ase_atoms_to_dm2_data(
                    atoms=atoms,
                    species_to_index=self.species_to_index,
                    cutoff=cutoff,
                    cooling_rate=cooling_rate,
                )
            )

    def __len__(self):
        return len(self.graphs) * self.duplicate

    def __getitem__(self, idx):
        graph = self.graphs[idx % len(self.graphs)]
        return graph.clone()
