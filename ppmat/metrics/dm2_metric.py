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
from typing import Sequence
from typing import Union

import ase.io
import numpy as np
from ase import Atoms


def _expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(paths, str):
        paths = [paths]

    expanded = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No DM2 metric reference files matched: {pattern}")
        expanded.extend(matches)
    return expanded


def _structure_array_to_atoms(structure_array: Dict) -> Atoms:
    frac_coords = np.asarray(structure_array["frac_coords"], dtype=np.float64)
    atom_types = np.asarray(structure_array["atom_types"], dtype=np.int64)
    lattice = np.asarray(structure_array["lattice"], dtype=np.float64).reshape(3, 3)
    return Atoms(
        numbers=atom_types,
        scaled_positions=frac_coords,
        cell=lattice,
        pbc=True,
    )


def _rdf_histogram(atoms_list: Sequence[Atoms], cutoff: float, bins: int):
    hist = np.zeros(bins, dtype=np.float64)
    for atoms in atoms_list:
        distances = atoms.get_all_distances(mic=True)
        upper = distances[np.triu_indices_from(distances, k=1)]
        upper = upper[(upper > 1e-8) & (upper <= cutoff)]
        values, _ = np.histogram(upper, bins=bins, range=(0.0, cutoff))
        hist += values.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _wasserstein_1d_from_hist(hist_a, hist_b, cutoff: float):
    bin_width = cutoff / len(hist_a)
    return float(np.abs(np.cumsum(hist_a) - np.cumsum(hist_b)).sum() * bin_width)


def _mean_coordination(
    atoms_list: Sequence[Atoms],
    center_atomic_number: int,
    neighbor_atomic_number: int,
    cutoff: float,
):
    values = []
    for atoms in atoms_list:
        numbers = np.asarray(atoms.numbers, dtype=np.int64)
        center_mask = numbers == int(center_atomic_number)
        neighbor_mask = numbers == int(neighbor_atomic_number)
        if not np.any(center_mask):
            continue
        distances = atoms.get_all_distances(mic=True)
        valid = (
            (distances <= float(cutoff))
            & (distances > 1e-8)
            & neighbor_mask[None, :]
        )
        values.extend(valid[center_mask].sum(axis=1).tolist())
    if not values:
        return float("nan")
    return float(np.mean(values))


class DM2AmorphousGenerationMetric:
    """RDF and coordination metrics for DM2 amorphous structure sampling."""

    def __init__(
        self,
        reference_paths: Union[str, Sequence[str]],
        file_format: str = "lammps-data",
        rdf_cutoff: float = 8.0,
        rdf_bins: int = 200,
        coordination_cutoffs: Sequence[Dict] = (),
    ):
        self.reference_paths = _expand_paths(reference_paths)
        self.file_format = file_format
        self.rdf_cutoff = float(rdf_cutoff)
        self.rdf_bins = int(rdf_bins)
        self.coordination_cutoffs = list(coordination_cutoffs)
        self.reference_atoms = [
            ase.io.read(path, format=file_format) for path in self.reference_paths
        ]
        self.reference_rdf = _rdf_histogram(
            self.reference_atoms,
            cutoff=self.rdf_cutoff,
            bins=self.rdf_bins,
        )

    def __call__(self, pred_structures: Sequence[Dict]):
        pred_atoms = [_structure_array_to_atoms(item) for item in pred_structures]
        pred_rdf = _rdf_histogram(
            pred_atoms,
            cutoff=self.rdf_cutoff,
            bins=self.rdf_bins,
        )

        metric = {
            "rdf_wasserstein": _wasserstein_1d_from_hist(
                pred_rdf,
                self.reference_rdf,
                cutoff=self.rdf_cutoff,
            )
        }

        for item in self.coordination_cutoffs:
            name = item.get("name", "coordination")
            pred_value = _mean_coordination(
                pred_atoms,
                center_atomic_number=item["center_atomic_number"],
                neighbor_atomic_number=item["neighbor_atomic_number"],
                cutoff=item["cutoff"],
            )
            ref_value = _mean_coordination(
                self.reference_atoms,
                center_atomic_number=item["center_atomic_number"],
                neighbor_atomic_number=item["neighbor_atomic_number"],
                cutoff=item["cutoff"],
            )
            metric[name] = pred_value
            metric[f"{name}_reference"] = ref_value
            metric[f"{name}_abs_diff"] = abs(pred_value - ref_value)

        return metric
