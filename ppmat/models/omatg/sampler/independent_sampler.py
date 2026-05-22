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

"""Independent Sampler implementation.
"""

import numpy as np
import paddle

from .abstracts import PositionDistribution, CellDistribution, SpeciesDistribution
from ppmat.datasets.omatg.structure import Structure
from ppmat.models.omatg.datamodule.omatg_data import OMATGData


class IndependentSampler:
    """Sample from product of independent distributions for cell, positions, species."""

    def __init__(
        self,
        position_distribution: PositionDistribution | None = None,
        cell_distribution: CellDistribution | None = None,
        species_distribution: SpeciesDistribution | None = None,
    ) -> None:
        super().__init__()

        self._position_distribution = position_distribution
        self._cell_distribution = cell_distribution
        self._species_distribution = species_distribution

        if (
            position_distribution is None
            and cell_distribution is None
            and species_distribution is None
        ):
            raise ValueError("At least one distribution must be provided.")

    def sample(
        self,
        pos: paddle.Tensor | None = None,
        pos_is_fractional: bool | None = None,
        cell: paddle.Tensor | None = None,
        species: paddle.Tensor | None = None,
    ) -> tuple[np.ndarray | None, bool | None, np.ndarray | None, np.ndarray | None,]:
        """Sample from each base distribution if provided."""
        sampled_pos = None
        sampled_pos_is_fractional = None
        sampled_cell = None
        sampled_species = None

        if self._position_distribution is not None:
            if pos is None:
                raise ValueError(
                    "Position distribution provided but no positions given."
                )
            if pos_is_fractional is None:
                raise ValueError(
                    "Position distribution provided but no coordinate system specified."
                )
            sampled_pos, sampled_pos_is_fractional = self._position_distribution(
                pos, pos_is_fractional
            )

        if self._cell_distribution is not None:
            if cell is None:
                raise ValueError("Cell distribution provided but no cell given.")
            sampled_cell = self._cell_distribution(cell)

        if self._species_distribution is not None:
            if species is None:
                raise ValueError("Species distribution provided but no species given.")
            sampled_species = self._species_distribution(species)

        return (
            sampled_pos,
            sampled_pos_is_fractional,
            sampled_cell,
            sampled_species,
        )

    def sample_p_0(self, x_1: OMATGData) -> OMATGData:
        """Sample base distribution given a batch x_1, returns batched OMATGData x_0."""
        structures = []
        for i in range(len(x_1.n_atoms)):
            sl = slice(int(x_1.ptr[i]), int(x_1.ptr[i + 1]))
            species = self._species_distribution(x_1.species[sl])
            pos, pos_is_fractional = self._position_distribution(
                x_1.pos[sl], bool(x_1.pos_is_fractional[i])
            )
            cell = self._cell_distribution(x_1.cell[i])
            sampled = Structure(
                cell=paddle.to_tensor(cell, dtype=x_1.cell.dtype),
                atomic_numbers=paddle.to_tensor(species, dtype="int64"),
                pos=paddle.to_tensor(pos, dtype=x_1.pos.dtype),
                pos_is_fractional=pos_is_fractional,
            )
            if bool(x_1.pos_is_fractional[i]):
                sampled.convert_to_fractional()
            else:
                sampled.convert_to_cartesian()
            structures.append(sampled)
        return OMATGData.from_batch(structures, concatenate=True)
