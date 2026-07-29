# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert crystal structures to atomistic graphs for MACE."""

from __future__ import annotations

from typing import Optional

from pymatgen.core.structure import Structure

from ppmat.models.common.graph_converter import FindPointsInSpheres


class MACEGraphConverter:
    """Build MACE input graphs via cutoff-radius neighbor search.

    Wraps the suite ``FindPointsInSpheres`` converter so that graph fields
    match Trainer / Predictor conventions (``atom_types``, ``cart_coords``,
    ``bond_dist``, ``lattice``, etc.).

    Args:
        cutoff (float): Neighbor cutoff radius in Angstrom. Default: 6.0
            (same as ``mace_mp0_medium``).
        num_elements (int): Number of element types. Default: 89
            (MACE-MP-0 coverage).
        pbc (tuple): Periodic boundary flags. Default: all enabled.
        num_cpus (Optional[int]): CPUs for parallel graph building;
            ``None`` means automatic.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        num_elements: int = 89,
        pbc: tuple = (1, 1, 1),
        num_cpus: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
        self.num_elements = num_elements
        self.pbc = pbc
        self.num_cpus = num_cpus
        self._converter = FindPointsInSpheres(
            cutoff=cutoff,
            pbc=pbc,
            num_cpus=num_cpus,
        )

    def __call__(self, structure: Structure):
        """Convert one or more ``Structure`` objects to PGL graphs."""
        return self._converter(structure)
