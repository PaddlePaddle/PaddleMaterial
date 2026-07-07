# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import paddle
from paddle import Tensor
import paddle.nn as nn

from ppmat.models.matterchat.chgnet.graph.crystalgraph import CrystalGraph
from ppmat.models.matterchat.chgnet.graph.graph import Graph, Node

if TYPE_CHECKING:
    from pymatgen.core import Structure

datatype = paddle.float32


class CrystalGraphConverter(nn.Layer):
    """Convert a pymatgen.core.Structure to a CrystalGraph."""

    def __init__(
        self, atom_graph_cutoff: float = 5, bond_graph_cutoff: float = 3
    ) -> None:
        """Initialize the Crystal Graph Converter."""
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        if bond_graph_cutoff is None:
            self.bond_graph_cutoff = atom_graph_cutoff
        else:
            self.bond_graph_cutoff = bond_graph_cutoff

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
    ) -> CrystalGraph:
        """Convert a structure, return a CrystalGraph."""
        n_atoms = len(structure)
        atomic_number = paddle.to_tensor(
            [i.specie.Z for i in structure], dtype="int64"
        )
        atomic_number.stop_gradient = True
        atom_frac_coord = paddle.to_tensor(
            structure.frac_coords, dtype=datatype
        )
        atom_frac_coord.stop_gradient = False
        lattice = paddle.to_tensor(
            structure.lattice.matrix, dtype=datatype
        )
        lattice.stop_gradient = False
        center_index, neighbor_index, image, distance = self.get_neighbors(structure)

        graph = Graph([Node(index=i) for i in range(n_atoms)])
        for ii, jj, img, dist in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)

        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = paddle.to_tensor(atom_graph, dtype="int64")
        directed2undirected = paddle.to_tensor(directed2undirected, dtype="int64")

        try:
            bond_graph, undirected2directed = graph.line_graph_adjacency_list(
                cutoff=self.bond_graph_cutoff
            )
        except Exception as exc:
            structure.to(filename="bond_graph_error.cif")
            raise SystemExit(
                f"Failed creating bond graph for {graph_id}, check bond_graph_error.cif"
            ) from exc
        bond_graph = paddle.to_tensor(bond_graph, dtype="int64")
        undirected2directed = paddle.to_tensor(undirected2directed, dtype="int64")

        has_isolated_atom = not set(range(n_atoms)).issubset(center_index)
        if has_isolated_atom:
            r_cutoff = self.atom_graph_cutoff
            msg = f"{graph_id=} has isolated atom with {r_cutoff=}, should be skipped"
            if on_isolated_atoms == "ignore":
                return None
            if on_isolated_atoms == "warn":
                print(msg, file=sys.stderr)
                return None
            raise ValueError(msg)

        return CrystalGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=paddle.to_tensor(image, dtype=datatype),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            bond_graph=bond_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            bond_graph_cutoff=self.bond_graph_cutoff,
        )

    def get_neighbors(
        self, structure: Structure
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get neighbor information from pymatgen utility function."""
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
        )
        return center_index, neighbor_index, image, distance

    def as_dict(self) -> dict[str, float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "bond_graph_cutoff": self.bond_graph_cutoff,
        }

    @classmethod
    def from_dict(cls, dict) -> CrystalGraphConverter:
        """Create converter from dictionary."""
        return CrystalGraphConverter(**dict)
