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

import os
from typing import Any

import paddle
from paddle import Tensor

datatype = paddle.float32


class CrystalGraph:
    """A data class for crystal graph."""

    def __init__(
        self,
        atomic_number: Tensor,
        atom_frac_coord: Tensor,
        atom_graph: Tensor,
        atom_graph_cutoff: float,
        neighbor_image: Tensor,
        directed2undirected: Tensor,
        undirected2directed: Tensor,
        bond_graph: Tensor,
        bond_graph_cutoff: float,
        lattice: Tensor,
        graph_id: str | None = None,
        mp_id: str | None = None,
        composition: str | None = None,
    ) -> None:
        """Initialize the crystal graph."""
        super().__init__()
        self.atomic_number = atomic_number
        self.atom_frac_coord = atom_frac_coord
        self.atom_graph = atom_graph
        self.atom_graph_cutoff = atom_graph_cutoff
        self.neighbor_image = neighbor_image
        self.directed2undirected = directed2undirected
        self.undirected2directed = undirected2directed
        self.bond_graph = bond_graph
        self.bond_graph_cutoff = bond_graph_cutoff
        self.lattice = lattice
        self.graph_id = graph_id
        self.mp_id = mp_id
        self.composition = composition
        if len(directed2undirected) != 2 * len(undirected2directed):
            raise ValueError(
                f"{graph_id} number of directed indices != 2 * number of undirected indices!"
            )

    def to(self, device: str = "cpu") -> CrystalGraph:
        """Move the graph to a device. Default = 'cpu'."""
        return CrystalGraph(
            atomic_number=self.atomic_number,
            atom_frac_coord=self.atom_frac_coord,
            atom_graph=self.atom_graph,
            atom_graph_cutoff=self.atom_graph_cutoff,
            neighbor_image=self.neighbor_image,
            directed2undirected=self.directed2undirected,
            undirected2directed=self.undirected2directed,
            bond_graph=self.bond_graph,
            bond_graph_cutoff=self.bond_graph_cutoff,
            lattice=self.lattice,
            graph_id=self.graph_id,
            mp_id=self.mp_id,
            composition=self.composition,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the graph to a dictionary."""
        return {
            "atomic_number": self.atomic_number,
            "atom_frac_coord": self.atom_frac_coord,
            "atom_graph": self.atom_graph,
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "neighbor_image": self.neighbor_image,
            "directed2undirected": self.directed2undirected,
            "undirected2directed": self.undirected2directed,
            "bond_graph": self.bond_graph,
            "bond_graph_cutoff": self.bond_graph_cutoff,
            "lattice": self.lattice,
            "graph_id": self.graph_id,
            "mp_id": self.mp_id,
            "composition": self.composition,
        }

    def save(self, fname: str | None = None, save_dir: str = ".") -> str:
        """Save the graph to a file."""
        if fname is not None:
            save_name = os.path.join(save_dir, fname)
        elif self.graph_id is not None:
            save_name = os.path.join(save_dir, f"{self.graph_id}.pdparams")
        else:
            save_name = os.path.join(save_dir, f"{self.composition}.pdparams")
        paddle.save(self.to_dict(), path=save_name)
        return save_name

    @classmethod
    def from_file(cls, file_name: str) -> CrystalGraph:
        """Load a crystal graph from a file."""
        return paddle.load(file_name)

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> CrystalGraph:
        """Load a CrystalGraph from a dictionary."""
        return CrystalGraph(**dic)

    def __repr__(self) -> str:
        """Details of the graph."""
        return (
            f"Crystal Graph {self.composition} \n"
            f"constructed using atom_graph_cutoff={self.atom_graph_cutoff}, "
            f"bond_graph_cutoff={self.bond_graph_cutoff} \n"
            f"(n_atoms={len(self.atomic_number)}, "
            f"atom_graph={len(self.atom_graph)}, "
            f"bond_graph={len(self.bond_graph)})"
        )
