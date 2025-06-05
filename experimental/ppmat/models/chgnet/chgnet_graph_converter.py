# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from https://github.com/CederGroupHub/chgnet

from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import pgl
from p_tqdm import p_map
from pymatgen.core.structure import Structure


class Node:
    """A node in a graph."""

    def __init__(self, index: int, info: dict | None = None) -> None:
        """Initialize a Node.

        Args:
            index (int): the index of this node
            info (dict, optional): any additional information about this node.
        """
        self.index = index
        self.info = info
        self.neighbors: dict[int, list[DirectedEdge | UndirectedEdge]] = {}

    def add_neighbor(self, index, edge) -> None:
        """Draw an directed edge between self and the node specified by index.

        Args:
            index (int): the index of neighboring node
            edge (DirectedEdge): an DirectedEdge object pointing from self to the node.
        """
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class Edge(ABC):
    """Abstract base class for edges in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        """Initialize an Edge."""
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self) -> str:
        """String representation of this edge."""
        nodes, index, info = self.nodes, self.index, self.info
        return f"{type(self).__name__}(nodes={nodes!r}, index={index!r}, info={info!r})"

    def __hash__(self) -> int:
        """Hash this edge."""
        img = (self.info or {}).get("image")
        img_str = "" if img is None else img.tobytes()
        return hash((self.nodes[0], self.nodes[1], img_str))

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check if two edges are equal."""
        raise NotImplementedError


class UndirectedEdge(Edge):
    """An undirected/bi-directed edge in a graph."""

    __hash__ = Edge.__hash__

    def __eq__(self, other: object) -> bool:
        """Check if two undirected edges are equal."""
        return set(self.nodes) == set(other.nodes) and self.info == other.info


class DirectedEdge(Edge):
    """A directed edge in a graph."""

    __hash__ = Edge.__hash__

    def make_undirected(self, index: int, info: dict | None = None) -> UndirectedEdge:
        """Make a directed edge undirected."""
        info = info or {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def __eq__(self, other: object) -> bool:
        """Check if the two directed edges are equal.

        Args:
            other (DirectedEdge): another DirectedEdge to compare to

        Returns:
            bool: True if other is the same directed edge, or if other is the directed
                edge with reverse direction of self, else False.
        """
        if not isinstance(other, DirectedEdge):
            return False
        self_img = (self.info or {}).get("image")
        other_img = (other.info or {}).get("image")
        none_img = self_img is other_img is None
        if self.nodes == other.nodes and (none_img or all(self_img == other_img)):
            print(
                (
                    "the two directed edges are equal but this operation "
                    "is not supposed to happen"
                ),
                file=sys.stderr,
            )
            return True
        return self.nodes == other.nodes[::-1] and (
            none_img or all(self_img == -1 * other_img)
        )


class GraphUtils:
    """A graph for storing the neighbor information of atoms."""

    def __init__(self, nodes: list[Node]) -> None:
        """Initialize a Graph from a list of nodes."""
        self.nodes = nodes
        self.directed_edges: dict[frozenset[int], list[DirectedEdge]] = {}
        self.directed_edges_list: list[DirectedEdge] = []
        self.undirected_edges: dict[frozenset[int], list[UndirectedEdge]] = {}
        self.undirected_edges_list: list[UndirectedEdge] = []

    def add_edge(
        self, center_index, neighbor_index, image, distance, dist_tol: float = 1e-06
    ) -> None:
        """Add an directed edge to the graph.

        Args:
            center_index (int): center node index
            neighbor_index (int): neighbor node index
            image (np.array): the periodic cell image the neighbor is from
            distance (float): distance between center and neighbor.
            dist_tol (float): tolerance for distance comparison between edges.
                Default = 1e-6
        """
        directed_edge_index = len(self.directed_edges_list)
        this_directed_edge = DirectedEdge(
            [center_index, neighbor_index],
            index=directed_edge_index,
            info={"image": image, "distance": distance},
        )
        tmp = frozenset([center_index, neighbor_index])
        if tmp not in self.undirected_edges:
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp] = [this_undirected_edge]
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)
        else:
            for undirected_edge in self.undirected_edges[tmp]:
                if (
                    abs(undirected_edge.info["distance"] - distance) < dist_tol
                    and len(undirected_edge.info["directed_edge_index"]) == 1
                ):
                    added_dir_edge = self.directed_edges_list[
                        undirected_edge.info["directed_edge_index"][0]
                    ]
                    if added_dir_edge == this_directed_edge:
                        this_directed_edge.info[
                            "undirected_edge_index"
                        ] = added_dir_edge.info["undirected_edge_index"]
                        self.nodes[center_index].add_neighbor(
                            neighbor_index, this_directed_edge
                        )
                        self.directed_edges_list.append(this_directed_edge)
                        undirected_edge.info["directed_edge_index"].append(
                            directed_edge_index
                        )
                        return
            this_directed_edge.info["undirected_edge_index"] = len(
                self.undirected_edges_list
            )
            this_undirected_edge = this_directed_edge.make_undirected(
                index=len(self.undirected_edges_list),
                info={"directed_edge_index": [directed_edge_index]},
            )
            self.undirected_edges[tmp].append(this_undirected_edge)
            self.undirected_edges_list.append(this_undirected_edge)
            self.nodes[center_index].add_neighbor(neighbor_index, this_directed_edge)
            self.directed_edges_list.append(this_directed_edge)

    def adjacency_list(self) -> tuple[list[list[int]], list[int]]:
        """Get the adjacency list
        Return:
            graph: the adjacency list
                [[0, 1],
                 [0, 2],
                 ...
                 [5, 2]
                 ...  ]]
                the fist column specifies center/source node,
                the second column specifies neighbor/destination node
            directed2undirected:
                [0, 1, ...]
                a list of length = num_directed_edge that specifies
                the undirected edge index corresponding to the directed edges
                represented in each row in the graph adjacency list.
        """
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]
        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff) -> tuple[list[list[int]], list[int]]:
        """Get the line graph adjacency list.

        Args:
            cutoff (float): a float to indicate the maximum edge length to be included
                in constructing the line graph, this is used to decrease computation
                complexity

        Return:
            line_graph:
                [[0, 1, 1, 2, 2],
                [0, 1, 1, 4, 23],
                [1, 4, 23, 5, 66],
                ... ...  ]
                the fist column specifies node(atom) index at this angle,
                the second column specifies 1st undirected edge(left bond) index,
                the third column specifies 1st directed edge(left bond) index,
                the fourth column specifies 2nd undirected edge(right bond) index,
                the fifth column specifies 2nd directed edge(right bond) index,.
            undirected2directed:
                [32, 45, ...]
                a list of length = num_undirected_edge that
                maps the undirected edge index to one of its directed edges indices
        """
        if len(self.directed_edges_list) != 2 * len(self.undirected_edges_list):
            raise ValueError(
                f"Error: number of directed edges={len(self.directed_edges_list)} "
                f"!= 2 * number of undirected edges={len(self.undirected_edges_list)}"
                "!This indicates directed edges are not complete"
            )
        line_graph = []
        undirected2directed = []
        for u_edge in self.undirected_edges_list:
            undirected2directed.append(u_edge.info["directed_edge_index"][0])
            if u_edge.info["distance"] > cutoff:
                continue
            if len(u_edge.info["directed_edge_index"]) != 2:
                raise ValueError(
                    f"Did not find 2 Directed_edges !!!undirected edge {u_edge} "
                    "has:edge.info['directed_edge_index'] = "
                    f"{u_edge.info['directed_edge_index']}len directed_edges_list = "
                    f"{len(self.directed_edges_list)}len undirected_edges_list = "
                    f"{len(self.undirected_edges_list)}"
                )
            for center, dir_edge in zip(
                u_edge.nodes, u_edge.info["directed_edge_index"], strict=True
            ):
                for directed_edges in self.nodes[center].neighbors.values():
                    for directed_edge in directed_edges:
                        if directed_edge.index == dir_edge:
                            continue
                        if directed_edge.info["distance"] < cutoff:
                            line_graph.append(
                                [
                                    center,
                                    u_edge.index,
                                    dir_edge,
                                    directed_edge.info["undirected_edge_index"],
                                    directed_edge.index,
                                ]
                            )
        return line_graph, undirected2directed

    def undirected2directed(self) -> list[int]:
        """The index map from undirected_edge index to one of its directed_edge
        index.
        """
        return [
            undirected_edge.info["directed_edge_index"][0]
            for undirected_edge in self.undirected_edges_list
        ]

    def as_dict(self) -> dict:
        """Return dictionary serialization of a Graph."""
        return {
            "nodes": self.nodes,
            "directed_edges": self.directed_edges,
            "directed_edges_list": self.directed_edges_list,
            "undirected_edges": self.undirected_edges,
            "undirected_edges_list": self.undirected_edges_list,
        }

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return (
            f"Graph(num_nodes={num_nodes!r}, num_directed_edges={num_directed_edges!r},"
            f" num_undirected_edges={num_undirected_edges!r})"
        )


class CHGNetGraphConverter:
    """Convert a structure to a CHGNet graph.

    https://www.nature.com/articles/s42256-023-00716-3

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        pbc (tuple[int, int, int], optional): Periodic boundary conditions.
            Defaults to (1, 1, 1).
        neighbor_strategy (str, optional): Strategy to determine neighbors.
            Defaults to "k-nearest".
        num_classes (int, optional): Number of classes. Defaults to 95.
        atom_graph_cutoff (float, optional): Atom graph cutoff. Defaults to 6.0.
        bond_graph_cutoff (float, optional): Bond graph cutoff. Defaults to 3.0.
        num_cpus (Optional[int], optional): Number of CPUs to use. Defaults to None.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        num_classes: int = 95,
        atom_graph_cutoff: float = 6.0,  # only used for method='chgnet_graph'
        bond_graph_cutoff: float = 3.0,  # only used for method='chgnet_graph'
        num_cpus: Optional[int] = None,
        **kwargs,  # any additional arguments
    ) -> None:

        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.num_classes = num_classes
        self.atom_graph_cutoff = atom_graph_cutoff
        self.bond_graph_cutoff = bond_graph_cutoff

        self.num_cpus = num_cpus
        self.eps = 1e-8

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_chgnet_graph(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_chgnet_graph,
                structure,
                num_cpus=self.num_cpus,
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [
            #     self.get_graph_by_chgnet_graph(struc) for struc in structure
            # ]
        return graph

    def get_graph_by_chgnet_graph(self, structure: Structure):
        n_atoms = len(structure)

        # for graph
        center_index, neighbor_index, image, distance = structure.get_neighbor_list(
            r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-08
        )
        graph_utils = GraphUtils([Node(index=idx) for idx in range(n_atoms)])
        for ii, jj, img, dist in zip(
            center_index, neighbor_index, image, distance, strict=True
        ):
            graph_utils.add_edge(
                center_index=ii, neighbor_index=jj, image=img, distance=dist
            )
        atom_graph, directed2undirected = graph_utils.adjacency_list()
        bond_graph, undirected2directed = graph_utils.line_graph_adjacency_list(
            cutoff=self.bond_graph_cutoff
        )
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})

        edge_indices = [
            (idx1, idx2) for idx1, idx2 in zip(center_index, neighbor_index)
        ]
        if len(edge_indices) == 0:
            edge_indices = np.zeros((0, 2), dtype="int64")
        if len(bond_graph) == 0:
            bond_graph = np.zeros((0, 5)).astype(np.int32)
        graph = self.build_pgl_graph(
            structure,
            edge_indices=edge_indices,
            to_jimages=image,
            edge_features={
                "atom_graph": np.asarray(atom_graph, dtype=np.int32),
                "bond_graph": np.asarray(bond_graph, dtype=np.int32),
                "directed2undirected": np.asarray(directed2undirected, dtype=np.int32),
                "undirected2directed": np.asarray(undirected2directed, dtype=np.int32),
                "directed2undirected_len": np.array(
                    [len(directed2undirected)], dtype=np.int32
                ),
                "undirected2directed_len": np.array(
                    [len(undirected2directed)], dtype=np.int32
                ),
                "image": np.asarray(image, dtype="float32"),
            },
        )

        atom_types = graph.node_feat["atom_types"]
        composition_fea = np.bincount(atom_types - 1, minlength=self.num_classes - 1)
        composition_fea = composition_fea / atom_types.shape[0]
        graph.node_feat["composition_fea"] = np.asarray([composition_fea]).astype(
            np.float32
        )

        graph.edge_feat["bond_vec"] = (
            graph.edge_feat["bond_vec"] / graph.edge_feat["bond_dist"][:, None]
        )

        graph.edge_feat["undirected_bond_lengths"] = graph.edge_feat["bond_dist"][
            undirected2directed
        ]

        if len(bond_graph) != 0:
            graph.edge_feat["bond_vec_i"] = graph.edge_feat["bond_vec"][
                np.asarray(bond_graph)[:, 2]
            ]
            graph.edge_feat["bond_vec_j"] = graph.edge_feat["bond_vec"][
                np.asarray(bond_graph)[:, 4]
            ]
        else:
            graph.edge_feat["bond_vec_i"] = np.zeros((0, 3), dtype=np.float32)
            graph.edge_feat["bond_vec_j"] = np.zeros((0, 3), dtype=np.float32)

        graph.edge_feat["num_atom_graph"] = np.array([len(atom_graph)], dtype=np.int32)
        graph.edge_feat["num_bond_graph"] = np.array([len(bond_graph)], dtype=np.int32)

        if n_isolated_atoms:
            graph.node_feat["isolation_flag"] = np.array([1])
        else:
            graph.node_feat["isolation_flag"] = np.array([0])

        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        assert node_features is None or isinstance(node_features, dict)
        assert edge_features is None or isinstance(edge_features, dict)

        # get atom types
        atom_types = np.array([site.specie.Z for site in structure])

        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        # convert to numpy array
        edge_indices = np.array(edge_indices)
        if to_jimages is not None:
            to_jimages = np.array(to_jimages)
        num_atoms = tuple(atom_types.shape)[0]

        # After multiple graph batch operations by the dataloader,
        # graph.num_nodes remains an integer, which is the sum of the number of
        # nodes in all graphs
        graph = pgl.Graph(edge_indices, num_nodes=num_atoms)
        # node features: frac_coords, cart_coords, atom_types
        graph.node_feat["frac_coords"] = structure.frac_coords.astype("float32")
        graph.node_feat["cart_coords"] = structure.cart_coords.astype("float32")
        graph.node_feat["atom_types"] = atom_types

        # graph features: lengths, angles, lattice, num_atoms
        # Due to the inability of pgl.graph to store graph level features,
        # we will store these features under node_feat
        graph.node_feat["lengths"] = lengths
        graph.node_feat["angles"] = angles
        graph.node_feat["lattice"] = lattice.reshape(1, 3, 3)
        # graph.node_feat['num_atoms'] is different from graph.num_nodes
        # After multiple graph batch operations by the dataloader,
        # graph.node_feat['num_atoms'] is a tensor of shape (batch_size),
        # where each value is the number of atoms in the corresponding graph.
        graph.node_feat["num_atoms"] = np.array([num_atoms])
        # edge features: pbc_offset, bond_vec, bond_dist
        if to_jimages is not None:
            graph.edge_feat["pbc_offset"] = to_jimages
            offset = np.matmul(to_jimages, lattice)
            dst_pos = graph.node_feat["cart_coords"][graph.edges[:, 1]] + offset
            src_pos = graph.node_feat["cart_coords"][graph.edges[:, 0]]
            bond_vec = dst_pos - src_pos
            bond_dist = np.linalg.norm(bond_vec, axis=1)
            graph.edge_feat["bond_vec"] = bond_vec.astype("float32")
            graph.edge_feat["bond_dist"] = bond_dist.astype("float32")
        graph.edge_feat["num_edges"] = np.array([edge_indices.shape[0]])

        if node_features is not None:
            graph.node_feat.update(node_features)
        if edge_features is not None:
            graph.edge_feat.update(edge_features)
        return graph
