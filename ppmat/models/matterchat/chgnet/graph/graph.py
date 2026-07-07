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

from ppmat.models.matterchat.chgnet.utils import write_json


class Node:
    """A node in a graph."""

    def __init__(self, index: int, info: dict = None) -> None:
        self.index = index
        self.info = info
        self.neighbors: dict[int, list] = {}

    def add_neighbor(self, index, edge):
        """Draw a directed edge between self and the node specified by index."""
        if index not in self.neighbors:
            self.neighbors[index] = [edge]
        else:
            self.neighbors[index].append(edge)


class UndirectedEdge:
    """An undirected/bi-directed edge in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        self.nodes = nodes
        self.index = index
        self.info = info

    def __repr__(self):
        return (
            f"UndirectedEdge between Nodes{self.nodes}, "
            f"info={self.info}, index={self.index}"
        )

    def __eq__(self, other):
        return set(self.nodes) == set(other.nodes) and self.info == other.info


class DirectedEdge:
    """A directed edge in a graph."""

    def __init__(
        self, nodes: list, index: int | None = None, info: dict | None = None
    ) -> None:
        self.nodes = nodes
        self.index = index
        self.info = info

    def make_undirected(self, index, info=None):
        """Make a directed edge undirected."""
        if info is None:
            info = {}
        info["distance"] = self.info["distance"]
        return UndirectedEdge(self.nodes, index, info)

    def __eq__(self, other) -> bool:
        """Check if two directed edges are equal or reverse of each other."""
        import numpy as np

        self_image = self.info["image"]
        other_image = other.info["image"]

        if isinstance(self_image, np.ndarray):
            eq_check = np.array_equal(self_image, other_image)
        else:
            eq_check = (self_image == other_image)

        if (
            self.nodes == other.nodes
            and eq_check
        ):
            print(
                "!!!!!! the two directed edges are equal but this operation is "
                "not supposed to happen"
            )
            return True

        neg_other_image = -1 * other_image
        if isinstance(neg_other_image, np.ndarray):
            neg_eq_check = np.array_equal(self_image, neg_other_image)
        else:
            neg_eq_check = (self_image == neg_other_image)

        if (
            self.nodes == other.nodes[::-1]
            and neg_eq_check
        ):
            return True
        return False

    def __repr__(self):
        return (
            f"DirectedEdge between Nodes{self.nodes}, "
            f"info={self.info}, index={self.index}"
        )


class Graph:
    """A graph for storing the neighbor information of atoms."""

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes
        self.directed_edges: dict[frozenset[int], list[DirectedEdge]] = {}
        self.directed_edges_list: list[DirectedEdge] = []
        self.undirected_edges: dict[frozenset[int], list[UndirectedEdge]] = {}
        self.undirected_edges_list: list[UndirectedEdge] = []

    def add_edge(self, center_index, neighbor_index, image, distance) -> None:
        """Add a directed edge to the graph."""
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
            # Check if this is the reverse directed edge of an existing undirected edge
            for undirected_edge in self.undirected_edges[tmp]:
                if (
                    abs(undirected_edge.info["distance"] - distance) < 1e-6
                    and len(undirected_edge.info["directed_edge_index"]) == 1
                ):
                    added_DE = self.directed_edges_list[
                        undirected_edge.info["directed_edge_index"][0]
                    ]
                    if added_DE == this_directed_edge:
                        this_directed_edge.info[
                            "undirected_edge_index"
                        ] = added_DE.info["undirected_edge_index"]
                        self.nodes[center_index].add_neighbor(
                            neighbor_index, this_directed_edge
                        )
                        self.directed_edges_list.append(this_directed_edge)
                        undirected_edge.info["directed_edge_index"].append(
                            directed_edge_index
                        )
                        return

            # No matching undirected edge; create a new one
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

    def adjacency_list(self):
        """Get the adjacency list."""
        graph = [edge.nodes for edge in self.directed_edges_list]
        directed2undirected = [
            edge.info["undirected_edge_index"] for edge in self.directed_edges_list
        ]
        return graph, directed2undirected

    def line_graph_adjacency_list(self, cutoff):
        """Get the line graph adjacency list."""
        assert len(self.directed_edges_list) == 2 * len(self.undirected_edges_list), (
            f"Error: number of directed edges={len(self.directed_edges_list)} != 2 * "
            f"number of undirected edges={len(self.undirected_edges_list)}!"
            f"This indicates directed edges are not complete"
        )
        line_graph = []
        undirected2directed = []
        for u_edge in self.undirected_edges_list:
            undirected2directed.append(u_edge.info["directed_edge_index"][0])
            if u_edge.info["distance"] > cutoff:
                continue
            center1, center2 = list(u_edge.nodes)
            try:
                directed_edge1, directed_edge2 = u_edge.info["directed_edge_index"]
            except ValueError:
                print("Did not find 2 Directed_edges !!!")
                print(u_edge)
                print(
                    "edge.info['directed_edge_index'] = ",
                    u_edge.info["directed_edge_index"],
                )
                print()
                print("len directed_edges_list = ", len(self.directed_edges_list))
                print("len undirected_edges_list = ", len(self.undirected_edges_list))
            for directed_edges in self.nodes[center1].neighbors.values():
                for directed_edge in directed_edges:
                    if directed_edge.index == directed_edge1:
                        continue
                    if directed_edge.info["distance"] < cutoff:
                        line_graph.append(
                            [
                                center1,
                                u_edge.index,
                                directed_edge1,
                                directed_edge.info["undirected_edge_index"],
                                directed_edge.index,
                            ]
                        )
            for directed_edges in self.nodes[center2].neighbors.values():
                for directed_edge in directed_edges:
                    if directed_edge.index == directed_edge2:
                        continue
                    if directed_edge.info["distance"] < cutoff:
                        line_graph.append(
                            [
                                center2,
                                u_edge.index,
                                directed_edge2,
                                directed_edge.info["undirected_edge_index"],
                                directed_edge.index,
                            ]
                        )
        return line_graph, undirected2directed

    def undirected2directed(self):
        """Map undirected_edge index to one of its directed_edge indices."""
        out = []
        for undirected_edge in self.undirected_edges_list:
            out.append(undirected_edge.info["directed_edge_index"][0])
        return out

    def as_dict(self):
        """Return dictionary serialization of a Graph."""
        return {
            "nodes": self.nodes,
            "directed_edges": self.directed_edges,
            "directed_edges_list": self.directed_edges_list,
            "undirected_edges": self.undirected_edges,
            "undirected_edges_list": self.undirected_edges_list,
        }

    def to(self, filename="graph.json"):
        """Save graph dictionary to file."""
        write_json(self.as_dict(), filename)

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return f"Graph({num_nodes=}, {num_directed_edges=}, {num_undirected_edges=})"
