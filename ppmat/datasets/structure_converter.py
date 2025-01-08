from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pgl
from jarvis.core.atoms import Atoms
from p_tqdm import p_map
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from rdkit.Chem import DataStructs as Mol
from rdkit.Chem.rdchem import BondType as BT

from ppmat.datasets.comformer_graph_utils import atom_dgl_multigraph
from ppmat.datasets.utils import lattice_params_to_matrix
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import ELEMENTS_94
from ppmat.utils import logger
from ppmat.utils.io import write_json


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

    def to(self, filename="graph.json") -> None:
        """Save graph dictionary to file."""
        write_json(filename, self.as_dict())

    def __repr__(self) -> str:
        """Return string representation of the Graph."""
        num_nodes = len(self.nodes)
        num_directed_edges = len(self.directed_edges_list)
        num_undirected_edges = len(self.undirected_edges_list)
        return (
            f"Graph(num_nodes={num_nodes!r}, num_directed_edges={num_directed_edges!r},"
            f" num_undirected_edges={num_undirected_edges!r})"
        )


class Structure2Graph:
    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        neighbor_strategy: str = "k-nearest",  # only used for method='comformer_graph'
        max_neighbors: int = 25,  # only used for method='comformer_graph'
        atom_features: str = "cgcnn",  # only used for method='comformer_graph'
        use_canonize: bool = True,  # only used for method='comformer_graph'
        use_lattice: bool = True,  # only used for method='comformer_graph'
        atom_graph_cutoff: float = 6.0,  # only used for method='chgnet_graph'
        bond_graph_cutoff: float = 3.0,  # only used for method='chgnet_graph'
        method: Literal[
            "crystalnn", "find_points_in_spheres", "comformer_graph", "chgnet_graph"
        ] = "crystalnn",
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
        num_cpus: Optional[int] = None,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.neighbor_strategy = neighbor_strategy
        self.max_neighbors = max_neighbors
        self.atom_features = atom_features
        self.use_canonize = use_canonize
        self.use_lattice = use_lattice

        self.atom_graph_cutoff = atom_graph_cutoff
        self.bond_graph_cutoff = bond_graph_cutoff

        assert method in [
            "crystalnn",
            "find_points_in_spheres",
            "comformer_graph",
            "chgnet_graph",
        ], (
            "method must be 'crystalnn' 'comformer_graph' 'find_points_in_spheres' "
            "or 'chgnet_graph'."
        )
        self.method = method

        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        elif element_types.upper() == "ELEMENTS_94":
            self.element_types = ELEMENTS_94
        else:
            raise ValueError("element_types must be 'DEFAULT_ELEMENTS'.")
        self.element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        self.num_cpus = num_cpus
        self.eps = eps

        if self.method == "crystalnn":
            self.CrystalNN = local_env.CrystalNN(
                distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
            )

    def __call__(self, structure: Structure):
        if self.method == "crystalnn":
            if isinstance(structure, Structure):
                graph = self.get_graph_by_crystalnn(structure)
            elif isinstance(structure, list):
                graph = p_map(
                    self.get_graph_by_crystalnn, structure, num_cpus=self.num_cpus
                )
                # the following code is equivalent to the above line, it is slower,
                # but easier to debug.
                # graph = [self.get_graph_by_crystalnn(struc) for struc in structure]
            else:
                raise TypeError(
                    "The input must be a pymatgen.Structure or a list of them."
                )
        elif self.method == "find_points_in_spheres":
            if isinstance(structure, Structure):
                graph = self.get_graph_by_find_points_in_spheres(structure)
            elif isinstance(structure, list):
                graph = p_map(
                    self.get_graph_by_find_points_in_spheres,
                    structure,
                    num_cpus=self.num_cpus,
                )
                # the following code is equivalent to the above line, it is slower,
                # but easier to debug.
                # graph = [
                # self.get_graph_by_find_points_in_spheres(struc)
                #     for struc in structure
                # ]
            else:
                raise TypeError(
                    "The input must be a pymatgen.Structure or a list of them."
                )
        elif self.method == "comformer_graph":
            if isinstance(structure, Structure):
                graph = self.get_graph_by_comformer_graph(structure)
            elif isinstance(structure, list):
                graph = p_map(
                    self.get_graph_by_comformer_graph,
                    structure,
                    num_cpus=self.num_cpus,
                )
                # the following code is equivalent to the above line, it is slower,
                # but easier to debug.
                # graph = [
                #     self.get_graph_by_comformer_graph(struc) for struc in structure
                # ]
        elif self.method == "chgnet_graph":
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
                # graph = []
                # for i, struc in enumerate(structure):
                #     print(i)
                #     g = self.get_graph_by_chgnet_graph(struc)
                #     graph.append(g)

        else:
            raise NotImplementedError()
        return graph

    def get_graph_by_comformer_graph(self, structure: Structure):
        # Convert pymatgen structure to jarvis atoms
        lattice_mat = structure.lattice.matrix
        coords = structure.frac_coords
        elements = [site.specie.symbol for site in structure]
        atoms = Atoms(lattice_mat=lattice_mat, coords=coords, elements=elements)
        edge_index, node_features, r, nei, atom_lat = atom_dgl_multigraph(
            atoms,
            neighbor_strategy=self.neighbor_strategy,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            atom_features=self.atom_features,
            use_canonize=self.use_canonize,
            use_lattice=self.use_lattice,
        )
        graph = self.build_pgl_graph(
            structure,
            edge_indices=edge_index,
            to_jimages=None,
            node_features={"node_feat": node_features, "atom_lat": atom_lat},
            edge_features={
                "r": r,
                "nei": nei,
            },
        )
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
        composition_fea = np.bincount(
            atom_types - 1, minlength=len(self.element_types) - 1
        )
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

    def get_graph_by_crystalnn(self, structure: Structure):

        try:
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, self.CrystalNN
            )
        except Exception:
            search_cutoff = 10
            while True:
                try:
                    crystalNN_tmp = local_env.CrystalNN(
                        distance_cutoffs=None,
                        x_diff_weight=-1,
                        porous_adjustment=False,
                        search_cutoff=search_cutoff,
                    )
                    structure_graph = StructureGraph.with_local_env_strategy(
                        structure, crystalNN_tmp
                    )
                    logger.info(
                        "Successfully generated graph by CrystalNN with "
                        f"search_cutoff={search_cutoff}."
                    )
                    break
                except Exception:
                    search_cutoff += 2
                    logger.info(f"Searching for new search_cutoff{search_cutoff}...")
                    if search_cutoff > 40:
                        logger.info(
                            "Failed to generate graph by CrystalNN with "
                            f"search_cutoff={search_cutoff}. "
                        )
                        break

        # atom_types = np.array(structure.atomic_numbers)
        lattice_parameters = structure.lattice.parameters
        lengths = lattice_parameters[:3]
        angles = lattice_parameters[3:]
        assert np.allclose(
            structure.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
        )

        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

        graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
        return graph

    def get_graph_by_find_points_in_spheres(self, structure: Structure):
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords

        cutoff = self.cutoff
        attempt = 3
        while attempt > 0:
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=cutoff,
                pbc=self.pbc,
                lattice=lattice_matrix,
                tol=self.eps,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > self.eps)
            src_id, dst_id, images, bond_dist = (
                src_id[exclude_self],
                dst_id[exclude_self],
                images[exclude_self],
                bond_dist[exclude_self],
            )

            edge_indices = [(u, v) for u, v in zip(src_id, dst_id)]
            to_jimages = np.array(images, dtype="float32")
            if len(edge_indices) == 0:
                logger.warning(
                    f"No edges found within cutoff {cutoff:.5f}. Trying again with "
                    "larger cutoff."
                )
                cutoff *= 2
                attempt -= 1
            else:
                break
        if len(edge_indices) == 0:
            raise RuntimeError(
                f"No edges found within cutoff {cutoff:.5f}. Please increase the "
                "cutoff."
            )

        graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
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
        atom_types = np.array(
            [self.element_types.index(site.specie.symbol) for site in structure]
        )

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


class Mol2Graph:
    def __init__(
        self,
        method: Literal["general"] = "general",
        remove_hydrogen: bool = False,
        num_cpus: Optional[int] = None,
        **kwargs,
    ):
        assert method in ["general"], "method must be 'general'."
        self.method = method
        self.remove_h = remove_hydrogen
        self.num_cpus = num_cpus

    def __call__(self, mol: Union[Mol, List[Mol]]):
        if self.method == "general":
            if isinstance(mol, Mol):
                graph = self.get_graph_by_general_method(mol)
            elif isinstance(mol, list):
                graph = p_map(
                    self.get_graph_by_general_method, mol, num_cpus=self.num_cpus
                )
                # the following code is equivalent to the above line, it is slower,
                # but easier to debug.
                # graph = [self.get_graph_by_general_method(mol_) for mol_ in mol]
            else:
                raise TypeError(
                    "The input must be a rdkit.chem.rdchem.Mol or a list of them."
                )
        else:
            raise NotImplementedError()
        return graph

    def get_graph_by_general_method(self, mol: Mol):
        # Convert rdkit mol to graph by general method
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "P": 5,
            "S": 6,
            "Cl": 7,
            "Br": 8,
            "I": 9,
        }

        # Number of atoms
        N = mol.GetNumAtoms()

        # Atom type indices
        type_idx = [types[atom.GetSymbol()] for atom in mol.GetAtoms()]

        # Edge construction
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()] + 1]

        edge_index = np.array([row, col], dtype=np.int64)
        edge_attr = np.eye(len(bonds) + 1)[edge_type].astype(np.float32)

        # Sort edges
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

        # Node features
        x = np.eye(len(types))[type_idx].astype(np.float32)

        # Remove hydrogens if specified
        if self.remove_h:
            to_keep = np.array(type_idx) > 0
            edge_index, edge_attr = self.subgraph(to_keep, edge_index, edge_attr)
            x = x[to_keep][:, 1:]  # Remove hydrogen feature)

            x = x[to_keep]
            x = x[:, 1:]

        # Placeholder for graph-lever features
        y = np.zeros((1, 0), dtype=np.float32)

        # Build the graph
        graph = self.build_pgl_graph(
            mol,
            edge_indexs=edge_index,
            node_features={"x": x},
            edge_features={"edge_attr": edge_attr},
            graph_features={"graphe_lever_features": y},
        )
        return graph

    def build_pgl_graph(
        self, edge_indices, node_features=None, edge_features=None, graph_features=None
    ):
        num_nodes = node_features["x"].shape[0]

        # Create a PGL graph
        graph = pgl.Graph(edge_indices.T, num_nodes=num_nodes)

        # Add node features
        for key, value in node_features.items():
            graph.node_feat[key] = value

        # Add edge features
        for key, value in edge_features.items():
            graph.edge_feat[key] = value

        # Add graph-level features
        if graph_features:
            for key, value in graph_features.items():
                graph.graph_feat[key] = value

        return graph

    @staticmethod
    def subgraph(to_keep, edge_index, edge_attr):
        mask = np.isin(edge_index, np.where(to_keep)[0]).all(axis=0)
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        remap = {old: new for new, old in enumerate(np.where(to_keep)[0])}
        edge_index = np.vectorize(remap.get)(edge_index)
        return edge_index, edge_attr
