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
from __future__ import annotations

from typing import Literal
from typing import Optional

import numpy as np
import pgl
from jarvis.core.atoms import Atoms
from p_tqdm import p_map
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

from ppmat.datasets.graph_utils.chgnet_graph_utils import GraphUtils
from ppmat.datasets.graph_utils.chgnet_graph_utils import Node
from ppmat.datasets.graph_utils.comformer_graph_utils import atom_multigraph
from ppmat.utils import logger


class Structure2Graph:
    """Convert a structure to a graph.

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        pbc (tuple[int, int, int], optional): Periodic boundary conditions.
            Defaults to (1, 1, 1).
        neighbor_strategy (str, optional): Strategy to determine neighbors.
            Defaults to "k-nearest".
        max_neighbors (int, optional): Maximum number of neighbors. Defaults to 25.
        atom_features (str, optional): Atom features. Defaults to "cgcnn".
        use_canonize (bool, optional): Whether to use canonize. Defaults to True.
        use_lattice (bool, optional): Whether to use lattice. Defaults to True.
        atom_graph_cutoff (float, optional): Atom graph cutoff. Defaults to 6.0.
        bond_graph_cutoff (float, optional): Bond graph cutoff. Defaults to 3.0.
        composition_fea_len (int, optional): Length of composition feature.
            Defaults to 94.
        method (Literal[&quot;crystalnn&quot;, &quot;find_points_in_spheres&quot;,
            &quot;comformer_graph&quot;, &quot;chgnet_graph&quot;], optional): Method
            to convert structure to graph. Defaults to "crystalnn".
        element_types (Literal[&quot;DEFAULT_ELEMENTS&quot;], optional): Element types.
            Defaults to "DEFAULT_ELEMENTS".
        num_cpus (Optional[int], optional): Number of CPUs to use. Defaults to None.
    """

    # TODO: Reorganize input parameters through graph method
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
        composition_fea_len: int = 94,  # only used for method='chgnet_graph'
        method: Literal[
            "crystalnn", "find_points_in_spheres", "comformer_graph", "chgnet_graph"
        ] = "crystalnn",
        num_cpus: Optional[int] = None,
        **kwargs,  # any additional arguments
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
        self.composition_fea_len = composition_fea_len

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

        self.num_cpus = num_cpus
        self.eps = 1e-8

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
        edge_index, node_features, r, nei, atom_lat = atom_multigraph(
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
            atom_types - 1, minlength=self.composition_fea_len
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
