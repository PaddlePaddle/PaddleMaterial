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

from ppmat.datasets.comformer_graph_utils import atom_dgl_multigraph
from ppmat.datasets.utils import lattice_params_to_matrix
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


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
        method: Literal[
            "crystalnn", "find_points_in_spheres", "comformer_graph"
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

        assert method in [
            "crystalnn",
            "find_points_in_spheres",
            "comformer_graph",
        ], "method must be 'crystalnn' 'comformer_graph' or 'find_points_in_spheres'."
        self.method = method

        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
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
