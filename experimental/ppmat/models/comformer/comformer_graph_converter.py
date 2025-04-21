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

# This code is adapted from https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pgl
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from p_tqdm import p_map
from pymatgen.core.structure import Structure


def same_line(a, b):
    a_new = a / (sum(a**2) ** 0.5)
    b_new = b / (sum(b**2) ** 0.5)
    flag = False
    if abs(sum(a_new * b_new) - 1.0) < 1e-5:
        flag = True
    elif abs(sum(a_new * b_new) + 1.0) < 1e-5:
        flag = True
    else:
        flag = False
    return flag


def same_plane(a, b, c):
    flag = False
    if abs(np.dot(np.cross(a, b), c)) < 1e-5:
        flag = True
    return flag


def angle_from_array(a, b, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    assert a_new.shape == a.shape
    value = sum(a_new * b_new)
    length = (sum(a_new**2) ** 0.5) * (sum(b_new**2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0


def correct_coord_sys(a, b, c, lattice):
    a_new = np.dot(a, lattice)
    b_new = np.dot(b, lattice)
    c_new = np.dot(c, lattice)
    assert a_new.shape == a.shape
    plane_vec = np.cross(a_new, b_new)
    value = sum(plane_vec * c_new)
    length = (sum(plane_vec**2) ** 0.5) * (sum(c_new**2) ** 0.5)
    cos = value / length
    angle = np.arccos(cos)
    return angle / np.pi * 180.0 <= 90.0


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids shift periodic images so the first vertex is in (0,0,0) image.
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    use_canonize=False,
    use_lattice=False,
):
    """Construct k-NN edge list."""
    lat = atoms.lattice
    all_neighbors_now = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors_now)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            use_lattice=use_lattice,
        )

    edges = defaultdict(set)
    # lattice correction process
    r_cut = max(lat.a, lat.b, lat.c) + 1e-2
    all_neighbors = atoms.get_all_neighbors(r=r_cut)
    neighborlist = all_neighbors[0]
    neighborlist = sorted(neighborlist, key=lambda x: x[2])
    ids = np.array([nbr[1] for nbr in neighborlist])
    images = np.array([nbr[3] for nbr in neighborlist])
    images = images[ids == 0]
    lat1 = images[0]
    # finding lat2
    start = 1
    for i in range(start, len(images)):
        lat2 = images[i]
        if not same_line(lat1, lat2):
            start = i
            break
    # finding lat3
    for i in range(start, len(images)):
        lat3 = images[i]
        if not same_plane(lat1, lat2, lat3):
            break
    # find the invariant corner
    if angle_from_array(lat1, lat2, lat.matrix) > 90.0:
        lat2 = -lat2
    if angle_from_array(lat1, lat3, lat.matrix) > 90.0:
        lat3 = -lat3
    # find the invariant coord system
    if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
        lat1 = -lat1
        lat2 = -lat2
        lat3 = -lat3

    # if not correct_coord_sys(lat1, lat2, lat3, lat.matrix):
    #     print(lat1, lat2, lat3)
    # lattice correction end
    for site_idx, neighborlist in enumerate(all_neighbors_now):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(lat1))
            edges[(site_idx, site_idx)].add(tuple(lat2))
            edges[(site_idx, site_idx)].add(tuple(lat3))

    return edges, lat1, lat2, lat3


def build_undirected_edgedata(
    atoms=None,
    edges={},
    a=None,
    b=None,
    c=None,
):
    """Build undirected graph data from edge set."""
    # second pass: construct *undirected* graph
    u, v, r, nei, atom_lat = [], [], [], [], []
    v1, v2, v3 = (
        atoms.lattice.cart_coords(a),
        atoms.lattice.cart_coords(b),
        atoms.lattice.cart_coords(c),
    )
    atom_lat.append([v1, v2, v3])
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                nei.append([v1, v2, v3])

    u = np.asarray(u, dtype="int64")
    v = np.asarray(v, dtype="int64")
    r = np.asarray(r, dtype="float32")
    nei = np.asarray(nei, dtype="float32")
    atom_lat = np.asarray(atom_lat, dtype="float32")
    return u, v, r, nei, atom_lat


def atom_multigraph(
    atoms=None,
    neighbor_strategy="k-nearest",
    cutoff=4.0,
    max_neighbors=25,
    atom_features="cgcnn",
    use_canonize: bool = False,
    use_lattice: bool = True,
):
    if neighbor_strategy == "k-nearest":
        edges, a, b, c = nearest_neighbor_edges_submit(
            atoms=atoms,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            use_canonize=use_canonize,
            use_lattice=use_lattice,
        )
        u, v, r, nei, atom_lat = build_undirected_edgedata(atoms, edges, a, b, c)
    else:
        raise ValueError("Not implemented yet", neighbor_strategy)

    # # build up atom attribute tensor
    sps_features = []
    for _, s in enumerate(atoms.elements):
        feat = list(get_node_attributes(s, atom_features=atom_features))
        sps_features.append(feat)
    node_features = np.array(sps_features)
    atom_lat = atom_lat.repeat(node_features.shape[0], axis=0)
    edge_index = np.stack([u, v], axis=1)

    return edge_index, node_features, r, nei, atom_lat


class ComformerGraphConverter:
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

        self.num_cpus = num_cpus
        self.eps = 1e-8

    def __call__(self, structure: Structure):
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
