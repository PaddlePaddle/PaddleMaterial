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

"""Periodic boundary condition (PBC) graph construction for crystal structures."""
from typing import Tuple

import paddle

from ppmat.utils.crystal import OFFSET_LIST
from ppmat.utils.scatter import scatter, scatter_min_with_argmin


def construct_fully_connected_graphs_with_periodic_boundaries(
    cart_coords: paddle.Tensor,
    lattice_matrix: paddle.Tensor,
    num_nodes_per_crystal: paddle.Tensor,
    break_minimum_edge_ties: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    batch_size = num_nodes_per_crystal.shape[0]
    atom_pos = cart_coords

    num_atoms_per_crystal_sqr = (num_nodes_per_crystal ** 2).cast(paddle.int64)

    first_node_index_per_crystal = (
        paddle.cumsum(num_nodes_per_crystal, axis=0) - num_nodes_per_crystal
    )
    first_node_index_per_crystal_expand = paddle.repeat_interleave(
        first_node_index_per_crystal, num_atoms_per_crystal_sqr
    )
    num_atoms_per_crystal_expand = paddle.repeat_interleave(
        num_nodes_per_crystal, num_atoms_per_crystal_sqr
    )

    num_atom_pairs = paddle.sum(num_atoms_per_crystal_sqr)
    index_sqr_offset = (
        paddle.cumsum(num_atoms_per_crystal_sqr, axis=0) - num_atoms_per_crystal_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(index_sqr_offset, num_atoms_per_crystal_sqr)
    atom_count_sqr = paddle.arange(int(num_atom_pairs.item())) - index_sqr_offset

    destination_index = (
        paddle.floor_divide(atom_count_sqr, num_atoms_per_crystal_expand)
        + first_node_index_per_crystal_expand
    )
    source_index = (
        atom_count_sqr % num_atoms_per_crystal_expand
        + first_node_index_per_crystal_expand
    )
    map_edge_to_crystal = paddle.arange(batch_size).repeat_interleave(
        num_atoms_per_crystal_sqr, axis=0
    )

    n_edges_per_crystal_before_masking = num_atoms_per_crystal_sqr

    source_position = atom_pos[source_index]
    destination_position = atom_pos[destination_index]

    num_supercell_images = len(OFFSET_LIST)
    supercell_frac_offsets = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
    batch_supercell_frac_offsets = supercell_frac_offsets.unsqueeze(0).expand(
        [batch_size, num_supercell_images, 3]
    )
    pbc_frac_offsets_per_source_atom = paddle.repeat_interleave(
        batch_supercell_frac_offsets, n_edges_per_crystal_before_masking, axis=0
    )

    pbc_cart_offsets_per_source_atom = paddle.bmm(
        pbc_frac_offsets_per_source_atom,
        paddle.repeat_interleave(
            lattice_matrix, n_edges_per_crystal_before_masking, axis=0
        ),
    )

    destination_position = destination_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
    source_position = (
        source_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
        + pbc_cart_offsets_per_source_atom
    )

    source_index = source_index.unsqueeze(1).expand([-1, num_supercell_images])
    destination_index = destination_index.unsqueeze(1).expand([-1, num_supercell_images])
    map_edge_to_crystal = map_edge_to_crystal.unsqueeze(1).expand([-1, num_supercell_images])

    inter_atom_distances = (source_position - destination_position).norm(axis=-1)

    mask = paddle.logical_or(
        source_index != destination_index,
        inter_atom_distances > 1e-5,
    )

    flat_mask = mask.reshape([-1])
    destination_index = destination_index[mask]
    source_index = source_index[mask]
    map_edge_to_crystal = map_edge_to_crystal[mask]
    pbc_frac_offsets = pbc_frac_offsets_per_source_atom.reshape([-1, 3])[flat_mask]
    pbc_cart_offsets = pbc_cart_offsets_per_source_atom.reshape([-1, 3])[flat_mask]

    source_position_flat = atom_pos[source_index]
    destination_position_flat = atom_pos[destination_index]
    inter_atom_distances = (
        destination_position_flat
        - source_position_flat
        - pbc_cart_offsets
    ).norm(axis=-1)

    (
        destination_index,
        source_index,
        pbc_frac_offsets,
        num_edges_per_crystal,
    ) = get_smallest_edge_per_primal_node_pair(
        dst_idx=destination_index,
        src_idx=source_index,
        map_edge_to_crystal=map_edge_to_crystal,
        inter_atom_distances=inter_atom_distances,
        pbc_frac_offsets_per_source_atom=pbc_frac_offsets,
        num_nodes_per_crystal=num_nodes_per_crystal,
        break_minimum_edge_ties=break_minimum_edge_ties,
    )
    return (
        destination_index,
        source_index,
        pbc_frac_offsets,
        num_edges_per_crystal,
    )


def get_smallest_edge_per_primal_node_pair(
    dst_idx: paddle.Tensor,
    src_idx: paddle.Tensor,
    map_edge_to_crystal: paddle.Tensor,
    inter_atom_distances: paddle.Tensor,
    pbc_frac_offsets_per_source_atom: paddle.Tensor,
    num_nodes_per_crystal: paddle.Tensor,
    break_minimum_edge_ties: bool,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    edges = paddle.stack([dst_idx, src_idx], axis=-1)
    unique_edges, map_edge_to_unique = paddle.unique(edges, axis=0, return_inverse=True)

    min_inter_atom_distances, smallest_edge_idxs = scatter_min_with_argmin(
        src=inter_atom_distances,
        index=map_edge_to_unique,
        dim_size=unique_edges.shape[0],
    )

    batch_size = num_nodes_per_crystal.shape[0]

    if break_minimum_edge_ties:
        edges = edges[smallest_edge_idxs]
        pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[smallest_edge_idxs]
        num_edges_per_crystal = (num_nodes_per_crystal ** 2).cast(paddle.int64)
    else:
        cutoff_distances = 1e-4 + min_inter_atom_distances[map_edge_to_unique]
        keep_edge_mask = inter_atom_distances < cutoff_distances
        indices_to_keep = paddle.nonzero(keep_edge_mask).reshape([-1])

        edges = edges[indices_to_keep]
        pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[indices_to_keep]

        num_edges_per_crystal = scatter(
            src=paddle.ones([edges.shape[0]], dtype=paddle.float32),
            index=map_edge_to_crystal[indices_to_keep],
            dim_size=batch_size,
            reduce="sum",
        ).cast(paddle.int64)

    dst_idx = edges[:, 0]
    src_idx = edges[:, 1]
    return dst_idx, src_idx, pbc_frac_offsets_per_source_atom, num_edges_per_crystal


def ocp_get_pbc_distances(
    coords: paddle.Tensor,
    source_id: paddle.Tensor,
    destination_id: paddle.Tensor,
    lattice: paddle.Tensor,
    pbc_frac_offsets_per_source_node: paddle.Tensor,
    num_edges_per_crystal: paddle.Tensor,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> dict:
    neighbors = num_edges_per_crystal.cast(paddle.int64)
    lattice = paddle.repeat_interleave(lattice, neighbors, axis=0)
    offsets = (
        pbc_frac_offsets_per_source_node.cast(paddle.float32)
        .unsqueeze(1)
        .bmm(lattice.cast(paddle.float32))
        .reshape([-1, 3])
    )
    distance_vectors = coords[source_id] + offsets - coords[destination_id]
    distances = distance_vectors.norm(axis=-1)
    edge_index = paddle.stack([source_id, destination_id], axis=0)

    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out
