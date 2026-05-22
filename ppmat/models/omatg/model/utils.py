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

"""Utility functions for CSPNet model.
"""

import paddle
from paddle_scatter import segment_coo, segment_csr

from ppmat.utils.crystal import OFFSET_LIST, lattice_params_to_matrix_paddle  # noqa: F811


def frac_to_cart_coords(
    frac_coords, lengths, angles, num_atoms, regularized=True, lattices=None
):
    if regularized:
        frac_coords = frac_coords % 1.0
    if lattices is None:
        lattices = lattice_params_to_matrix_paddle(lengths, angles)
    lattice_nodes = paddle.repeat_interleave(lattices, num_atoms, axis=0)
    pos = paddle.einsum("bi,bij->bj", frac_coords, lattice_nodes)

    return pos


def cart_to_frac_coords(cart_coords, lengths, angles, num_atoms, regularized=True):
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    inv_lattice = paddle.linalg.pinv(lattice)
    inv_lattice_nodes = paddle.repeat_interleave(inv_lattice, num_atoms, axis=0)
    frac_coords = paddle.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    if regularized:
        frac_coords = frac_coords % 1.0
    return frac_coords


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    """Repeat blocks of indices. From https://stackoverflow.com/questions/51154989"""
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    sizes_nonzero = sizes > 0
    if not paddle.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = paddle.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, paddle.Tensor):
            repeats = paddle.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, paddle.Tensor):
            repeat_inc = paddle.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, paddle.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = paddle.ones([1], dtype=sizes.dtype, place=sizes.place)
            zero = paddle.zeros([1], dtype=sizes.dtype, place=sizes.place)
            sizes = paddle.concat([one, sizes])
            repeats = paddle.concat([one, repeats])
            if isinstance(block_inc, paddle.Tensor):
                block_inc = paddle.concat([zero, block_inc])
            if isinstance(repeat_inc, paddle.Tensor):
                repeat_inc = paddle.concat([zero, repeat_inc])
    else:
        assert repeats >= 0
        insert_dummy = False

    r1 = paddle.repeat_interleave(
        paddle.arange(len(sizes), dtype="int64", place=sizes.place), repeats
    )

    N = (sizes * repeats).sum()

    id_ar = paddle.ones([N], dtype="int64", place=sizes.place)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, paddle.Tensor) and paddle.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = paddle.concat(
            [paddle.zeros([1], dtype=sizes.dtype, place=sizes.place), diffs.cumsum(0)]
        )
        if continuous_indexing:
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        if isinstance(block_inc, paddle.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1

        insert_val[idx] += block_inc

    if isinstance(repeat_inc, paddle.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, paddle.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    if isinstance(repeats, paddle.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    id_ar[0] += start_idx

    res = id_ar.cumsum(0)
    return res


def radius_graph_pbc(
    pos,
    lengths,
    angles,
    natoms,
    radius,
    max_num_neighbors_threshold,
    device,
    lattices=None,
):
    """Compute PBC graph edges using radius graph."""
    batch_size = len(natoms)
    atom_pos = pos

    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).cast("int64")

    index_offset = paddle.cumsum(num_atoms_per_image, axis=0) - num_atoms_per_image

    index_offset_expand = paddle.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = paddle.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    num_atom_pairs = paddle.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        paddle.cumsum(num_atoms_per_image_sqr, axis=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = paddle.arange(
        num_atom_pairs.cast("int64"), dtype="int64", place=device
    ) - index_sqr_offset.cast("int64")

    # Compute the indices for the pairs of atoms (using division and mod)
    index1 = ((atom_count_sqr // num_atoms_per_image_expand.cast("int64"))).cast(
        "int64"
    ) + index_offset_expand.cast("int64")
    index2 = (atom_count_sqr % num_atoms_per_image_expand.cast("int64")).cast(
        "int64"
    ) + index_offset_expand.cast("int64")
    # Get the positions for each atom
    pos1 = paddle.index_select(atom_pos, index1, axis=0)
    pos2 = paddle.index_select(atom_pos, index2, axis=0)

    unit_cell = paddle.to_tensor(OFFSET_LIST, dtype="float32", place=device)
    num_cells = unit_cell.shape[0]
    unit_cell_per_atom = unit_cell.reshape([1, num_cells, 3]).expand(
        [len(index2), -1, -1]
    )
    unit_cell = paddle.transpose(unit_cell, [1, 0])
    unit_cell_batch = unit_cell.reshape([1, 3, num_cells]).expand([batch_size, -1, -1])

    # lattice matrix
    lattice = lattice_params_to_matrix_paddle(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = paddle.transpose(lattice, [1, 2])
    pbc_offsets = paddle.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, axis=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.reshape([-1, 3, 1]).expand([-1, -1, num_cells])
    pos2 = pos2.reshape([-1, 3, 1]).expand([-1, -1, num_cells])
    index1 = index1.reshape([-1, 1]).expand([-1, num_cells]).reshape([-1])
    index2 = index2.reshape([-1, 1]).expand([-1, num_cells]).reshape([-1])
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = paddle.sum((pos1 - pos2) ** 2, axis=1)
    atom_distance_sqr = atom_distance_sqr.reshape([-1])

    # Remove pairs that are too far apart
    mask_within_radius = atom_distance_sqr <= radius * radius
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = atom_distance_sqr > 0.0001
    mask = paddle.logical_and(mask_within_radius, mask_not_same)
    index1 = paddle.mask_select(index1, mask)
    index2 = paddle.mask_select(index2, mask)
    unit_cell = paddle.mask_select(
        unit_cell_per_atom.reshape([-1, 3]), mask.reshape([-1, 1]).expand([-1, 3])
    )
    unit_cell = unit_cell.reshape([-1, 3])
    atom_distance_sqr = paddle.mask_select(atom_distance_sqr, mask)

    if max_num_neighbors_threshold is not None:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not paddle.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = paddle.mask_select(index1, mask_num_neighbors)
            index2 = paddle.mask_select(index2, mask_num_neighbors)
            unit_cell = paddle.mask_select(
                unit_cell.reshape([-1, 3]),
                mask_num_neighbors.reshape([-1, 1]).expand([-1, 3]),
            )
            unit_cell = unit_cell.reshape([-1, 3])
    else:
        ones = paddle.ones_like(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = paddle.zeros([natoms.shape[0] + 1], dtype="int64", place=device)
        image_indptr[1:] = paddle.cumsum(natoms, axis=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = paddle.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.place
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = paddle.ones_like(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = paddle.clip(
        num_neighbors, max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = paddle.zeros([natoms.shape[0] + 1], dtype="int64", place=device)
    image_indptr[1:] = paddle.cumsum(natoms, axis=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = paddle.tensor(
            [True], dtype="bool", place=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = paddle.full(
        [num_atoms * max_num_neighbors], float("inf"), dtype="float32", place=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = paddle.cumsum(num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + paddle.arange(len(index), dtype="int64", place=device)
        - index_neighbor_offset_expand.cast("int64")
    )
    distance_sort = paddle.scatter(
        distance_sort, index_sort_map.cast("int64"), atom_distance
    )
    distance_sort = distance_sort.reshape([num_atoms, max_num_neighbors])

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = paddle.sort(distance_sort, axis=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_real_cutoff = (
        distance_sort[:, max_num_neighbors_threshold]
        .reshape([-1, 1])
        .expand([-1, max_num_neighbors])
        + 0.01
    )

    mask_distance = distance_sort < distance_real_cutoff

    index_sort = index_sort + index_neighbor_offset.reshape([-1, 1]).expand(
        [-1, max_num_neighbors]
    )

    # Remove "unused pairs" with infinite distances
    mask_finite = paddle.isfinite(distance_sort)
    index_sort = paddle.mask_select(index_sort, mask_finite & mask_distance)

    num_neighbor_per_node = (mask_finite & mask_distance).sum(axis=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = paddle.zeros([len(index)], dtype="bool", place=device)
    mask_num_neighbors = paddle.scatter(
        mask_num_neighbors,
        index_sort.cast("int64"),
        paddle.ones_like(index_sort, dtype="bool"),
    )

    return mask_num_neighbors, num_neighbors_image
