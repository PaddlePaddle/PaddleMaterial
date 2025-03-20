import sys

import numpy as np
import paddle

import copy

# Temporary use of alternative methods, no longer using paddle_stcatter
# https://github.com/PFCCLab/paddle_scatter/tree/main
# from paddle_scatter import segment_coo
# from paddle_scatter import segment_csr

from paddle_utils import *  # noqa
from paddle_utils import dim2perm

"""
Code derived from the OCP codebase:
https://github.com/Open-Catalyst-Project/ocp

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in
https://github.com/Open-Catalyst-Project/ocp/blob/main/LICENSE.md.
"""


def get_pbc_distances(
    pos: paddle.Tensor,
    edge_index: paddle.Tensor,
    cell: paddle.Tensor,
    cell_offsets: paddle.Tensor,
    neighbors: paddle.Tensor,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> dict:
    row, col = edge_index
    distance_vectors = pos[row] - pos[col]
    neighbors = neighbors.to(cell.place)
    cell = paddle.repeat_interleave(x=cell, repeats=neighbors, axis=0)
    offsets = (
        cell_offsets.astype(dtype="float32")
        .view(-1, 1, 3)
        .bmm(y=cell.astype(dtype="float32"))
        .view(-1, 3)
    )
    distance_vectors += offsets
    distances = distance_vectors.norm(axis=-1)
    nonzero_idx = paddle.arange(end=len(distances))[distances > 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]
    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]
    return out


def radius_graph_pbc(
    pos: paddle.Tensor,
    pbc: paddle.Tensor | None,
    natoms: paddle.Tensor,
    cell: paddle.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = sys.maxsize,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Function computing the graph in periodic boundary conditions on a (batched) set
    of positions and cells.

    This function is copied from
    https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py,
    commit 480eb9279ec4a5885981f1ee588c99dcb38838b5

    Args:
        pos (LongTensor): Atomic positions in cartesian coordinates
            :obj:`[n, 3]`
        pbc (BoolTensor): indicates periodic boundary conditions per structure.
            :obj:`[n_structures, 3]`
        natoms (IntTensor): number of atoms per structure. Has shape
            :obj:`[n_structures]`
        cell (Tensor): atomic cell. Has shape
            :obj:`[n_structures, 3, 3]`
        radius (float): cutoff radius distance
        max_num_neighbors_threshold (int): Maximum number of neighbours to consider.

    Returns:
        edge_index (IntTensor): index of atoms in edges. Has shape
            :obj:`[n_edges, 2]`
        cell_offsets (IntTensor): cell displacement w.r.t. their original position of
            atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        num_neighbors_image (IntTensor): Number of neighbours per cell image.
            :obj:`[n_structures]`
        offsets (LongTensor): cartesian displacement w.r.t. their original position of
            atoms in edges. Has shape
            :obj:`[n_edges, 3, 3]`
        atom_distance (LongTensor): edge length. Has shape
            :obj:`[n_edges]`
    """
    batch_size = len(natoms)
    pbc_ = [False, False, False]
    if pbc is not None:
        pbc = paddle.atleast_2d(pbc)
        for i in range(3):
            if not paddle.any(x=pbc[:, i]).item():
                pbc_[i] = False
            elif paddle.all(x=pbc[:, i]).item():
                pbc_[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC "
                    "configurations. This is not currently supported."
                )
    natoms_squared = (natoms**2).astype(dtype="int64")
    index_offset = paddle.cumsum(x=natoms, axis=0) - natoms
    index_offset_expand = paddle.repeat_interleave(x=index_offset, repeats=natoms_squared)  # noqa
    natoms_expand = paddle.repeat_interleave(x=natoms, repeats=natoms_squared)
    num_atom_pairs = paddle.sum(x=natoms_squared)
    index_squared_offset = paddle.cumsum(x=natoms_squared, axis=0) - natoms_squared
    index_squared_offset = paddle.repeat_interleave(
        x=index_squared_offset, repeats=natoms_squared
    )  # noqa
    atom_count_squared = paddle.arange(end=num_atom_pairs) - index_squared_offset

    index1_tmp = paddle.divide(x=atom_count_squared, y=paddle.to_tensor(natoms_expand))
    index1 = paddle.floor(index1_tmp).astype("int64") + index_offset_expand
    index2 = atom_count_squared % natoms_expand + index_offset_expand
    pos1 = paddle.index_select(x=pos, axis=0, index=index1)
    pos2 = paddle.index_select(x=pos, axis=0, index=index2)
    cross_a2a3 = paddle.cross(x=cell[:, 1], y=cell[:, 2], axis=-1)
    cell_vol = paddle.sum(x=cell[:, 0] * cross_a2a3, axis=-1, keepdim=True)
    if pbc_[0]:
        inv_min_dist_a1 = paddle.linalg.norm(x=cross_a2a3 / cell_vol, p=2, axis=-1)
        rep_a1 = paddle.ceil(x=radius * inv_min_dist_a1)
    else:
        rep_a1 = paddle.zeros(shape=[1], dtype=cell.dtype)
    if pbc_[1]:
        cross_a3a1 = paddle.cross(x=cell[:, 2], y=cell[:, 0], axis=-1)
        inv_min_dist_a2 = paddle.linalg.norm(x=cross_a3a1 / cell_vol, p=2, axis=-1)
        rep_a2 = paddle.ceil(x=radius * inv_min_dist_a2)
    else:
        rep_a2 = paddle.zeros(shape=[1], dtype=cell.dtype)
    if pbc_[2]:
        cross_a1a2 = paddle.cross(x=cell[:, 0], y=cell[:, 1], axis=-1)
        inv_min_dist_a3 = paddle.linalg.norm(x=cross_a1a2 / cell_vol, p=2, axis=-1)
        rep_a3 = paddle.ceil(x=radius * inv_min_dist_a3)
    else:
        rep_a3 = paddle.zeros(shape=[1], dtype=cell.dtype)
    max_rep = [
        min(int(rep_a1.max()), max_cell_images_per_dim),
        min(int(rep_a2.max()), max_cell_images_per_dim),
        min(int(rep_a3.max()), max_cell_images_per_dim),
    ]
    cells_per_dim = [
        paddle.arange(start=-rep, end=rep + 1, dtype="float32") for rep in max_rep
    ]  # noqa
    cell_offsets = paddle.cartesian_prod(x=cells_per_dim)
    num_cells = len(cell_offsets)
    cell_offsets_per_atom = cell_offsets.view(1, num_cells, 3).tile(
        repeat_times=[len(index2), 1, 1]
    )
    cell_offsets = paddle.transpose(x=cell_offsets, perm=dim2perm(cell_offsets.ndim, 0, 1))  # noqa
    cell_offsets_batch = cell_offsets.view(1, 3, num_cells).expand(
        shape=[batch_size, -1, -1]
    )  # noqa
    data_cell = paddle.transpose(x=cell, perm=dim2perm(cell.ndim, 1, 2))
    pbc_offsets = paddle.bmm(x=data_cell, y=cell_offsets_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=natoms_squared, axis=0
    )  # noqa
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    index1 = index1.view(-1, 1).tile(repeat_times=[1, num_cells]).view(-1)
    index2 = index2.view(-1, 1).tile(repeat_times=[1, num_cells]).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_squared = paddle.sum(x=(pos1 - pos2) ** 2, axis=1)
    atom_distance_squared = atom_distance_squared.view(-1)
    mask_within_radius = paddle.less_equal(
        x=atom_distance_squared, y=paddle.to_tensor(radius * radius)
    )
    mask_not_same = paddle.greater_than(x=atom_distance_squared, y=paddle.to_tensor(0.0001))  # noqa
    mask = paddle.logical_and(x=mask_within_radius, y=mask_not_same)
    index1 = paddle.masked_select(x=index1, mask=mask)
    index2 = paddle.masked_select(x=index2, mask=mask)
    cell_offsets = paddle.masked_select(
        x=cell_offsets_per_atom.view(-1, 3), mask=mask.view(-1, 1).expand(shape=[-1, 3])
    )
    cell_offsets = cell_offsets.view(-1, 3)
    atom_distance_squared = paddle.masked_select(x=atom_distance_squared, mask=mask)
    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=natoms,
        index=index1,
        atom_distance_squared=atom_distance_squared,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )
    if not paddle.all(x=mask_num_neighbors):
        index1 = paddle.masked_select(x=index1, mask=mask_num_neighbors)
        index2 = paddle.masked_select(x=index2, mask=mask_num_neighbors)
        atom_distance_squared = paddle.masked_select(
            x=atom_distance_squared, mask=mask_num_neighbors
        )
        cell_offsets = paddle.masked_select(
            x=cell_offsets.view(-1, 3),
            mask=mask_num_neighbors.view(-1, 1).expand(shape=[-1, 3]),
        )
        cell_offsets = cell_offsets.view(-1, 3)
    edge_index = paddle.stack(x=(index2, index1))
    cell_repeated = paddle.repeat_interleave(x=cell, repeats=num_neighbors_image, axis=0)  # noqa
    offsets = (
        -cell_offsets.astype(dtype="float32")
        .view(-1, 1, 3)
        .bmm(y=cell_repeated.astype(dtype="float32"))
        .view(-1, 3)
    )
    return (
        edge_index,
        cell_offsets,
        num_neighbors_image,
        offsets,
        paddle.sqrt(x=atom_distance_squared),
    )


def get_max_neighbors_mask(
    natoms: paddle.Tensor,
    index: paddle.Tensor,
    atom_distance_squared: paddle.Tensor,
    max_num_neighbors_threshold: int,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.place
    num_atoms = natoms.sum()

    # Temporary use of alternative methods, no longer using paddle_stcatter
    # https://github.com/PFCCLab/paddle_scatter/tree/main
    #===================================================================================
    # ones = paddle.ones(shape=[1], dtype=index.dtype).expand_as(y=index)
    # num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    
    num_neighbors = paddle.zeros(shape=num_atoms)
    num_neighbors.index_add_(axis=0, index=index, value=paddle.ones(shape=len(index)))
    num_neighbors = num_neighbors.astype(dtype="int64")
    #===================================================================================

    # Temporary use of alternative methods, no longer using paddle_stcatter
    # https://github.com/PFCCLab/paddle_scatter/tree/main
    #===================================================================================
    # max_num_neighbors = num_neighbors.max()
    # num_neighbors_thresholded = num_neighbors.clip(max=max_num_neighbors_threshold)
    # image_indptr = paddle.zeros(shape=tuple(natoms.shape)[0] + 1, dtype="int64")
    # image_indptr[1:] = paddle.cumsum(x=natoms, axis=0)
    # num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    max_num_neighbors = paddle.max(x=num_neighbors).astype(dtype="int64")
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = paddle.zeros(shape=num_atoms + 1).astype(dtype="int64")
    _natoms = paddle.zeros(shape=tuple(natoms.shape)[0] + 1).astype(dtype="int64")
    _num_neighbors[1:] = paddle.cumsum(x=_max_neighbors, axis=0)
    _natoms[1:] = paddle.cumsum(x=natoms, axis=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    #===================================================================================

    if max_num_neighbors <= max_num_neighbors_threshold or max_num_neighbors_threshold <= 0:  # noqa
        mask_num_neighbors = paddle.to_tensor(
            data=[True], dtype=bool, place=device
        ).expand_as(  # noqa
            y=index
        )  # noqa
        return mask_num_neighbors, num_neighbors_image
    distance_sort = paddle.full(shape=[num_atoms * max_num_neighbors], fill_value=np.inf)  # noqa
    index_neighbor_offset = paddle.cumsum(x=num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        x=index_neighbor_offset, repeats=num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + paddle.arange(end=len(index))
        - index_neighbor_offset_expand  # noqa
    )
    distance_sort.scatter_(index_sort_map, atom_distance_squared)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
    distance_sort, index_sort = paddle.sort(x=distance_sort, axis=1), paddle.argsort(
        x=distance_sort, axis=1
    )
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        shape=[-1, max_num_neighbors_threshold]
    )
    mask_finite = paddle.isfinite(x=distance_sort)
    index_sort = paddle.masked_select(x=index_sort, mask=mask_finite)
    mask_num_neighbors = paddle.zeros(shape=len(index), dtype=bool)
    mask_num_neighbors.index_fill_(axis=0, index=index_sort, value=True)
    return mask_num_neighbors, num_neighbors_image
