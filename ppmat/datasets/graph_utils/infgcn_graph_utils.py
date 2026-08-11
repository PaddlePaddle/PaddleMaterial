# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle


def _to_np(array):
    """Return a numpy view of a paddle tensor or pass through an ndarray."""
    if hasattr(array, "numpy"):
        return array.numpy()
    return np.asarray(array)


def _concat_edges(src_list, dst_list, dist_list):
    """Flatten per-graph edge lists into global edge arrays."""
    if not src_list:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    return (
        np.concatenate(src_list).astype(np.int64),
        np.concatenate(dst_list).astype(np.int64),
        np.concatenate(dist_list).astype(np.float64),
    )


def _cap_edges_per_destination(src, dst, distances, max_num_neighbors):
    """Deterministically retain the nearest ``max_num_neighbors`` edges per destination.

    Edges are grouped by ``dst``; within each group the shortest edge wins, ties are
    broken by ``src`` index. The result is therefore reproducible and independent of
    input ordering or chunk size. ``distances`` may be squared Euclidean lengths;
    only their relative order matters.

    Args:
        src, dst, distances: 1D arrays of equal length describing one edge each.
        max_num_neighbors: per-destination cap; ``None`` or a value >= 1_000_000
            disables truncation (matches the historical sentinel).

    Returns:
        np.ndarray of bool, the keep mask aligned with the inputs.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    n = src.shape[0]
    if n == 0 or max_num_neighbors is None or max_num_neighbors >= 1000000:
        return np.ones(n, dtype=bool)

    # lexsort keys are reversed: the last key is primary, so sort by
    # (dst, distance, src) -> deterministic nearest-K per destination.
    order = np.lexsort((src, distances, dst))
    dst_sorted = dst[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(dst_sorted[1:], dst_sorted[:-1], out=new_group[1:])
    group_start = np.where(new_group)[0]
    group_id = np.searchsorted(group_start, np.arange(n), side="right") - 1
    rank = np.arange(n) - group_start[group_id]
    keep_sorted = rank < int(max_num_neighbors)
    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def radius_graph(
    x: paddle.Tensor,
    r: float,
    batch: paddle.Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> paddle.Tensor:
    """Build a radius graph, capping incoming edges per node deterministically.

    Returns ``[src, dst]`` (``[2, num_edges]``), where every node keeps at most
    ``max_num_neighbors`` nearest incoming neighbours.
    """
    x_np = _to_np(x)
    if batch is None:
        batch = paddle.zeros((x_np.shape[0],), dtype="int64")
    b_np = _to_np(batch)

    if x_np.shape[0] <= 1000:
        src, dst, dist = radius_graph_simple(x_np, r, b_np, loop)
    else:
        src, dst, dist = radius_graph_grid(x_np, r, b_np, loop)

    keep = _cap_edges_per_destination(src, dst, dist, max_num_neighbors)
    src, dst = src[keep], dst[keep]
    if src.shape[0] == 0:
        return paddle.zeros([2, 0], dtype="int64")
    return paddle.stack(
        [paddle.to_tensor(src, dtype="int64"), paddle.to_tensor(dst, dtype="int64")],
        axis=0,
    )


def radius(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: paddle.Tensor | None = None,
    batch_y: paddle.Tensor | None = None,
    max_num_neighbors: int = 32,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Build a bipartite radius graph from ``x`` (sources) onto ``y`` (destinations).

    Returns ``(dst, src)`` aligned with the historical convention: the first tensor
    indexes ``y`` (e.g. grid points) and the second indexes ``x`` (e.g. atoms).
    Each destination keeps at most ``max_num_neighbors`` nearest sources.
    """
    x_np = _to_np(x)
    y_np = _to_np(y)
    if batch_x is None:
        batch_x = paddle.zeros((x_np.shape[0],), dtype="int64")
    if batch_y is None:
        batch_y = paddle.zeros((y_np.shape[0],), dtype="int64")
    bx_np = _to_np(batch_x)
    by_np = _to_np(batch_y)

    atoms_grids_scenario = x_np.shape[0] < 1000 and y_np.shape[0] > 1000
    if atoms_grids_scenario:
        src, dst, dist = radius_atoms_to_grids(x_np, y_np, r, bx_np, by_np)
    elif x_np.shape[0] > 1000 or y_np.shape[0] > 1000:
        src, dst, dist = radius_grid(x_np, y_np, r, bx_np, by_np)
    else:
        src, dst, dist = radius_simple(x_np, y_np, r, bx_np, by_np)

    keep = _cap_edges_per_destination(src, dst, dist, max_num_neighbors)
    src, dst = src[keep], dst[keep]
    return paddle.to_tensor(dst, dtype="int64"), paddle.to_tensor(src, dtype="int64")


def radius_graph_simple(
    x_np: np.ndarray,
    r: float,
    batch_np: np.ndarray,
    loop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense radius graph for small node counts.

    Returns ``(src, dst, dist)`` global edge arrays for every within-cutoff pair.
    """
    batch_size = int(batch_np.max()) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mask = batch_np == b
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        sx = x_np[mask]
        d2 = ((sx[:, None, :] - sx[None, :, :]) ** 2).sum(-1)
        adj = d2 <= r2
        if not loop:
            np.fill_diagonal(adj, False)
        row, col = np.where(adj)
        if row.size:
            src_list.append(idx[row])
            dst_list.append(idx[col])
            dist_list.append(d2[row, col])
    return _concat_edges(src_list, dst_list, dist_list)


def radius_simple(
    x_np: np.ndarray,
    y_np: np.ndarray,
    r: float,
    batch_x_np: np.ndarray,
    batch_y_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense bipartite radius graph for small node counts.

    ``x`` are sources, ``y`` are destinations. Returns ``(src, dst, dist)`` global
    edge arrays for every within-cutoff pair.
    """
    batch_size = max(int(batch_x_np.max()), int(batch_y_np.max())) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mx = batch_x_np == b
        my = batch_y_np == b
        ix = np.where(mx)[0]
        iy = np.where(my)[0]
        if ix.size == 0 or iy.size == 0:
            continue
        sx = x_np[mx]
        sy = y_np[my]
        d2 = ((sx[:, None, :] - sy[None, :, :]) ** 2).sum(-1)
        row, col = np.where(d2 <= r2)
        if row.size:
            src_list.append(ix[row])
            dst_list.append(iy[col])
            dist_list.append(d2[row, col])
    return _concat_edges(src_list, dst_list, dist_list)


def radius_atoms_to_grids(
    x_np: np.ndarray,
    y_np: np.ndarray,
    r: float,
    batch_x_np: np.ndarray,
    batch_y_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-hashed bipartite radius graph for many destinations.

    ``x`` are atoms (few), ``y`` are grid points (many). Returns ``(src, dst, dist)``
    global edge arrays for every within-cutoff pair; no truncation.
    """
    batch_size = max(int(batch_x_np.max()), int(batch_y_np.max())) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    grid_chunk_size = 10000
    for b in range(batch_size):
        atom_mask = batch_x_np == b
        grid_mask = batch_y_np == b
        if not atom_mask.any() or not grid_mask.any():
            continue
        atoms = x_np[atom_mask]
        atom_indices = np.where(atom_mask)[0]
        grid_indices = np.where(grid_mask)[0]
        grids = y_np[grid_mask]

        min_coords = atoms.min(axis=0) - r
        cell_size = r
        atom_grid: dict[tuple[int, int, int], list[int]] = {}
        for i, atom_pos in enumerate(atoms):
            cell_idx = tuple(np.floor((atom_pos - min_coords) / cell_size).astype(int))
            atom_grid.setdefault(cell_idx, []).append(i)

        for start in range(0, grids.shape[0], grid_chunk_size):
            end = min(start + grid_chunk_size, grids.shape[0])
            for gi, grid_pos in enumerate(grids[start:end]):
                cell_idx = tuple(
                    np.floor((grid_pos - min_coords) / cell_size).astype(int)
                )
                nearby: list[int] = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            nearby.extend(
                                atom_grid.get(
                                    (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz),
                                    [],
                                )
                            )
                if not nearby:
                    continue
                nearby_arr = np.asarray(nearby)
                dists = ((atoms[nearby_arr] - grid_pos) ** 2).sum(axis=1)
                valid = dists <= r2
                if valid.any():
                    valid_atoms = nearby_arr[valid]
                    src_list.append(atom_indices[valid_atoms])
                    dst_list.append(
                        np.full(
                            valid_atoms.shape, grid_indices[start + gi], dtype=np.int64
                        )
                    )
                    dist_list.append(dists[valid])
    return _concat_edges(src_list, dst_list, dist_list)


def radius_graph_grid(
    x_np: np.ndarray,
    r: float,
    batch_np: np.ndarray,
    loop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-hashed radius graph for many nodes.

    Returns ``(src, dst, dist)`` global edge arrays for every within-cutoff pair.
    """
    batch_size = int(batch_np.max()) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mask = batch_np == b
        if not mask.any():
            continue
        sx = x_np[mask]
        idx = np.where(mask)[0]
        n = sx.shape[0]

        min_coords = sx.min(axis=0) - r
        cell_size = r
        cell_of: dict[tuple[int, int, int], list[int]] = {}
        for j in range(n):
            cell_idx = tuple(np.floor((sx[j] - min_coords) / cell_size).astype(int))
            cell_of.setdefault(cell_idx, []).append(j)

        for i in range(n):
            point = sx[i]
            cell_idx = tuple(np.floor((point - min_coords) / cell_size).astype(int))
            nearby: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(
                            cell_of.get(
                                (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz),
                                [],
                            )
                        )
            if not nearby:
                continue
            nearby_arr = np.asarray(nearby)
            dists = ((sx[nearby_arr] - point) ** 2).sum(axis=1)
            valid = dists <= r2
            if not loop:
                valid = valid & (nearby_arr != i)
            if valid.any():
                valid_j = nearby_arr[valid]
                src_list.append(np.full(valid_j.shape, idx[i], dtype=np.int64))
                dst_list.append(idx[valid_j])
                dist_list.append(dists[valid])
    return _concat_edges(src_list, dst_list, dist_list)


def radius_grid(
    x_np: np.ndarray,
    y_np: np.ndarray,
    r: float,
    batch_x_np: np.ndarray,
    batch_y_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-hashed bipartite radius graph for large inputs.

    ``x`` are sources, ``y`` are destinations. Returns ``(src, dst, dist)`` global
    edge arrays for every within-cutoff pair; no truncation.
    """
    batch_size = max(int(batch_x_np.max()), int(batch_y_np.max())) + 1
    r2 = r * r
    src_list, dst_list, dist_list = [], [], []
    for b in range(batch_size):
        mx = batch_x_np == b
        my = batch_y_np == b
        if not mx.any() or not my.any():
            continue
        sx = x_np[mx]
        sy = y_np[my]
        ix = np.where(mx)[0]
        iy = np.where(my)[0]
        nx, ny = sx.shape[0], sy.shape[0]

        min_coords = np.vstack([sx.min(axis=0), sy.min(axis=0)]).min(axis=0) - r
        cell_size = r
        cell_of: dict[tuple[int, int, int], list[int]] = {}
        for j in range(ny):
            cell_idx = tuple(np.floor((sy[j] - min_coords) / cell_size).astype(int))
            cell_of.setdefault(cell_idx, []).append(j)

        for i in range(nx):
            point = sx[i]
            cell_idx = tuple(np.floor((point - min_coords) / cell_size).astype(int))
            nearby: list[int] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(
                            cell_of.get(
                                (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz),
                                [],
                            )
                        )
            if not nearby:
                continue
            nearby_arr = np.asarray(nearby)
            dists = ((sy[nearby_arr] - point) ** 2).sum(axis=1)
            valid = dists <= r2
            if valid.any():
                valid_j = nearby_arr[valid]
                src_list.append(np.full(valid_j.shape, ix[i], dtype=np.int64))
                dst_list.append(iy[valid_j])
                dist_list.append(dists[valid])
    return _concat_edges(src_list, dst_list, dist_list)
