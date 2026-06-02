from __future__ import annotations

from typing import Optional

import paddle

from ppmat.utils.crystal import get_pbc_distances, radius_graph_pbc


def _filter_edges_by_node_partition(
    node_partition: paddle.Tensor,
    edge_index: paddle.Tensor,
    cell_offsets: paddle.Tensor,
    neighbors: paddle.Tensor,
    num_atoms: int,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    target_atoms = edge_index[1]
    node_mask = paddle.zeros([num_atoms], dtype="bool")
    node_mask[node_partition] = True
    local_edge_mask = node_mask[target_atoms]

    num_systems = neighbors.shape[0]
    edge_system_idx = paddle.repeat_interleave(
        paddle.arange(num_systems, dtype=neighbors.dtype), neighbors
    )

    edge_index = edge_index[:, local_edge_mask]
    cell_offsets = cell_offsets[local_edge_mask]
    if neighbors.shape[0] == 1:
        new_neighbors = local_edge_mask.astype(neighbors.dtype).sum().reshape([1])
        return edge_index, cell_offsets, new_neighbors

    filtered_edge_system_idx = edge_system_idx[local_edge_mask]
    new_neighbors = paddle.zeros([num_systems], dtype=neighbors.dtype)
    ones = paddle.ones(filtered_edge_system_idx.shape, dtype=neighbors.dtype)
    new_neighbors.index_add_(0, filtered_edge_system_idx, ones)
    return edge_index, cell_offsets, new_neighbors


def _radius_graph_no_pbc(
    pos: paddle.Tensor,
    natoms: paddle.Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    src_parts = []
    dst_parts = []
    neighbors = []
    start = 0
    for n_atoms_tensor in natoms:
        n_atoms = int(n_atoms_tensor)
        pos_i = pos[start : start + n_atoms]
        diff = pos_i.unsqueeze(1) - pos_i.unsqueeze(0)
        dist = paddle.linalg.norm(diff, axis=-1)
        inf = paddle.full([n_atoms], cutoff + 1.0, dtype=dist.dtype)
        dist = dist + paddle.diag(inf)

        sys_edge_count = 0
        for target in range(n_atoms):
            within_cutoff = paddle.nonzero(dist[target] <= cutoff).flatten()
            if within_cutoff.numel() == 0:
                continue
            if max_neighbors > 0 and within_cutoff.numel() > max_neighbors:
                target_dist = paddle.index_select(dist[target], within_cutoff, axis=0)
                order = paddle.argsort(target_dist)[:max_neighbors]
                within_cutoff = paddle.index_select(within_cutoff, order, axis=0)
            src_parts.append(within_cutoff + start)
            dst_parts.append(
                paddle.full([within_cutoff.shape[0]], target + start, dtype="int64")
            )
            sys_edge_count += int(within_cutoff.shape[0])

        neighbors.append(sys_edge_count)
        start += n_atoms

    if src_parts:
        src = paddle.concat(src_parts)
        dst = paddle.concat(dst_parts)
        edge_index = paddle.stack([src, dst], axis=0)
        cell_offsets = paddle.zeros([src.shape[0], 3], dtype="int64")
    else:
        edge_index = paddle.zeros([2, 0], dtype="int64")
        cell_offsets = paddle.zeros([0, 3], dtype="int64")

    neighbors_tensor = paddle.to_tensor(neighbors, dtype="int64")
    return edge_index, cell_offsets, neighbors_tensor


def generate_graph(
    data: dict,
    cutoff: float,
    max_neighbors: int,
    enforce_max_neighbors_strictly: bool,
    radius_pbc_version: int,
    pbc: paddle.Tensor,
    node_partition: Optional[paddle.Tensor] = None,
) -> dict:
    del enforce_max_neighbors_strictly

    if radius_pbc_version not in (1, 2, 3):
        raise ValueError(f"Invalid radius_pbc version {radius_pbc_version}")

    pbc_all_true = bool(paddle.all(pbc).item())
    pbc_all_false = bool(paddle.all(~pbc).item())
    if not (pbc_all_true or pbc_all_false):
        raise ValueError("Only all-True or all-False pbc is supported")

    if pbc_all_true:
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            cart_coords=data["pos"],
            lattice=data["cell"],
            num_atoms=data["natoms"],
            radius=cutoff,
            max_num_neighbors_threshold=max_neighbors,
            device=data["pos"].place,
        )
        if node_partition is not None and radius_pbc_version != 2:
            edge_index, cell_offsets, neighbors = _filter_edges_by_node_partition(
                node_partition,
                edge_index,
                cell_offsets,
                neighbors,
                num_atoms=data["pos"].shape[0],
            )
        out = get_pbc_distances(
            coords=data["pos"],
            edge_index=edge_index,
            lattice=data["cell"],
            to_jimages=cell_offsets,
            num_atoms=data["natoms"],
            num_bonds=neighbors,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_dist = out["distances"]
        offset_distances = out["offsets"]
        edge_distance_vec = out["distance_vec"]
    else:
        edge_index, cell_offsets, neighbors = _radius_graph_no_pbc(
            pos=data["pos"],
            natoms=data["natoms"],
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        if node_partition is not None:
            edge_index, cell_offsets, neighbors = _filter_edges_by_node_partition(
                node_partition,
                edge_index,
                cell_offsets,
                neighbors,
                num_atoms=data["pos"].shape[0],
            )
        edge_distance_vec = data["pos"][edge_index[0]] - data["pos"][edge_index[1]]
        edge_dist = paddle.linalg.norm(edge_distance_vec, axis=-1)
        offset_distances = paddle.zeros_like(edge_distance_vec)

    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": edge_distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": offset_distances,
        "neighbors": neighbors,
    }
