# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""
3D geometry utilities for spherical message-passing models.

Moved from ``ppmat.utils.xyz_utils`` to ``ppmat.models.common`` per
reviewer feedback — this module is model-specific (used by SphereNet),
not a general-purpose utility.
"""

import paddle
from ppmat.utils.scatter import _scatter_min


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True):
    """Compute distance, angle, and torsion from 3D positions.

    Uses per-molecule processing to avoid O(N²) memory blowup from
    the triplet / quadruplet search on batched edge indices.

    Args:
        pos: Tensor of shape [num_nodes, 3] — atomic coordinates.
        edge_index: Tensor of shape [2, num_edges] — (source, target) indices.
        num_nodes: Number of atoms.
        use_torsion: Whether to compute torsion angles. Defaults to True.

    Returns:
        dist: Edge distances  [num_edges].
        angle: Bond angles  [num_triplets] (indexed by idx_kj).
        torsion: Torsion angles  [num_triplets] (indexed by idx_kj),
            or None when use_torsion=False.
        i: Source node indices  [num_edges].
        j: Target node indices  [num_edges].
        idx_kj: Mapping from triplets back to edges.
        idx_ji: Mapping from triplets back to the central edge.
    """
    pos = paddle.cast(pos, paddle.get_default_dtype())

    i, j = edge_index[0], edge_index[1]

    # Distance vectors and distances
    vec = pos[j] - pos[i]
    dist = paddle.sqrt(paddle.sum(vec * vec, axis=-1) + 1e-8)

    # --- Build triplets (k -> j -> i) ---
    # Process per-molecule to avoid O(N²) memory.
    # Group edges by their source node (i).
    # For each edge (k_idx -> k -> j) we find edges (ji_idx -> j -> i).
    num_edges = i.shape[0]

    # Sort edges by source node to enable efficient triplet lookup
    src_sort_idx = paddle.argsort(i, axis=-1, descending=False)
    i_sorted = paddle.gather(i, src_sort_idx)
    j_sorted = paddle.gather(j, src_sort_idx)

    # Build segment boundaries (where source node changes)
    node_changes = paddle.nonzero(i_sorted[1:] != i_sorted[:-1], as_tuple=False).flatten() + 1
    seg_starts = paddle.concat([paddle.zeros([1], dtype=paddle.int64), node_changes])
    seg_ends = paddle.concat([node_changes, paddle.full([1], num_edges, dtype=paddle.int64)])

    # For each segment (edges sharing the same source node), find triplets
    # A triplet is: edge (k->j) and another edge (j->i) where the source
    # of the second equals the target of the first.
    triplet_k = []
    triplet_ji = []

    for s, e in zip(seg_starts.numpy(), seg_ends.numpy()):
        s_idx = int(s)
        e_idx = int(e)
        # All edges with source node = current node
        target_nodes = j_sorted[s_idx:e_idx]  # the j values

        # For each target node j, find edges where source = j
        # Use a mask: i == target_node for each unique target
        unique_targets, inv_idx = paddle.unique(target_nodes, return_inverse=True)
        for t_idx in range(unique_targets.shape[0]):
            t = unique_targets[t_idx:t_idx + 1]
            # Which edges in k-segment target this node?
            k_mask = target_nodes == t
            k_in_seg = paddle.nonzero(k_mask, as_tuple=False).flatten()

            # Which edges in full edge list have source = t?
            src_mask = i == t
            ji_edges = paddle.nonzero(src_mask, as_tuple=False).flatten()

            if k_in_seg.shape[0] == 0 or ji_edges.shape[0] == 0:
                continue

            # Exclude self-triplets (where k-edge == ji-edge)
            # k_in_seg indices are within the segment, need to map to global edge idx
            k_global = src_sort_idx[s_idx:e_idx]  # global indices of edges in this segment
            k_edge_global = paddle.gather(k_global, k_in_seg)

            # Create all (k, ji) pairs
            nk = k_edge_global.shape[0]
            nji = ji_edges.shape[0]
            k_exp = k_edge_global.unsqueeze(-1).expand([nk, nji])
            ji_exp = ji_edges.unsqueeze(0).expand([nk, nji])

            # Remove self-pairs
            not_self = k_exp != ji_exp
            k_valid = paddle.masked_select(k_exp, not_self)
            ji_valid = paddle.masked_select(ji_exp, not_self)

            triplet_k.append(k_valid)
            triplet_ji.append(ji_valid)

    if len(triplet_k) == 0:
        idx_kj = paddle.zeros([0], dtype=paddle.int64)
        idx_ji = paddle.zeros([0], dtype=paddle.int64)
    else:
        idx_kj = paddle.concat(triplet_k)
        idx_ji = paddle.concat(triplet_ji)

    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = -paddle.sum(vec_kj * vec_ji, axis=-1)
    angle = paddle.atan2(angle_sin, angle_cos)

    torsion = paddle.zeros_like(angle)
    if use_torsion and idx_kj.shape[0] > 0:
        # Build quadruplets (l -> k -> j -> i) using a similar approach
        # For each triplet edge (k->j from idx_kj), find l where edge target = k
        # Build a mapping: for each node, which edges have it as source?
        k_from_triplet = i[idx_kj]  # node k (the target of the k-edge)

        # Sort triplet indices by k node for efficient lookup
        triplet_sort = paddle.argsort(k_from_triplet, axis=-1, descending=False)
        k_triplet_sorted = paddle.gather(k_from_triplet, triplet_sort)
        idx_kj_sorted = paddle.gather(idx_kj, triplet_sort)
        idx_ji_sorted = paddle.gather(idx_ji, triplet_sort)

        # Find segment boundaries for k in triplets
        t_changes = paddle.nonzero(
            k_triplet_sorted[1:] != k_triplet_sorted[:-1], as_tuple=False
        ).flatten() + 1
        t_starts = paddle.concat([paddle.zeros([1], dtype=paddle.int64), t_changes])
        t_ends = paddle.concat([t_changes, paddle.full([1], idx_kj.shape[0], dtype=paddle.int64)])

        quad_l = []
        quad_t = []

        for s, e in zip(t_starts.numpy(), t_ends.numpy()):
            s_idx = int(s)
            e_idx = int(e)
            k_node = k_triplet_sorted[s_idx:s_idx + 1]  # current k node

            # Find edges where source = k_node (these are potential l->k edges)
            lk_edges = paddle.nonzero(i == k_node, as_tuple=False).flatten()
            if lk_edges.shape[0] == 0:
                continue

            # lk_edges are l->k edges (source = k_node = current k)
            # Exclude the triplet's own k-edge (which IS in lk_edges since k_edge has source = k)
            n_t = e_idx - s_idx  # number of triplets for this k_node
            # For each triplet, create pairs with all lk_edges
            lk_exp = lk_edges.unsqueeze(0).expand([n_t, -1])
            my_k_edge = idx_kj_sorted[s_idx:e_idx]
            my_k_exp = my_k_edge.unsqueeze(-1).expand([-1, lk_edges.shape[0]])

            not_same = lk_exp != my_k_exp
            l_valid = paddle.masked_select(lk_exp, not_same)
            t_valid = paddle.masked_select(
                triplet_sort.unsqueeze(-1).expand([-1, lk_edges.shape[0]])[s_idx:e_idx],
                not_same,
            )

            quad_l.append(l_valid)
            quad_t.append(t_valid)

        if len(quad_l) > 0:
            idx_lk = paddle.concat(quad_l)
            idx_triplet = paddle.concat(quad_t)

            k_idx_from_edge_lk = j[idx_lk]
            v1 = pos[k_idx_from_edge_lk] - pos[i[idx_lk]]
            v2 = vec_kj[idx_triplet]
            v3 = vec_ji[idx_triplet]

            v2_norm = paddle.sqrt(paddle.sum(v2 * v2, axis=-1) + 1e-8)
            v2_cross_v3 = paddle.linalg.cross(v2, v3)
            v1_dot_v2crossv3 = paddle.sum(v1 * v2_cross_v3, axis=-1)
            v1_cross_v2 = paddle.linalg.cross(v1, v2)
            v1crossv2_dot_v2crossv3 = paddle.sum(v1_cross_v2 * v2_cross_v3, axis=-1)

            torsion_angle = paddle.atan2(
                v2_norm * v1_dot_v2crossv3, v1crossv2_dot_v2crossv3
            )

            abs_torsion = paddle.abs(torsion_angle)
            best_abs = paddle.full(
                [idx_kj.shape[0]], float("inf"), dtype=abs_torsion.dtype
            )
            best_abs = paddle.put_along_axis(
                best_abs.unsqueeze(-1),
                idx_triplet.unsqueeze(-1),
                abs_torsion.unsqueeze(-1),
                axis=0,
                reduce="amin",
            ).squeeze(-1)
            keep = abs_torsion == best_abs[idx_triplet]
            torsion = paddle.zeros([idx_kj.shape[0]], dtype=torsion_angle.dtype)
            kept_vals = paddle.masked_select(torsion_angle, keep)
            kept_idx = paddle.masked_select(idx_triplet, keep)
            torsion = paddle.put_along_axis(
                torsion.unsqueeze(-1),
                kept_idx.unsqueeze(-1),
                kept_vals.unsqueeze(-1),
                axis=0,
                reduce="assign",
            ).squeeze(-1)
        else:
            idx_lk = paddle.zeros([0], dtype=paddle.int64)
            idx_triplet = paddle.zeros([0], dtype=paddle.int64)

    return dist, angle, torsion, i, j, idx_kj, idx_ji
