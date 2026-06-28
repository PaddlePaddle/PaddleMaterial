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

    Given atomic positions and a neighbor edge index, computes:
        - Pairwise distances for each edge
        - Bond angles for each triplet (k -> j -> i)
        - Torsion (dihedral) angles for each quadruplet (l -> k -> j -> i)

    All operations are fully vectorised (no Python loops).

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

    # --- Build triplets (k -> j -> i) via per-node grouping ---
    # Avoids O(E²) memory by grouping edges by (target, source) per node.
    num_edges = j.shape[0]
    in_idx = [[] for _ in range(num_nodes)]
    out_idx = [[] for _ in range(num_nodes)]
    for e in range(num_edges):
        in_idx[int(j[e])].append(e)
        out_idx[int(i[e])].append(e)

    idx_kj_list, idx_ji_list = [], []
    for n in range(num_nodes):
        kj_list = in_idx[n]
        ji_list = out_idx[n]
        if kj_list and ji_list:
            n_kj, n_ji = len(kj_list), len(ji_list)
            # Vectorised cross: repeat kj n_ji times, tile ji n_kj times
            idx_kj_list.extend(kj_list * n_ji)
            idx_ji_list.extend(ji_list * n_kj)

    idx_kj = paddle.to_tensor(idx_kj_list)
    idx_ji = paddle.to_tensor(idx_ji_list)

    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = -paddle.sum(vec_kj * vec_ji, axis=-1)
    # FIXME: detach to avoid Paddle atan2 2nd-order grad NaN (create_graph=False still
    # has numerical instability for planar geometries e.g. benzene).
    angle = paddle.atan2(angle_sin, angle_cos).detach()

    torsion = paddle.zeros_like(angle)
    if use_torsion and idx_kj.shape[0] > 0:
        # Build quadruplet (l -> k -> j -> i) by grouping triplets by k-node
        k_nodes = i[idx_kj]
        in_idx_edges = [[] for _ in range(num_nodes)]
        for e in range(num_edges):
            in_idx_edges[int(j[e])].append(e)

        idx_lk_list, idx_triplet_list = [], []
        for t in range(idx_kj.shape[0]):
            k_node = int(k_nodes[t])
            lk_list = in_idx_edges[k_node]
            if lk_list:
                idx_lk_list.extend(lk_list)
                idx_triplet_list.extend([t] * len(lk_list))

        idx_lk = paddle.to_tensor(idx_lk_list)
        idx_triplet = paddle.to_tensor(idx_triplet_list)

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
        ).detach()

        # Per-triplet best-angle selection.
        if idx_triplet.shape[0] > 0:
            with paddle.no_grad():
                abs_a = paddle.abs(torsion_angle)
                inv = paddle.unique(idx_triplet, return_inverse=True)[1]
                scale = abs_a.max() + 1.0
                order = paddle.argsort(inv.to(abs_a.dtype) * scale + abs_a)
                _, first = paddle.unique(inv[order], return_index=True)
                keep = order[first]
                sel = inv[keep]
            torsion = paddle.scatter(
                paddle.zeros([idx_kj.shape[0]], dtype=torsion_angle.dtype),
                sel,
                torsion_angle[keep],
                overwrite=True,
            )

    return dist, angle, torsion, i, j, idx_kj, idx_ji
