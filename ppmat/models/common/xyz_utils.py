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

SphereNet-specific — computes distance, angle, and torsion from 3D
atomic coordinates for spherical message passing.
"""

import numpy as np
import paddle


def _select_min_abs_torsion(torsion_angle, idx_triplet, num_triplets):
    """For each triplet, select the quadruplet candidate with smallest |torsion|.

    Each triplet may have multiple quadruplet candidates (different l nodes).
    The canonical torsion for a triplet is the one with the smallest absolute
    value (closest to 0°).

    Implements a grouped-argmin pattern: sorts by (triplet_id * scale + |val|),
    then keeps the first element per unique triplet_id.
    """
    out = paddle.zeros([num_triplets], dtype=torsion_angle.dtype)
    if idx_triplet.shape[0] == 0:
        return out
    # Group quadruplets by triplet index, then within each group sort by |torsion|
    abs_a = paddle.abs(torsion_angle)
    inv = paddle.unique(idx_triplet, return_inverse=True)[1]
    scale = abs_a.max() + 1.0
    order = paddle.argsort(inv.to(abs_a.dtype) * scale + abs_a)
    _, first = paddle.unique(inv[order], return_index=True)
    keep = order[first]
    sel = inv[keep]
    out = paddle.scatter(out, sel, torsion_angle[keep], overwrite=True)
    return out


def compute_triplet_indices(edge_index, num_nodes):
    """Precompute triplet/quadruplet selection indices from graph structure.

    Triplet (k -> j -> i) and quadruplet (l -> k -> j -> i) indices depend
    only on the *connectivity* (edge_index, num_nodes), not on atomic
    positions.  Caching them per molecule avoids the O(∑ deg×deg) Python
    loop on every forward pass.

    Args:
        edge_index: Tensor [2, E] — (source, target) pairs.
        num_nodes: Number of atoms.

    Returns:
        dict with keys:
            i, j: source/target node indices  [E].
            idx_kj, idx_ji: triplet → edge / triplet → central-edge maps.
            idx_lk, idx_triplet: quadruplet → edge / quadruplet → triplet maps.
    """
    i, j = edge_index[0], edge_index[1]

    # Convert to numpy for fast per-element access (avoids GPU sync per call).
    i_np = i.numpy()
    j_np = j.numpy()
    num_edges = len(j_np)

    in_idx = [[] for _ in range(num_nodes)]
    out_idx = [[] for _ in range(num_nodes)]
    for e in range(num_edges):
        in_idx[j_np[e]].append(e)
        out_idx[i_np[e]].append(e)

    # Triplets — exclude loopback (k == i) where the path i->j->i
    # forms an in-and-out through the same neighbor with no physical angle.
    idx_kj_list, idx_ji_list = [], []
    for n in range(num_nodes):
        kj_list = in_idx[n]
        ji_list = out_idx[n]
        if kj_list and ji_list:
            for kj_e in kj_list:
                k_atom = i_np[kj_e]
                for ji_e in ji_list:
                    i_atom = j_np[ji_e]
                    if k_atom != i_atom:
                        idx_kj_list.append(kj_e)
                        idx_ji_list.append(ji_e)

    idx_kj = paddle.to_tensor(idx_kj_list, dtype='int64')
    idx_ji = paddle.to_tensor(idx_ji_list, dtype='int64')

    # Quadruplets: use numpy to avoid GPU sync in loop.
    k_np = i_np[idx_kj_list] if idx_kj_list else np.array([], dtype=np.int64)
    in_idx_edges = [[] for _ in range(num_nodes)]
    for e in range(num_edges):
        in_idx_edges[j_np[e]].append(e)

    idx_lk_list, idx_triplet_list = [], []
    for t in range(len(k_np)):
        lk_list = in_idx_edges[k_np[t]]
        if lk_list:
            idx_lk_list.extend(lk_list)
            idx_triplet_list.extend([t] * len(lk_list))

    idx_lk = paddle.to_tensor(idx_lk_list, dtype='int64') if idx_lk_list else paddle.empty([0], dtype='int64')
    idx_triplet = paddle.to_tensor(idx_triplet_list, dtype='int64') if idx_triplet_list else paddle.empty([0], dtype='int64')

    return {
        'i': i,
        'j': j,
        'idx_kj': idx_kj,
        'idx_ji': idx_ji,
        'idx_lk': idx_lk,
        'idx_triplet': idx_triplet,
    }


def xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True):
    """Compute distance, angle, and torsion from 3D positions.

    Args:
        pos: Tensor of shape [num_nodes, 3].
        edge_index: Tensor of shape [2, num_edges].
        num_nodes: Number of atoms.
        use_torsion: Whether to compute torsion angles. Defaults to True.

    Returns:
        dist: Edge distances  [num_edges].
        angle: Bond angles  [num_triplets].
        torsion: Torsion angles  [num_triplets].
        i: Source node indices  [num_edges].
        j: Target node indices  [num_edges].
        idx_kj: Triplet → edge index map.
        idx_ji: Triplet → central edge map.
    """
    indices = compute_triplet_indices(edge_index, num_nodes)
    i, j = indices['i'], indices['j']
    idx_kj = indices['idx_kj']
    idx_ji = indices['idx_ji']
    idx_lk = indices['idx_lk']
    idx_triplet = indices['idx_triplet']

    vec = pos[j] - pos[i]
    dist = paddle.sqrt(paddle.sum(vec * vec, axis=-1) + 1e-8)

    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = -paddle.sum(vec_kj * vec_ji, axis=-1)
    angle = paddle.atan2(angle_sin, angle_cos).detach()

    torsion = paddle.zeros_like(angle)
    if use_torsion and idx_lk.shape[0] > 0:
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

        torsion = _select_min_abs_torsion(
            torsion_angle, idx_triplet, idx_kj.shape[0]
        )

    return dist, angle, torsion, i, j, idx_kj, idx_ji
