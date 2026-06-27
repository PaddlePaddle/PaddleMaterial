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

    # --- Build triplets (k -> j -> i) ---
    num_edges = j.shape[0]
    j_expand = j.unsqueeze(0).expand([num_edges, -1])
    i_candidate = i.unsqueeze(-1).expand([-1, num_edges])

    valid_triplet = j_expand == i_candidate
    not_self = paddle.arange(num_edges).unsqueeze(0).expand(
        [num_edges, -1]
    ) != paddle.arange(num_edges).unsqueeze(-1).expand([-1, num_edges])
    valid_triplet = valid_triplet & not_self

    idx_kj, idx_ji = paddle.nonzero(valid_triplet, as_tuple=True)
    idx_kj = idx_kj.flatten()
    idx_ji = idx_ji.flatten()

    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = -paddle.sum(vec_kj * vec_ji, axis=-1)
    angle = paddle.atan2(angle_sin, angle_cos)

    torsion = paddle.zeros_like(angle)
    if use_torsion:
        k_nodes = i[idx_kj]
        k_expand = k_nodes.unsqueeze(0).expand([num_edges, -1])
        j_all = j.unsqueeze(-1).expand([-1, idx_kj.shape[0]])

        valid_quad = j_all == k_expand
        l_edge_idx = (
            paddle.arange(num_edges).unsqueeze(-1).expand([-1, idx_kj.shape[0]])
        )
        not_k = l_edge_idx != idx_kj.unsqueeze(0)
        valid_quad = valid_quad & not_k

        idx_lk, idx_triplet = paddle.nonzero(valid_quad, as_tuple=True)
        idx_lk = idx_lk.flatten()
        idx_triplet = idx_triplet.flatten()

        k_idx_from_edge_lk = j[idx_lk]
        v1 = pos[k_idx_from_edge_lk] - pos[i[idx_lk]]
        v2 = vec_kj[idx_triplet]
        v3 = vec_ji[idx_triplet]

        v2_norm = paddle.sqrt(paddle.sum(v2 * v2, axis=-1) + 1e-8)
        v2_cross_v3 = paddle.linalg.cross(v2, v3)
        v1_dot_v2crossv3 = paddle.sum(v1 * v2_cross_v3, axis=-1)
        v1_cross_v2 = paddle.linalg.cross(v1, v2)
        v1crossv2_dot_v2crossv3 = paddle.sum(v1_cross_v2 * v2_cross_v3, axis=-1)

        # Torsion angles are used as geometric features only;
        # detach to avoid Paddle put_along_axis gradient issues.
        torsion_angle = paddle.atan2(
            v2_norm * v1_dot_v2crossv3, v1crossv2_dot_v2crossv3
        )

        # Per-triplet best-angle selection without reduce="amin"
        # (Paddle put_along_axis amin gradient is buggy).
        # Use argsort + unique to pick the quadruplet with smallest
        # abs torsion for each triplet.
        if idx_triplet.shape[0] > 0:
            abs_torsion = paddle.abs(torsion_angle)
            # Sort by (triplet_id, abs_torsion) so smallest abs comes first per group
            sort_key = idx_triplet.to(abs_torsion.dtype) * (
                abs_torsion.max() + 1.0
            ) + abs_torsion
            order = paddle.argsort(sort_key)
            sorted_triplet = idx_triplet[order]
            unique_triplet, first_idx = paddle.unique(
                sorted_triplet, return_index=True
            )
            keep_idx = order[first_idx]
            torsion = paddle.zeros([idx_kj.shape[0]], dtype=torsion_angle.dtype)
            torsion = paddle.put_along_axis(
                torsion.unsqueeze(-1),
                unique_triplet.unsqueeze(-1),
                torsion_angle[keep_idx].unsqueeze(-1),
                axis=0,
                reduce="assign",
            ).squeeze(-1)
        else:
            torsion = paddle.zeros_like(angle)

    return dist, angle, torsion, i, j, idx_kj, idx_ji
