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

import paddle

from ppmat.utils.scatter import scatter_argmin


def compute_geometry(pos, edge_index, triplet_indices):
    """Compute SphereNet distance, angle, and torsion tensors.

    Args:
        pos: Atom positions with shape [num_nodes, 3].
        edge_index: Directed edge indices with shape [2, num_edges].
        triplet_indices: Precomputed edge and triplet index tensors.

    Returns:
        Distance, angle, torsion, edge source/target indices, and triplet maps.
    """
    i, j = edge_index
    idx_kj = triplet_indices["idx_kj"]
    idx_ji = triplet_indices["idx_ji"]
    idx_lk = triplet_indices["idx_lk"]
    idx_triplet = triplet_indices["idx_triplet"]

    vec = pos[j] - pos[i]
    dist = paddle.sqrt(paddle.sum(vec * vec, axis=-1) + 1e-8)

    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = -paddle.sum(vec_kj * vec_ji, axis=-1)
    angle = paddle.atan2(angle_sin, angle_cos).detach()

    torsion = paddle.zeros_like(angle)
    if idx_lk.shape[0] > 0:
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

        torsion_indices = scatter_argmin(
            paddle.abs(torsion_angle), idx_triplet, idx_kj.shape[0]
        )
        torsion = paddle.where(
            torsion_indices >= 0,
            torsion_angle[paddle.clip(torsion_indices, min=0)],
            torsion,
        )

    return dist, angle, torsion, i, j, idx_kj, idx_ji
