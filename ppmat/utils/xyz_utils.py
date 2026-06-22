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
3D geometry utilities for molecular systems.

Provides functions to compute distances, angles, and torsion (dihedral)
angles from atomic 3D coordinates, used by message-passing neural networks.
"""

import paddle


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
    # For each edge (j -> i), find all (k -> j) where k != i.
    # Vectorized approach: expand edge pairs and filter.
    num_edges = j.shape[0]
    j_expand = j.unsqueeze(0).expand([num_edges, -1])  # [E, E]
    i_candidate = i.unsqueeze(-1).expand([-1, num_edges])  # [E, E]

    # k_j_edge targets == j_i_edge sources
    valid_triplet = j_expand == i_candidate

    # k != i (exclude the reverse edge)
    not_self = paddle.arange(num_edges).unsqueeze(0).expand(
        [num_edges, -1]
    ) != paddle.arange(num_edges).unsqueeze(-1).expand([-1, num_edges])
    valid_triplet = valid_triplet & not_self

    # Get indices of valid triplets
    idx_kj, idx_ji = paddle.nonzero(valid_triplet, as_tuple=True)

    # For the angle computation we need both edges:
    # v_kj = pos[k] - pos[j] for edge k_j
    # v_ji = pos[j] - pos[i] for edge j_i
    vec_kj = vec[idx_kj]  # vectors along k->j edges
    vec_ji = vec[idx_ji]  # vectors along j->i edges

    # Compute angle using arctan2 for numerical stability
    # angle = atan2(||cross(v_kj, v_ji)||, dot(v_kj, v_ji))
    angle_cross = paddle.linalg.cross(vec_kj, vec_ji)
    angle_sin = paddle.sqrt(paddle.sum(angle_cross * angle_cross, axis=-1) + 1e-8)
    angle_cos = paddle.sum(vec_kj * vec_ji, axis=-1)
    angle = paddle.atan2(angle_sin, angle_cos)

    torsion = paddle.zeros_like(angle)
    if use_torsion:
        # Build quadruplets (l -> k -> j -> i)
        # l is the source of edge l_k such that l_k.targets are k_j.sources

        # For each triplet (idx_kj, idx_ji):
        #   - edge_idx_kj has source=k, target=j
        #   - edge_idx_ji has source=j, target=i
        # The next level: edges (l -> k) where l_k.target == k_j.source (= k)

        # k is source of edge k_j
        k_nodes = i[idx_kj]  # source of k_j edge = k

        # Build: for each quadruplet candidate, k_j_edge.source == l_k_edge.target
        k_expand = k_nodes.unsqueeze(0).expand([num_edges, -1])
        j_all = j.unsqueeze(-1).expand([-1, idx_kj.shape[0]])

        valid_quad = j_all == k_expand

        # l != k (l shouldn't be the same as the quadruplet's k)
        l_edge_idx = (
            paddle.arange(num_edges).unsqueeze(-1).expand([-1, idx_kj.shape[0]])
        )
        not_k = l_edge_idx != idx_kj.unsqueeze(0)
        valid_quad = valid_quad & not_k

        idx_lk, idx_triplet = paddle.nonzero(valid_quad, as_tuple=True)

        # Now compute torsion: for quadruple (l, k, j, i)
        # v1 = pos[l] - pos[k]  (edge l_k)
        # v2 = pos[k] - pos[j]  (edge k_j)
        # v3 = pos[j] - pos[i]  (edge j_i)
        k_idx_from_edge_lk = j[idx_lk]
        v1 = pos[k_idx_from_edge_lk] - pos[i[idx_lk]]

        # For each quadruplet indexed by idx_triplet:
        # v2 = vec_kj[idx_triplet], v3 = vec_ji[idx_triplet]
        v2 = vec_kj[idx_triplet]
        v3 = vec_ji[idx_triplet]

        # Torsion = atan2(
        #     ||v2|| * v1 . (v2 x v3),
        #     (v1 x v2) . (v2 x v3)
        # )
        v2_norm = paddle.sqrt(paddle.sum(v2 * v2, axis=-1) + 1e-8)
        v2_cross_v3 = paddle.linalg.cross(v2, v3)
        v1_dot_v2crossv3 = paddle.sum(v1 * v2_cross_v3, axis=-1)
        v1_cross_v2 = paddle.linalg.cross(v1, v2)
        v1crossv2_dot_v2crossv3 = paddle.sum(v1_cross_v2 * v2_cross_v3, axis=-1)

        torsion_angle = paddle.atan2(
            v2_norm * v1_dot_v2crossv3, v1crossv2_dot_v2crossv3
        )

        # Aggregate torsion per triplet: for each triplet, take the
        # torsion with minimum absolute value (closest neighbor).
        from ppmat.utils.scatter import scatter_min as _scatter_min

        abs_torsion = paddle.abs(torsion_angle)
        # Use scatter_min with absolute values as proxy — the per-triplet
        # torsion with the smallest |torsion| is kept via reduce="min".
        _ = _scatter_min(
            abs_torsion,
            idx_triplet,
            dim=0,
            dim_size=idx_kj.shape[0],
        )
        # Recompute keep mask via index comparison
        best_abs = paddle.full([idx_kj.shape[0]], float("inf"), dtype=abs_torsion.dtype)
        best_abs = paddle.put_along_axis(
            best_abs.unsqueeze(-1),
            idx_triplet.unsqueeze(-1),
            abs_torsion.unsqueeze(-1),
            axis=0,
            reduce="amin",
        ).squeeze(-1)
        keep = abs_torsion == best_abs[idx_triplet]
        # Scatter the kept torsion values
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

    return dist, angle, torsion, i, j, idx_kj, idx_ji
