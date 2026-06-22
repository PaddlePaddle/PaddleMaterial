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
Graph construction utilities for molecular systems.

Provides functions to build radius graphs (neighbour lists) from
3D coordinates, supporting both molecular and periodic (crystal) systems.
"""

import paddle


def radius_graph(pos, r, batch, loop=False, max_num_neighbors=32):
    """Build a radius graph from 3D positions.

    Computes all pairwise distances within the same molecule (determined by
    the batch assignment) and returns edges where the distance is within the
    cutoff radius.

    This is a pure-Paddle implementation (no external cluster library needed)
    using the full N x N distance matrix. For large systems, a grid-based
    approach may be more efficient.

    Args:
        pos: Tensor of shape [num_nodes, 3] — atomic coordinates.
        r: Cutoff radius.
        batch: Tensor of shape [num_nodes] — batch assignment (int64).
        loop: Whether to include self-loops. Defaults to False.
        max_num_neighbors: Maximum number of neighbors per node (not enforced
            in this simple implementation — use a grid-based method for large
            systems).

    Returns:
        edge_index: Tensor of shape [2, num_edges] — (source, target) indices.
    """
    pos = paddle.cast(pos, paddle.get_default_dtype())
    num_nodes = pos.shape[0]

    # Pairwise squared distances using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    pos_sq = paddle.sum(pos * pos, axis=-1)  # [N]
    pos_sq_expand = pos_sq.unsqueeze(0).expand([num_nodes, -1])  # [N, N]
    pos_dot = paddle.mm(pos, pos.transpose([1, 0]))  # [N, N]
    dist_sq = pos_sq_expand + pos_sq_expand.transpose([1, 0]) - 2.0 * pos_dot

    # Mask: within cutoff
    mask = dist_sq <= r * r

    # Mask: same molecule (batch)
    batch_x = batch.unsqueeze(0).expand([num_nodes, -1])  # [N, N]
    batch_y = batch.unsqueeze(-1).expand([-1, num_nodes])  # [N, N]
    same_batch = batch_x == batch_y
    mask = mask & same_batch

    if not loop:
        # Mask out self-loops (float eye cast to bool for GPU compat)
        diag_mask = paddle.eye(num_nodes, dtype=paddle.get_default_dtype()).cast(
            paddle.bool
        )
        mask = mask & ~diag_mask

    # Get edge indices
    edge_index = paddle.nonzero(mask, as_tuple=False).transpose([1, 0])

    return edge_index
