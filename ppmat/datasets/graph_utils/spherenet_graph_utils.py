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
"""SphereNet-specific graph utilities."""

import paddle


def radius_graph(pos, batch, cutoff, loop=False):
    """Build edge indices for a batch of molecules within a cutoff radius.

    Processes each molecule independently to avoid O(N²) memory on the full
    concatenated batch. For each molecule, builds a local N_mol × N_mol
    distance matrix, then remaps edge indices to global positions.

    Args:
        pos: Tensor of shape ``(num_nodes, 3)`` with atomic coordinates.
        batch: Tensor of shape ``(num_nodes,)`` with batch indices.
        cutoff: Neighbor cutoff distance in Ångström.
        loop: Whether to include self-loops (default False).

    Returns:
        edge_index: Tensor of shape ``(2, num_edges)`` with global edge indices.
    """
    num_nodes = pos.shape[0]
    # Global → local index: graph index → first node position in the batch
    unique_batches, counts = paddle.unique(batch, return_counts=True)
    # For single-graph batches, just compute one N×N matrix.
    edge_list = []
    start = 0
    for i, g in enumerate(unique_batches):
        n = int(counts[i])
        local_pos = pos[start : start + n]
        # Pairwise squared distance
        diff = local_pos.unsqueeze(1) - local_pos.unsqueeze(0)  # [n, n, 3]
        dist_sq = paddle.sum(diff * diff, axis=-1)  # [n, n]
        # Exclude self (diagonal) unless loop=True
        mask = dist_sq < cutoff * cutoff
        if not loop:
            mask = mask & (~paddle.eye(n, dtype=paddle.bool))
        # Local indices
        src, dst = paddle.where(mask)
        # Remap to global
        edge_list.append(paddle.stack([src + start, dst + start], axis=0))
        start += n

    return paddle.concat(edge_list, axis=1)

