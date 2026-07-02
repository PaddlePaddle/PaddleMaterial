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

import os
import os.path as osp
import pickle

import numpy as np
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


def build_md17_graph(idx, z, pos, converter, cache_dir):
    """Build and cache graph + triplet indices for one MD17 frame.

    Used by :class:`~ppmat.datasets.md17_dataset.MD17Dataset` for parallel
    graph pre-computation.  Each worker builds a radius graph and computes
    triplet indices, then pickles the result.

    Args:
        idx: Frame index.
        z: Atomic numbers of a single molecule (``[num_atoms]``).
        pos: 3-D coordinates of the current frame (``[num_atoms, 3]``).
        converter: Graph converter instance (e.g. ``RadiusGraph``).
        cache_dir: Directory for pickle output.

    Returns:
        Frame index (for progress tracking).
    """
    from ppmat.models.common.xyz_utils import compute_triplet_indices

    batch_t = np.zeros(z.shape[0], dtype=np.int64)
    ei = converter(paddle.to_tensor(pos), paddle.to_tensor(batch_t))
    ti = compute_triplet_indices(ei, z.shape[0])
    cache_data = {
        "edge_index": ei.numpy(),
        "ti_i": ti["i"].numpy(),
        "ti_j": ti["j"].numpy(),
        "ti_idx_kj": ti["idx_kj"].numpy(),
        "ti_idx_ji": ti["idx_ji"].numpy(),
        "ti_idx_lk": ti["idx_lk"].numpy(),
        "ti_idx_triplet": ti["idx_triplet"].numpy(),
    }
    save_path = osp.join(cache_dir, f"{idx:010d}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(cache_data, f)
    return idx

