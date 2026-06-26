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
"""Standalone radius graph builder for molecular 3D coordinates.

Moved from the ``RadiusGraph`` class in ``graph_converter.py`` to a pure
function under ``ppmat.models.common``, as requested by reviewer: the graph
should be built during dataset preprocessing and passed into the model via
batch, not computed inside the model forward pass.
"""

import paddle


def radius_graph(pos, batch, cutoff, loop=False):
    """Build edge indices for a batch of molecules within a cutoff radius.

    Processes each molecule independently to avoid O(N²) memory on the full
    concatenated batch. For each molecule, builds a local N_mol × N_mol
    distance matrix, then remaps edge indices to global positions.

    Args:
        pos: Tensor [num_nodes, 3] — 3D coordinates.
        batch: Tensor [num_nodes] — batch assignment (integers per graph).
        cutoff: Neighbor cutoff distance in Ångström.
        loop: Whether to include self-loops. Default ``False``.

    Returns:
        edge_index: Tensor [2, num_edges] — (source, target) edge indices.
    """
    pos = paddle.cast(pos, paddle.get_default_dtype())

    if pos.shape[0] == 0:
        return paddle.zeros([2, 0], dtype=paddle.int64)

    unique_batches = paddle.unique(batch)
    all_edges = []

    for b in unique_batches:
        mol_mask = batch == b
        global_ids = paddle.nonzero(mol_mask, as_tuple=False).squeeze(-1)
        mol_n = global_ids.shape[0]
        mol_pos = pos[global_ids]

        # N_mol x N_mol squared distance matrix
        pos_sq = paddle.sum(mol_pos * mol_pos, axis=-1)
        pos_sq_expand = pos_sq.unsqueeze(0).expand([mol_n, -1])
        pos_dot = paddle.mm(mol_pos, mol_pos.transpose([1, 0]))
        dist_sq = pos_sq_expand + pos_sq_expand.transpose([1, 0]) - 2.0 * pos_dot

        mask = dist_sq <= cutoff * cutoff
        if not loop:
            diag_mask = paddle.eye(mol_n, dtype=paddle.get_default_dtype()).cast(
                paddle.bool
            )
            mask = mask & ~diag_mask

        local_edges = paddle.nonzero(mask, as_tuple=False)
        if local_edges.shape[0] > 0:
            flat_global = paddle.gather(global_ids, local_edges.reshape([-1]))
            all_edges.append(flat_global.reshape([-1, 2]).transpose([1, 0]))

    if len(all_edges) == 0:
        return paddle.zeros([2, 0], dtype=paddle.int64)

    return paddle.concat(all_edges, axis=1)
