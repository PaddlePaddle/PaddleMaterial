# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Graph utility for converting dense adjacency to COO edge_index."""

import paddle


def dense_to_sparse(adj):
    """Convert a dense adjacency matrix to sparse (edge_index, edge_attr) COO format.

    Supports both single-graph (2D) and batched (3D) dense adjacency tensors.
    """
    if adj.ndim == 2:
        nonzero = paddle.nonzero(adj.cast("float32"))
        if nonzero.shape[0] == 0:
            return paddle.zeros([2, 0], dtype="int64"), paddle.zeros(
                [0], dtype="float32"
            )
        edge_index = nonzero.t()
        return edge_index, paddle.ones([edge_index.shape[1]], dtype="float32")

    # Batched case
    batch_size, num_nodes, _ = adj.shape
    edge_indices_list = []
    edge_attrs_list = []

    for b in range(batch_size):
        adj_b = adj[b]
        nonzero = paddle.nonzero(adj_b.cast("float32"))
        if nonzero.shape[0] == 0:
            continue
        edge_indices_list.append(nonzero.t())
        edge_attrs_list.append(paddle.ones([nonzero.shape[0]], dtype="float32"))

    if not edge_indices_list:
        return paddle.zeros([2, 0], dtype="int64"), paddle.zeros([0], dtype="float32")

    edge_index = paddle.concat(edge_indices_list, axis=1)
    edge_attr = paddle.concat(edge_attrs_list, axis=0)
    return edge_index, edge_attr
