# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Graph utilities for molecular graphs using PGL (Paddle Graph Learning).

This module provides graph utilities that replace DGL functionality for PaddlePaddle.
"""

import typing

import numpy as np
import paddle
import pgl


def mean_nodes(graph: pgl.Graph, feat_name: str = "h") -> paddle.Tensor:
    """Compute mean of node features per graph in batch.

    This function replaces dgl.mean_nodes() by averaging node features
    within each graph in the batch using pgl's batch structure.

    Args:
        graph: Batched pgl.Graph object
        feat_name: Name of node feature to average (default: "h")

    Returns:
        Tensor of shape [batch_size, feat_dim] with averaged features

    Example:
        >>> batched = pgl.Graph.batch([g1, g2])
        >>> batched.node_feat["h"]  # [total_nodes, feat_dim]
        >>> graph_means = mean_nodes(batched, "h")  # [batch_size, feat_dim]
    """
    # Get node features
    node_feats = graph.node_feat[feat_name]  # [total_nodes, feat_dim]

    graph_node_id = graph.graph_node_id  # [total_nodes], values in [0, num_graph)
    num_graph = int(graph.num_graph)

    # Compute mean for each graph using segment operations
    result = segment_mean(node_feats, graph_node_id, num_graph)

    return result


def segment_sum(
    data: paddle.Tensor,
    segment_ids: typing.Union[paddle.Tensor, np.ndarray],
    num_segments: int,
) -> paddle.Tensor:
    """Sum data along segments defined by segment_ids.

    This is a utility function for message passing aggregation.

    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments).
            Can be a paddle Tensor or numpy array.
        num_segments: Number of segments

    Returns:
        Tensor of shape [num_segments, ...] with summed data per segment
    """
    original_shape = data.shape
    ndim = len(original_shape)

    # Flatten all dimensions except the first (segment dimension)
    if ndim > 1:
        feat_dim = int(np.prod(original_shape[1:]))
        data = data.reshape([original_shape[0], feat_dim])
    else:
        feat_dim = 1

    segment_ids = paddle.to_tensor(segment_ids).reshape([-1])

    # Create one-hot encoding
    one_hot = paddle.nn.functional.one_hot(segment_ids, num_segments).cast("float32")

    # Sum using matrix multiplication: [num_segments, N] @ [N, feat_dim] -> [num_segments, feat_dim]
    result = paddle.matmul(one_hot.T, data)

    # Reshape back to original dimensions (excluding the segment dimension)
    if ndim > 1:
        result = result.reshape([num_segments] + list(original_shape[1:]))

    return result


def segment_mean(
    data: paddle.Tensor,
    segment_ids: typing.Union[paddle.Tensor, np.ndarray],
    num_segments: int,
) -> paddle.Tensor:
    """Compute mean of data along segments defined by segment_ids.

    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments).
            Can be a paddle Tensor or numpy array.
        num_segments: Number of segments

    Returns:
        Tensor of shape [num_segments, ...] with mean data per segment
    """
    sums = segment_sum(data, segment_ids, num_segments)

    # Count elements per segment
    segment_ids = paddle.to_tensor(segment_ids)
    counts = paddle.zeros([num_segments], dtype=data.dtype)
    for i in range(num_segments):
        counts[i] = paddle.sum(segment_ids == i).cast(data.dtype)

    # Avoid division by zero
    counts = paddle.maximum(counts, paddle.ones_like(counts))

    result = sums / counts.unsqueeze(-1) if len(sums.shape) > 1 else sums / counts
    return result


def segment_max(
    data: paddle.Tensor, segment_ids: paddle.Tensor, num_segments: int
) -> paddle.Tensor:
    """Compute max of data along segments defined by segment_ids.

    Args:
        data: Input tensor of shape [N, ...]
        segment_ids: Segment indices of shape [N], values in [0, num_segments)
        num_segments: Number of segments

    Returns:
        Tensor of shape [num_segments, ...] with max data per segment
    """
    result = paddle.full(
        [num_segments] + list(data.shape[1:]), -float("inf"), dtype=data.dtype
    )

    for i in range(num_segments):
        mask = segment_ids == i
        if paddle.any(mask):
            result[i] = paddle.max(data[mask], axis=0)

    return result
