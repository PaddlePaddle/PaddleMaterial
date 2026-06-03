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
Graph neural network layers for GDI-NN.

This module provides PaddlePaddle implementations of graph neural network layers
that replace DGL's NNConv and GraphConv layers.
"""

from typing import Callable
from typing import Optional
from typing import Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.gdinn.utils.graph_utils import segment_max
from ppmat.models.gdinn.utils.graph_utils import segment_mean
from ppmat.models.gdinn.utils.graph_utils import segment_sum


def get_activation(
    activation: Optional[str] = None, get_nn: bool = False
) -> Union[Callable, nn.Layer]:
    """Get activation function or layer based on activation name.

    This function returns either a callable activation function from paddle.nn.functional
    or an nn.Layer activation module, controlled by the get_nn parameter.

    Args:
        activation: Name of the activation function. Supported values:
            - None, "relu", "ReLU", "RELU": ReLU activation
            - "elu", "ELU": ELU activation
            - "leaky_relu", "LeakyReLU", "LeakyRELU", "leakyrelu": LeakyReLU activation
            - "sigmoid", "Sigmoid", "SIGMOID": Sigmoid activation
            - "softplus", "Softplus", "SOFTPLUS": Softplus activation
            - "silu", "SiLU", "SILU": SiLU activation
            - "tanh": Tanh activation
        get_nn: If True, returns nn.Layer activation module; if False, returns functional activation

    Returns:
        If get_nn=False: Callable activation function from paddle.nn.functional
        If get_nn=True: nn.Layer activation module

    Examples:
        >>> # Get functional activation
        >>> act_func = get_activation("relu")
        >>> output = act_func(input_tensor)
        >>>
        >>> # Get layer activation for nn.Sequential
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     get_activation("relu", get_nn=True)()
        ... )
    """
    if activation is None or activation in ["relu", "ReLU", "RELU"]:
        if get_nn:
            return nn.ReLU
        return F.relu
    elif activation in ["elu", "ELU"]:
        if get_nn:
            return nn.ELU
        return F.elu
    elif activation in [
        "leaky_relu",
        "LeakyReLU",
        "LeakyRELU",
        "leakyrelu",
        "leakyReLU",
        "leakyRELU",
    ]:
        if get_nn:
            return nn.LeakyReLU
        return F.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn:
            return nn.Sigmoid
        return F.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn:
            return nn.Softplus
        return F.softplus
    elif activation in ["silu", "SiLU", "SILU"]:
        if get_nn:
            return nn.SiLU
        return F.silu
    elif activation == "tanh":
        if get_nn:
            return nn.Tanh
        return F.tanh
    else:
        # Default to ReLU
        if get_nn:
            return nn.ReLU
        return F.relu


class NNConv(nn.Layer):
    """Edge-conditioned graph convolution layer.

    This layer implements neural network convolution where edge features
    are transformed by an MLP to produce edge-specific message weights.
    This replaces dgl.nn.NNConv for PaddlePaddle.

    Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry"
    (https://arxiv.org/abs/1704.01212)

    Args:
        in_feats: Number of input node features
        out_feats: Number of output node features
        edge_func: MLP that transforms edge features to edge weights
        aggregator_type: Aggregation method ('sum', 'mean', 'max')
        bias: Whether to include bias term
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        edge_func: nn.Layer,
        aggregator_type: str = "sum",
        bias: bool = True,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_func = edge_func
        self.aggregator_type = aggregator_type

        if aggregator_type not in ["sum", "mean", "max"]:
            raise ValueError(f"Unsupported aggregator_type: {aggregator_type}")

        # Bias term
        if bias:
            self.bias = self.create_parameter(
                shape=[out_feats], default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None

    def forward(
        self, graph, feat: paddle.Tensor, edge_feat: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward pass.

        Args:
            graph: pgl.Graph object
            feat: Node features of shape [num_nodes, in_feats]
            edge_feat: Edge features of shape [num_edges, edge_feat_dim]

        Returns:
            Updated node features of shape [num_nodes, out_feats]
        """
        # Get edge information from pgl.Graph
        # pgl.Graph.edges is a tensor of shape [num_edges, 2]
        edge_tensor = graph.edges
        src = edge_tensor[:, 0]
        dst = edge_tensor[:, 1]
        num_nodes = graph.num_nodes
        num_edges = len(src)

        # Transform edge features to edge weights
        # edge_func(e_ij) returns [num_edges, in_feats * out_feats]
        edge_weights = self.edge_func(edge_feat)
        edge_weights = edge_weights.reshape([num_edges, self.in_feats, self.out_feats])

        # Get source node features
        src_feat = feat[src]  # [num_edges, in_feats]

        # Compute messages: m_ij = (edge_func(e_ij) * h_j)
        # Equivalent to einsum('ni,nio->no', src_feat, edge_weights)
        # Using bmm instead of einsum because einsum_grad doesn't support
        # higher-order gradients needed by paddle.grad(create_graph=True)
        # bmm: [num_edges, 1, in_feats] @ [num_edges, in_feats, out_feats] -> [num_edges, 1, out_feats]
        messages = paddle.bmm(src_feat.unsqueeze(1), edge_weights).squeeze(1)

        # Aggregate messages at destination nodes
        if self.aggregator_type == "sum":
            out = segment_sum(messages, dst, num_nodes)
        elif self.aggregator_type == "mean":
            out = segment_mean(messages, dst, num_nodes)
        elif self.aggregator_type == "max":
            out = segment_max(messages, dst, num_nodes)
        else:
            out = segment_sum(messages, dst, num_nodes)

        if self.bias is not None:
            out = out + self.bias

        return out


class GraphConv(nn.Layer):
    """Standard graph convolution layer (GCN).

    This replaces dgl.nn.GraphConv for PaddlePaddle.

    Args:
        in_feats: Number of input node features
        out_feats: Number of output node features
        norm: Whether to apply symmetric normalization
        bias: Whether to include bias term
        activation: Activation function (None, 'relu', 'leaky_relu', etc.)
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        norm: bool = True,
        bias: bool = True,
        activation: Optional[str] = None,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.activation = activation

        self.weight = self.create_parameter(
            shape=[in_feats, out_feats],
            default_initializer=nn.initializer.XavierUniform(),
        )

        if bias:
            self.bias = self.create_parameter(
                shape=[out_feats], default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.bias = None

    def forward(self, graph, feat: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            graph: pgl.Graph object
            feat: Node features of shape [num_nodes, in_feats]

        Returns:
            Updated node features of shape [num_nodes, out_feats]
        """
        # Get edge information from pgl.Graph
        # pgl.Graph.edges is a tensor of shape [num_edges, 2]
        edge_tensor = graph.edges
        src = edge_tensor[:, 0]
        dst = edge_tensor[:, 1]
        num_nodes = graph.num_nodes

        # Transform node features
        feat = feat @ self.weight

        # Message passing
        if self.norm:
            # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
            deg = paddle.to_tensor(graph.indegree(), dtype="float32")
            norm_coeff = paddle.pow(deg, -0.5)
            norm_coeff = paddle.where(
                paddle.isinf(norm_coeff), paddle.zeros_like(norm_coeff), norm_coeff
            )

            # Get source features and normalize them
            src_feat = feat[src]  # [num_edges, out_feats]
            src_norm = norm_coeff[src]  # [num_edges]
            src_feat = src_feat * src_norm.unsqueeze(-1)  # [num_edges, out_feats]

            # Aggregate normalized messages
            out = segment_sum(src_feat, dst, num_nodes)

            # Normalize by destination degrees
            dst_norm = norm_coeff  # [num_nodes]
            out = out * dst_norm.unsqueeze(-1)  # [num_nodes, out_feats]
        else:
            # No normalization
            src_feat = feat[src]
            out = segment_sum(src_feat, dst, num_nodes)

        if self.bias is not None:
            out = out + self.bias

        # Apply activation
        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out)
        elif self.activation == "sigmoid":
            out = F.sigmoid(out)
        elif self.activation == "tanh":
            out = F.tanh(out)

        return out


class MPNNConv(nn.Layer):
    """Message Passing Neural Network convolution layer.

    This is the core component of GDI-NN, combining NNConv with GRU
    for iterative message passing over multiple steps.

    Args:
        node_in_feats: Input node feature dimension
        edge_in_feats: Input edge feature dimension
        node_out_feats: Output node feature dimension
        edge_hidden_feats: Hidden dimension for edge function
        num_step_message_passing: Number of message passing steps
        activation: Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh', 'softplus')
        dropout: Dropout rate (0.0 means no dropout)
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        node_out_feats: int = 128,
        edge_hidden_feats: int = 32,
        num_step_message_passing: int = 6,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.node_in_feats = node_in_feats
        self.edge_in_feats = edge_in_feats
        self.node_out_feats = node_out_feats
        self.num_step_message_passing = num_step_message_passing
        self.activation = activation

        self.mpnn_activation = get_activation(activation)

        # Project node features: Linear + Activation (matches original)
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            get_activation(activation, get_nn=True)(),
        )

        # Edge function MLP: transforms edge features to edge weights
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            get_activation(activation, get_nn=True)(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )

        # NNConv layer
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type="sum",
        )

        self.gru_cell = nn.GRUCell(node_out_feats, node_out_feats)

    def forward(
        self, graph, node_feats: paddle.Tensor, edge_feats: paddle.Tensor
    ) -> paddle.Tensor:
        """Forward pass with iterative message passing.

        Matches original GDI-NN MPNNconv.forward:
        1. Project node features (Linear + Activation)
        2. Initialize hidden state from projected features
        3. For each message passing step:
           a. Apply activation(gnn_layer(graph, node_feats, edge_feats))
           b. GRU update with hidden state

        Args:
            graph: pgl.Graph object
            node_feats: Node features of shape [num_nodes, node_in_feats]
            edge_feats: Edge features of shape [num_edges, edge_in_feats]

        Returns:
            Updated node features of shape [num_nodes, node_out_feats]
        """
        # Project node features: Linear + Activation
        node_feats = self.project_node_feats(node_feats)

        # Initialize hidden state from projected features
        hidden_feats = node_feats  # [num_nodes, node_out_feats]

        # Message passing for multiple steps
        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(
                self.gnn_layer(graph, node_feats, edge_feats)
            )

            # GRUCell update: input [num_nodes, feat], hidden [num_nodes, feat]
            hidden_feats, _ = self.gru_cell(node_feats, hidden_feats)
            node_feats = hidden_feats  # GRUCell output is the new hidden state

        return node_feats
