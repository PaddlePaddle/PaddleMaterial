# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .utils import radial_basis


class RadialBasisFunction(nn.Layer):
    """Radial basis function layer"""

    def __init__(self, r_max, num_basis):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis

    def forward(self, r):
        """Forward pass of radial basis function"""
        return radial_basis(r, self.r_max, self.num_basis)


class EquivariantLayer(nn.Layer):
    """Equivariant message passing layer"""

    def __init__(self, hidden_dim, num_basis, r_max):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.r_max = r_max

        # Radial basis function
        self.rbf = RadialBasisFunction(r_max, num_basis)

        # Message passing weights
        self.W_message = nn.Linear(num_basis + hidden_dim, hidden_dim)

        # Update weights
        self.W_update = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_indices, edge_distances):
        """Forward pass of equivariant layer"""
        # Get radial basis features
        rbf_features = self.rbf(edge_distances)

        # Get source and target nodes
        src, dst = edge_indices

        # Message passing
        x_src = x[src]
        message = paddle.concat([x_src, rbf_features], axis=-1)
        message = self.W_message(message)
        message = F.relu(message)

        # 按目标节点聚合消息（保留原 scatter_add 累加语义）
        aggregated = paddle.zeros_like(x)
        # index 形状需与 message 一致：[E, H]
        index = dst.reshape([-1, 1]).expand_as(message)
        aggregated = paddle.scatter_add(aggregated, 0, index.astype("int64"), message)

        # Update node features
        x_new = paddle.concat([x, aggregated], axis=-1)
        x_new = self.W_update(x_new)
        x_new = F.relu(x_new)

        return x_new
