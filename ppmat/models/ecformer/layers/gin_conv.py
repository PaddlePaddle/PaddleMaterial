# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_geometric.nn import MessagePassing
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GINConv(MessagePassing):
    """图同构卷积层"""
    
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), 
            nn.BatchNorm1D(emb_dim), 
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.eps = paddle.create_parameter(
            shape=[1], 
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Assign(paddle.to_tensor([0.]))
        )
    
    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out
    
    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out