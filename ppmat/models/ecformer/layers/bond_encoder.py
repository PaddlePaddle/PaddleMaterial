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

import paddle.nn as nn

class BondEncoder(nn.Layer):
    """键特征编码器 - 将离散键特征映射为连续向量"""
    
    def __init__(self, full_bond_feature_dims, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = nn.LayerList()
        
        for dim in full_bond_feature_dims:
            emb = nn.Embedding(dim + 5, emb_dim)
            nn.initializer.XavierUniform()(emb.weight)
            self.bond_embedding_list.append(emb)
    
    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return bond_embedding