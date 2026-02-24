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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..layers.atom_encoder import AtomEncoder
from ..layers.bond_encoder import BondEncoder
from ..layers.rbf import BondFloatRBF, BondAngleFloatRBF
from ..layers.gin_conv import GINConv


class GINNodeEmbedding(nn.Layer):
    """GIN节点嵌入模块 - 支持几何增强的双图结构"""
    
    def __init__(
        self,
        full_atom_feature_dims,
        full_bond_feature_dims,
        bond_float_names,
        bond_angle_float_names,
        bond_id_names,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        use_geometry_enhanced=True
    ):
        super(GINNodeEmbedding, self).__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.use_geometry_enhanced = use_geometry_enhanced
        self.bond_id_names = bond_id_names
        
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        # 编码器
        self.atom_encoder = AtomEncoder(full_atom_feature_dims, emb_dim)
        self.bond_encoder = BondEncoder(full_bond_feature_dims, emb_dim)
        self.bond_float_encoder = BondFloatRBF(bond_float_names, emb_dim)
        self.bond_angle_encoder = BondAngleFloatRBF(bond_angle_float_names, emb_dim)
        
        # GNN层列表
        self.convs = nn.LayerList()
        self.convs_bond_angle = nn.LayerList()
        self.convs_bond_embedding = nn.LayerList()
        self.convs_bond_float = nn.LayerList()
        self.convs_angle_float = nn.LayerList()
        self.batch_norms = nn.LayerList()
        self.batch_norms_ba = nn.LayerList()
        
        for _ in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.convs_bond_angle.append(GINConv(emb_dim))
            self.convs_bond_embedding.append(BondEncoder(full_bond_feature_dims, emb_dim))
            self.convs_bond_float.append(BondFloatRBF(bond_float_names, emb_dim))
            self.convs_angle_float.append(BondAngleFloatRBF(bond_angle_float_names, emb_dim))
            self.batch_norms.append(nn.BatchNorm1D(emb_dim))
            self.batch_norms_ba.append(nn.BatchNorm1D(emb_dim))
    
        
    def forward(
        self,
        x,                    # [N, F] 原子特征
        edge_index,           # [2, E] 边索引
        edge_attr,            # [E, D] 边特征
        # 几何增强相关输入
        ba_edge_index=None,   # [2, E_ba] 键角图边索引
        ba_edge_attr=None,   # [E_ba, D_ba] 键角图边特征
    ):
        """
        前向传播
        """
        # 1. 原子特征编码
        if x.dtype != paddle.int64:
            x = x.astype(paddle.int64)
        h_list = [self.atom_encoder(x)]
        
        if self.use_geometry_enhanced and ba_edge_index is not None:
            return self._forward_enhanced(
                h_list, edge_index, edge_attr, 
                ba_edge_index, ba_edge_attr
            )
        else:
            return self._forward_simple(
                h_list, edge_index, edge_attr
            )
    
    def _forward_enhanced(self, h_list, edge_index, edge_attr, 
                          ba_edge_index, ba_edge_attr):
        """几何增强前向传播"""
        
        bond_id_len = len(self.bond_id_names)
        
        # 初始化边表示
        h_list_ba = [self.bond_float_encoder(
            edge_attr[:, bond_id_len:edge_attr.shape[1]+1].astype('float32')
        ) + self.bond_encoder(
            edge_attr[:, 0:bond_id_len].astype('int64')
        )]
        
        for layer in range(self.num_layers):
            # 节点更新
            h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
            
            # 边更新
            cur_h_ba = self.convs_bond_embedding[layer](
                edge_attr[:, 0:bond_id_len].astype('int64')
            ) + self.convs_bond_float[layer](
                edge_attr[:, bond_id_len:edge_attr.shape[1]+1].astype('float32')
            )
            cur_angle_hidden = self.convs_angle_float[layer](ba_edge_attr)
            h_ba = self.convs_bond_angle[layer](cur_h_ba, ba_edge_index, cur_angle_hidden)
            
            # Dropout和残差
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
                h_ba = F.dropout(h_ba, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
            
            if self.residual:
                h += h_list[layer]
                h_ba += h_list_ba[layer]
            
            h_list.append(h)
            h_list_ba.append(h_ba)
        
        # JK连接策略
        if self.JK == "last":
            node_representation = h_list[-1]
            edge_representation = h_list_ba[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)
            edge_representation = sum(h_list_ba)
        
        return node_representation, edge_representation
    
    def _forward_simple(self, h_list, edge_index, edge_attr):
        """简化前向传播"""
        bond_id_len = len(self.bond_id_names)
        
        for layer in range(self.num_layers):
            h = self.convs[layer](
                h_list[layer],
                edge_index,
                self.convs_bond_embedding[layer](edge_attr[:, 0:bond_id_len].astype('int64')) +
                self.convs_bond_float[layer](edge_attr[:, bond_id_len:edge_attr.shape[1]+1].astype('float32'))
            )
            h = self.batch_norms[layer](h)
            
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
            if self.residual:
                h += h_list[layer]
            
            h_list.append(h)
        
        if self.JK == "last":
            return h_list[-1]
        elif self.JK == "sum":
            return sum(h_list)