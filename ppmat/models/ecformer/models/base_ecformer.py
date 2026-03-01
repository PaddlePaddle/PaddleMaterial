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

from abc import ABC
import paddle
import paddle.nn as nn
from paddle.nn import TransformerEncoder, TransformerEncoderLayer

from ..encoders.gin_node_embedding import GINNodeEmbedding
from ..utils.graph_utils import pad_node_features, feat_padding_mask
from paddle_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

def fix_mask_for_paddle(mask, n_head=None):
    """
    简单直接的掩码修复函数
    
    Args:
        mask: 输入掩码
        n_head: 注意力头数 (attention mask 时需要)
    """
    shape = mask.shape
    assert len(shape) == 2
    # 如果是 [batch_size, src_len] 但想用作 attention mask
    batch_size, s_len = shape
    # [32, 73] -> [32, 1, 73, 73]
    if n_head:
        return mask.reshape([batch_size, 1, 1, s_len]).expand([-1, n_head, s_len, -1])
    else:
        return mask.unsqueeze(1).unsqueeze(2).expand([-1, -1, s_len, -1])


class ECFormerBase(nn.Layer, ABC):
    """ECFormer基类 - 所有谱图预测模型的抽象接口"""
    
    def __init__(
        self,
        # GNN参数
        full_atom_feature_dims,
        full_bond_feature_dims,
        bond_float_names,
        bond_angle_float_names,
        bond_id_names,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0.0,
        JK="last",
        residual=False,
        graph_pooling="attention",
        use_geometry_enhanced=True,
        max_node_num=63,
        # Transformer参数
        num_heads=4,
        tf_layers=2,
        tf_dropout=0.1,
        max_peaks=9,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.max_node_num = max_node_num
        self.max_peaks = max_peaks
        self.use_geometry_enhanced = use_geometry_enhanced
        
        # 1. GNN节点编码器
        self.gnn_node = GINNodeEmbedding(
            full_atom_feature_dims=full_atom_feature_dims,
            full_bond_feature_dims=full_bond_feature_dims,
            bond_float_names=bond_float_names,
            bond_angle_float_names=bond_angle_float_names,
            bond_id_names=bond_id_names,
            num_layers=num_layers,
            emb_dim=emb_dim,
            drop_ratio=drop_ratio,
            JK=JK,
            residual=residual,
            use_geometry_enhanced=use_geometry_enhanced
        )
        
        # 2. 图池化层
        self.pool = self._build_pooling(graph_pooling, emb_dim)
        
        # 3. Query嵌入（峰查询向量）
        self.query_embed = nn.Embedding(max_peaks, emb_dim)
        
        # 4. Transformer编码器
        self.tf_encoder = self._build_transformer(emb_dim, num_heads, tf_layers, tf_dropout)
    
    def _build_pooling(self, graph_pooling, emb_dim):
        """构建图池化层"""
        if graph_pooling == "sum":
            return global_add_pool
        elif graph_pooling == "mean":
            return global_mean_pool
        elif graph_pooling == "max":
            return global_max_pool
        elif graph_pooling == "attention":
            return GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1D(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, 1)
                )
            )
        elif graph_pooling == "set2set":
            return Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError(f"Invalid graph pooling type: {graph_pooling}")
    
    def _build_transformer(self, emb_dim, num_heads, num_layers, dropout):
        """构建Transformer编码器"""
        
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim,
            dropout=dropout,
            activation='relu',
        )
        return TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def encode_molecule(
        self,
        x,                    # [N, F] 原子特征
        edge_index,           # [2, E] 边索引
        edge_attr,            # [E, D] 边特征
        batch_data,           # [N] 批次信息
        # 几何增强相关
        ba_edge_index=None,   # [2, E_ba] 键角图边索引
        ba_edge_attr=None,   # [E_ba, D_ba] 键角图边特征
    ):
        """分子编码器 - 纯Tensor输入"""

        # 1. GNN编码
        if self.use_geometry_enhanced and ba_edge_index is not None:
            h_node, _ = self.gnn_node(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                ba_edge_index=ba_edge_index,
                ba_edge_attr=ba_edge_attr
            )
        else:
            h_node = self.gnn_node(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        
        # 2. 节点特征padding（需要batch信息）
        batch_size = batch_data[-1] + 1

        node_feat, node_index = pad_node_features(
            h_node, batch_data, batch_size, self.max_node_num, self.emb_dim
        )
        
        # 3. 图池化
        h_graph = self.pool(h_node, batch_data).unsqueeze(1)
        
        # 4. 拼接图特征和节点特征

        total_node_feat = paddle.concat([h_graph, node_feat], axis=1)
        
        # 5. 生成padding mask
        node_padding_mask = feat_padding_mask(node_index, self.max_node_num)
        pooling_padding_mask = paddle.zeros([node_padding_mask.shape[0], 1], dtype='float32')
        total_padding_mask = paddle.concat([pooling_padding_mask, node_padding_mask], axis=1)
        
        return total_node_feat, total_padding_mask, node_padding_mask
    
    def forward(self, 
                x: paddle.Tensor, 
                edge_index: paddle.Tensor, 
                edge_attr: paddle.Tensor,
                batch_data: paddle.Tensor,
                ba_edge_index: paddle.Tensor = None,
                ba_edge_attr: paddle.Tensor = None,
                query_mask: paddle.Tensor = None):
        # 0. 数据类型检查
        if batch_data.dtype != paddle.int64:
            batch_data = batch_data.astype(paddle.int64)

        # 1. 分子编码
        node_feat, padding_mask, node_padding_mask = self.encode_molecule(x, edge_index, edge_attr,batch_data, ba_edge_index, ba_edge_attr)
        
        # 2. 峰数预测（从图特征）
        graph_feat = node_feat[:, 0, :]
        pred_number = self.pred_number_layer(graph_feat)
        
        # 3. Query准备
        query_feat = self.query_embed.weight.unsqueeze(0).tile([node_feat.shape[0], 1, 1])
        
        # 推理时根据预测峰数生成query mask
        if not self.training:
            pred_peak_num = pred_number.argmax(axis=1)
            peak_position = [
                [1] * int(pred_peak_num[i]) + [-1] * (self.max_peaks - int(pred_peak_num[i]))
                for i in range(len(pred_peak_num))
            ]
            peak_position = paddle.to_tensor(peak_position)
            query_mask = get_key_padding_mask(peak_position)
        
        # 4. Transformer编码
        encoder_input = paddle.concat([node_feat, query_feat], axis=1)
        encoder_padding_mask = paddle.concat([padding_mask, query_mask], axis=1)
        
        encoder_output = self.tf_encoder(encoder_input, fix_mask_for_paddle(encoder_padding_mask))
        
        # 5. 峰位置和符号预测
        query_output = encoder_output[:, node_feat.shape[1]:, :]
        pred_position = self.pred_position_layer(query_output)
        pred_height = self.pred_height_layer(query_output)
        
        # 6. 注意力权重（用于可视化）
        node_feat_output = encoder_output[:, :node_feat.shape[1], :]
        attn_weights = paddle.einsum("bid,bjd->bij", 
            node_feat_output, 
            query_output[:, 0, :].unsqueeze(1)
        )
        attn_weights = attn_weights[:, 1:, :].squeeze()
        attn_mask = node_padding_mask[:, 1:]
        
        return {
            'peak_number': pred_number,
            'peak_position': pred_position,
            'peak_height': pred_height,
            'attention': {
                'weights': attn_weights.cpu().tolist() if not self.training else None,
                'mask': attn_mask.cpu().tolist() if not self.training else None
            }
        }
 
def get_key_padding_mask(tokens):
    key_padding_mask = paddle.zeros(tokens.shape)
    key_padding_mask[tokens == -1] = -paddle.inf
    return key_padding_mask