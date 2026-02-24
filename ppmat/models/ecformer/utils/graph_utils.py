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
import numpy as np

def index_transform(raw_index, batch_size):
    """将压缩的批次索引还原为每个样本的节点索引列表"""
    
    def get_index1(lst=None, batch_num=-1):
        return [index for (index, value) in enumerate(lst) if value == batch_num]
    
    raw_index = raw_index.tolist()
    index_list = []
    for batch_id in range(batch_size):
        index_list.append(get_index1(raw_index, batch_id))
    return index_list


def get_key_padding_mask(tokens):
    """生成key padding mask"""
    key_padding_mask = paddle.zeros(tokens.shape)
    key_padding_mask[tokens == -1] = -paddle.inf
    return key_padding_mask


def feat_padding_mask(index, max_node_num):
    """根据节点索引生成特征padding mask"""
    new_index = []
    for itm_list in index:
        new_index.append(itm_list + [-1] * (max_node_num - len(itm_list)))
    new_index = paddle.to_tensor(new_index)
    return get_key_padding_mask(new_index)


def pad_node_features(molecule_features, batch_index, this_batch_size, max_node_num, emb_dim):
    """将压缩的节点特征padding为 [batch, max_node, emb_dim] 格式"""
    index_list = index_transform(batch_index, this_batch_size)
    
    new_batch_list = []
    for batch_id in range(this_batch_size):
        empty_batch_tensor = paddle.zeros([max_node_num, emb_dim])
        for i in range(len(index_list[batch_id])):
            empty_batch_tensor[i, :] = molecule_features[index_list[batch_id][i], :]
        new_batch_list.append(empty_batch_tensor)
    
    node_feature = paddle.stack(new_batch_list, axis=0)
    return node_feature, index_list