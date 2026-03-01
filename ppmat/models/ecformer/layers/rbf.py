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
import numpy as np

class RBF(nn.Layer):
    """径向基函数"""
    
    def __init__(self,
                 centers: paddle.nn.parameter.Parameter,
                 gamma: paddle.nn.parameter.Parameter):
        super(RBF, self).__init__()
        self.centers = centers.data.reshape([1, -1])
        self.gamma = gamma.data
    
    def forward(self, x):
        x = x.reshape([-1, 1])
        return paddle.exp(-self.gamma * paddle.square(x - self.centers))


class BondFloatRBF(nn.Layer):
    """连续键特征RBF编码器"""
    
    def __init__(self, bond_float_names, embed_dim, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names
        
        if rbf_params is None:
            self.rbf_params = self._default_rbf_params()
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)
    
    def _default_rbf_params(self):
        return {
            'bond_length': (paddle.create_parameter(shape=paddle.arange(0, 2, 0.1).shape, 
                                                       dtype=paddle.arange(0, 2, 0.1).dtype,
                                                       default_initializer=paddle.nn.initializer.Assign(paddle.arange(0, 2, 0.1))), 
                            paddle.create_parameter(shape=paddle.to_tensor([10.0]).shape, 
                                                       dtype=paddle.to_tensor([10.0]).dtype,
                                                       default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([10.0])))),
        }
    
    def forward(self, bond_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[:, i].reshape([-1, 1])
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Layer):
    """键角连续特征RBF编码器"""
    
    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names
        
        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (paddle.create_parameter(shape=paddle.arange(0, np.pi, 0.1).shape, 
                                                      dtype='float32', 
                                                      default_initializer=paddle.nn.initializer.Assign(paddle.arange(0, np.pi, 0.1))), 
                               paddle.create_parameter(shape=paddle.to_tensor([10.0]).shape,
                                                      dtype=paddle.to_tensor([10.0]).dtype,
                                                      default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([10.0])))),
            }
        else:
            self.rbf_params = rbf_params
        
        self.linear_list = nn.LayerList()
        self.rbf_list = nn.LayerList()
        
        for name in self.bond_angle_float_names:
            if name == 'bond_angle':
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers, gamma)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim)
                self.linear_list.append(linear)
            else:
                linear = nn.Linear(len(self.bond_angle_float_names) - 1, embed_dim)
                self.linear_list.append(linear)
                break
    
    def forward(self, bond_angle_float_features):
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            if name == 'bond_angle':
                x = bond_angle_float_features[:, i].reshape([-1, 1])
                rbf_x = self.rbf_list[i](x)
                out_embed += self.linear_list[i](rbf_x)
            else:
                x = bond_angle_float_features[:, 1:]
                out_embed += self.linear_list[i](x)
                break
        return out_embed