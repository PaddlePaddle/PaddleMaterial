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

from .base_ecformer import ECFormerBase
from ..utils.graph_utils import get_key_padding_mask


class ECFormerECD(ECFormerBase):
    """ECFormer for ECD光谱预测 - 峰属性解耦版本"""
    
    def __init__(
        self,
        num_position_classes = 20,
        height_classes       = 2,       
        loss_weight_height   = 2.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_position_classes = num_position_classes
        self.height_classes = height_classes
        self.loss_weight_height = loss_weight_height
        
        # 峰数预测头
        self.pred_number_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 2, self.max_peaks)
        )
        
        # 峰位置预测头
        self.pred_position_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, num_position_classes)
        )
        
        # 峰符号预测头
        self.pred_height_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, height_classes)
        )
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
    
    def get_loss(self, predictions, targets):
        """ECD任务损失：峰数 + 位置 + 符号"""
        # 峰数损失
        loss_num = self.ce_loss(predictions['peak_number'], targets['peak_num'])
        
        # 由于每个样本峰数不同，需要动态处理
        batch_size = targets['peak_num'].shape[0]
        
        loss_pos_total = 0
        loss_height_total = 0
        valid_samples = 0
        
        for i in range(batch_size):
            n_peaks = int(targets['peak_num'][i])
            if n_peaks == 0:
                continue
            
            # 位置损失
            pos_pred = predictions['peak_position'][i, :n_peaks, :].reshape([-1, self.num_position_classes])
            pos_gt = targets['peak_position'][i, :n_peaks].reshape([-1])
            loss_pos_total += self.ce_loss(pos_pred, pos_gt)
            
            # 符号损失
            height_pred = predictions['peak_height'][i, :n_peaks, :].reshape([-1, self.height_classes])
            height_gt = targets['peak_height'][i, :n_peaks].reshape([-1])
            loss_height_total += self.ce_loss(height_pred, height_gt)
            
            valid_samples += 1
        
        if valid_samples > 0:
            loss_pos = loss_pos_total / valid_samples
            loss_height = loss_height_total / valid_samples
        else:
            loss_pos = paddle.to_tensor(0.0)
            loss_height = paddle.to_tensor(0.0)
        
        return loss_num + self.loss_weight_height * loss_height + loss_pos
    
    def get_metrics(self, predictions, targets):
        """ECD任务评估指标：Number-RMSE, Position-RMSE, Symbol-Acc"""
        
        batch_size = targets['peak_num'].shape[0]
        
        # 峰数预测
        pred_nums = predictions['peak_number'].argmax(axis=1)
        true_nums = targets['peak_num']
        
        # 位置误差
        pos_errors = []
        # 符号准确率
        symbol_correct = 0
        symbol_total = 0
        # 首峰符号准确率
        first_symbol_correct = 0
        first_symbol_total = 0
        
        for i in range(batch_size):
            n_true = int(true_nums[i])
            n_pred = int(pred_nums[i])
            
            if n_true > 0 and n_pred > 0:
                n_match = min(n_true, n_pred)
                
                # 位置误差
                pos_true = targets['peak_position'][i, :n_match].numpy()
                pos_pred = predictions['peak_position'][i, :n_match, :].argmax(axis=1).numpy()
                pos_errors.extend(pos_pred - pos_true)
                
                # 符号准确率
                height_true = targets['peak_height'][i, :n_match].numpy()
                height_pred = predictions['peak_height'][i, :n_match, :].argmax(axis=1).numpy()
                
                symbol_correct += np.sum(height_true == height_pred)
                symbol_total += n_match
                
                # 首峰符号准确率
                if height_true[0] == height_pred[0]:
                    first_symbol_correct += 1
                first_symbol_total += 1
        
        # 计算指标
        pos_rmse = np.sqrt(np.mean(np.square(pos_errors))) if pos_errors else 0.0
        num_rmse = np.sqrt(np.mean(np.square((pred_nums - true_nums).numpy())))
        symbol_acc = symbol_correct / symbol_total if symbol_total > 0 else 0.0
        first_symbol_acc = first_symbol_correct / first_symbol_total if first_symbol_total > 0 else 0.0
        
        return {
            'num_rmse': num_rmse,
            'pos_rmse': pos_rmse,
            'symbol_acc': symbol_acc,
            'first_symbol_acc': first_symbol_acc
        }