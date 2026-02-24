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
from sklearn.metrics import mean_squared_error

from .base_ecformer import ECFormerBase
from ..utils.graph_utils import get_key_padding_mask
from ..utils.loss.soft_dtw_cuda import SoftDTW


class ECFormerIR(ECFormerBase):
    """ECDFormer for IR光谱预测 - 序列回归版本"""
    
    def __init__(
        self,
        spectrum_length=1000,
        num_position_classes=36,
        use_height_prediction=True,
        dtw_gamma=0.1,
        **kwargs
    ):
        # IR任务最大峰数不同
        kwargs['max_peaks'] = kwargs.get('max_peaks', 15)
        
        super().__init__(**kwargs)
        
        self.spectrum_length = spectrum_length
        self.num_position_classes = num_position_classes
        self.use_height_prediction = use_height_prediction
        
        # 峰数预测头（IR最多15个峰）
        self.pred_number_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 2, self.max_peaks + 1)
        )
        
        # 峰位置预测头（IR位置分类更多）
        self.pred_position_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, num_position_classes)
        )
        
        # 峰强度预测头（IR回归）
        if use_height_prediction:
            self.pred_height_layer = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim // 4),
                nn.ReLU(),
                nn.Linear(self.emb_dim // 4, 1)
            )
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        use_cuda = True if "gpu" in paddle.device.get_device() else False
        self.dtw_loss = SoftDTW(use_cuda=use_cuda, gamma=dtw_gamma, normalize=True,)
    
    def get_loss(self, predictions, targets):
        """IR任务损失：峰数 + 位置 + 强度"""
        
        # 峰数损失
        loss_num = self.ce_loss(predictions['peak_number'], targets['peak_num'])
        
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
            
            # 强度损失（回归）
            if self.use_height_prediction and 'peak_height' in predictions:
                height_pred = predictions['peak_height'][i, :n_peaks].reshape([-1])
                height_gt = targets['peak_height'][i, :n_peaks].reshape([-1])
                loss_height_total += self.mse_loss(height_pred, height_gt)
            
            valid_samples += 1
        
        loss_pos = loss_pos_total / valid_samples if valid_samples > 0 else paddle.to_tensor(0.0)
        loss = loss_num + loss_pos
        
        if self.use_height_prediction:
            loss_height = loss_height_total / valid_samples if valid_samples > 0 else paddle.to_tensor(0.0)
            loss += loss_height
        
        return loss
    
    def get_metrics(self, predictions, targets):
        """IR任务评估指标"""
        
        batch_size = targets['peak_num'].shape[0]
        
        # 峰数预测
        pred_nums = predictions['peak_number'].argmax(axis=1)
        true_nums = targets['peak_num']
        
        # 位置误差
        pos_errors = []
        # 高度误差
        height_errors = []
        
        for i in range(batch_size):
            n_true = int(true_nums[i])
            n_pred = int(pred_nums[i])
            
            if n_true > 0 and n_pred > 0:
                n_match = min(n_true, n_pred)
                
                # 位置误差
                pos_true = targets['peak_position'][i, :n_match].numpy()
                pos_pred = predictions['peak_position'][i, :n_match, :].argmax(axis=1).numpy()
                pos_errors.extend(pos_pred - pos_true)
                
                # 强度误差
                if 'peak_height' in predictions:
                    height_true = targets['peak_height'][i, :n_match].numpy()
                    height_pred = predictions['peak_height'][i, :n_match].numpy().flatten()
                    height_errors.extend(np.abs(height_true - height_pred))
        
        # 计算指标
        pos_rmse = np.sqrt(np.mean(np.square(pos_errors))) if pos_errors else 0.0
        num_rmse = np.sqrt(np.mean(np.square((pred_nums - true_nums).numpy())))
        height_rmse = np.sqrt(np.mean(np.square(height_errors))) if height_errors else 0.0
        
        metrics = {
            'num_rmse': num_rmse,
            'pos_rmse': pos_rmse,
        }
        
        if self.use_height_prediction:
            metrics['height_rmse'] = height_rmse
        
        return metrics