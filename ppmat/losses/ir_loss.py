# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn


class IRLoss(nn.Layer):
    """Loss function for ECFormer IR task.
    
    Combines cross-entropy loss for peak position and peak number,
    and MSE loss for peak intensity (height) regression.
    """
    
    def __init__(self, num_position_classes=36, use_height_prediction=True):
        """
        Args:
            num_position_classes (int): Number of position classes (default: 36 for IR)
            use_height_prediction (bool): Whether to use height (intensity) regression loss
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.num_position_classes = num_position_classes
        self.use_height_prediction = use_height_prediction
        
        # Accumulators for epoch-level statistics
        self.reset()
    
    def forward(self, predictions, targets):
        """
        Compute IR task losses.
        
        Args:
            predictions (dict): Model outputs containing:
                - peak_number (Tensor): [batch_size, max_peaks+1] logits for peak count
                - peak_position (Tensor): [batch_size, max_peaks, num_position_classes] logits for positions
                - peak_height (Tensor, optional): [batch_size, max_peaks] predicted intensity values
            targets (dict): Ground truth containing:
                - peak_number (Tensor): [batch_size] true peak counts
                - peak_position (Tensor): [batch_size, max_peaks] true position labels
                - peak_height (Tensor): [batch_size, max_peaks] true intensity values
        
        Returns:
            dict: Loss components and total loss
        """
        # Peak number loss
        loss_num = self.ce_loss(predictions['peak_number'], targets['peak_number'])
        
        batch_size = targets['peak_number'].shape[0]
        
        loss_pos_total = 0.0
        loss_height_total = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            n_peaks = int(targets['peak_number'][i])
            if n_peaks == 0:
                continue
            
            # Position loss (cross-entropy)
            pos_pred = predictions['peak_position'][i, :n_peaks, :].reshape([-1, self.num_position_classes])
            pos_gt = targets['peak_position'][i, :n_peaks].reshape([-1])
            loss_pos_total += self.ce_loss(pos_pred, pos_gt)
            
            # Height loss (MSE regression) if enabled
            if self.use_height_prediction and 'peak_height' in predictions:
                height_pred = predictions['peak_height'][i, :n_peaks].reshape([-1])
                height_gt = targets['peak_height'][i, :n_peaks].reshape([-1])
                loss_height_total += self.mse_loss(height_pred, height_gt)
            
            valid_samples += 1
        
        # Average losses over valid samples
        loss_pos = loss_pos_total / valid_samples if valid_samples > 0 else paddle.to_tensor(0.0)
        total_loss = loss_num + loss_pos
        
        if self.use_height_prediction:
            loss_height = loss_height_total / valid_samples if valid_samples > 0 else paddle.to_tensor(0.0)
            total_loss += loss_height
        else:
            loss_height = paddle.to_tensor(0.0)
        
        # Update accumulators for epoch statistics
        self._accumulate(loss_num, loss_pos, loss_height, valid_samples)
        
        return {
            "loss": total_loss,
            "loss_num": loss_num,
            "loss_pos": loss_pos,
            "loss_height": loss_height,
        }
    
    def _accumulate(self, loss_num, loss_pos, loss_height, valid_samples):
        """Accumulate losses for epoch-level statistics."""
        self.loss_num_sum += loss_num.item() if hasattr(loss_num, 'item') else loss_num
        self.loss_pos_sum += loss_pos.item() if hasattr(loss_pos, 'item') else loss_pos
        self.loss_height_sum += loss_height.item() if hasattr(loss_height, 'item') else loss_height
        self.total_samples += valid_samples
    
    def reset(self):
        """Reset accumulated statistics."""
        self.loss_num_sum = 0.0
        self.loss_pos_sum = 0.0
        self.loss_height_sum = 0.0
        self.total_samples = 0
    
    def log_epoch_metrics(self):
        """Return epoch-level loss statistics."""
        if self.total_samples == 0:
            return {
                "train_epoch/loss_num": -1.0,
                "train_epoch/loss_pos": -1.0,
                "train_epoch/loss_height": -1.0,
            }
        
        return {
            "train_epoch/loss_num": self.loss_num_sum / self.total_samples,
            "train_epoch/loss_pos": self.loss_pos_sum / self.total_samples,
            "train_epoch/loss_height": self.loss_height_sum / self.total_samples,
        }