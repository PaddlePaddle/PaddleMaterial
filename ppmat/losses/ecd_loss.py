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


class ECDLoss(nn.Layer):
    """Loss function for ECFormer ECD task.
    
    Combines three cross-entropy losses for peak number, position, and symbol,
    with symbol loss weighted as in the original paper.
    """
    
    def __init__(self, loss_weight_height=2.0, num_position_classes=20, height_classes=2):
        """
        Args:
            loss_weight_height (float): Weight for peak symbol loss (2.0 in paper)
            num_position_classes (int): Number of position classes (default: 20)
            height_classes (int): Number of symbol classes (default: 2: positive/negative)
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_weight_height = loss_weight_height
        self.num_position_classes = num_position_classes
        self.height_classes = height_classes
        
        # Accumulators for epoch-level statistics
        self.reset()
    
    def forward(self, predictions, targets):
        """
        Compute ECD task losses.
        
        Args:
            predictions (dict): Model outputs containing:
                - peak_number (Tensor): [batch_size, max_peaks] logits for peak count
                - peak_position (Tensor): [batch_size, max_peaks, num_position_classes] logits for positions
                - peak_height (Tensor): [batch_size, max_peaks, height_classes] logits for symbols
            targets (dict): Ground truth containing:
                - peak_num (Tensor): [batch_size] true peak counts
                - peak_position (Tensor): [batch_size, max_peaks] true position labels
                - peak_height (Tensor): [batch_size, max_peaks] true symbol labels
        
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
            
            # Position loss (only for valid peaks)
            pos_pred = predictions['peak_position'][i, :n_peaks, :].reshape([-1, self.num_position_classes])
            pos_gt = targets['peak_position'][i, :n_peaks].reshape([-1])
            loss_pos_total += self.ce_loss(pos_pred, pos_gt)
            
            # Symbol loss
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
        
        # Total loss with weighted symbol term
        total_loss = loss_num + self.loss_weight_height * loss_height + loss_pos
        
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