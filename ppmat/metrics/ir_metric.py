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
from typing import Dict, Optional, Any


# =========================
# Utilities
# =========================

def _is_dist():
    """Check if distributed training is initialized."""
    try:
        import paddle.distributed as dist
        return dist.is_initialized() and dist.get_world_size() > 1
    except Exception:
        return False


def _all_reduce_sum_(t: paddle.Tensor) -> paddle.Tensor:
    """In-place SUM all_reduce if distributed; returns t."""
    if _is_dist():
        import paddle.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _to_f32(x) -> paddle.Tensor:
    """Convert to float32 tensor."""
    return (
        paddle.to_tensor(x, dtype="float32")
        if not isinstance(x, paddle.Tensor)
        else x.astype("float32")
    )


# =========================
# IR Metrics
# =========================

class IRMetrics(nn.Layer):
    """Evaluation metrics for ECFormer IR task.
    
    Computes:
        - Number-RMSE: RMSE of predicted vs true peak count
        - Position-RMSE: RMSE of predicted vs true peak positions (class indices)
        - Height-RMSE: RMSE of predicted vs true peak intensities (if enabled)
    """
    
    def __init__(self, use_height_prediction=True):
        """
        Args:
            use_height_prediction (bool): Whether height prediction is used
        """
        super().__init__()
        self.use_height_prediction = use_height_prediction
        
        # Accumulators for streaming metrics
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.num_rmse_sum = _to_f32(0.0)
        self.pos_rmse_sum = _to_f32(0.0)
        self.height_rmse_sum = _to_f32(0.0)
        self.pos_count = _to_f32(0.0)
        self.height_count = _to_f32(0.0)
        self.num_samples = _to_f32(0.0)
    
    def update(self, predictions: Dict[str, paddle.Tensor], targets: Dict[str, paddle.Tensor]):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            predictions: dict from model forward
                - peak_number: [batch_size, max_peaks+1] logits for peak count
                - peak_position: [batch_size, max_peaks, num_position_classes] logits for positions
                - peak_height (optional): [batch_size, max_peaks] predicted intensity values
            targets: dict from dataloader
                - peak_num: [batch_size] true peak counts
                - peak_position: [batch_size, max_peaks] true position labels
                - peak_height: [batch_size, max_peaks] true intensity values
        """
        batch_size = targets['peak_num'].shape[0]
        
        # Peak number predictions
        pred_nums = predictions['peak_number'].argmax(axis=1)
        true_nums = targets['peak_num']
        
        # Number RMSE accumulation
        num_errors = (pred_nums - true_nums).astype('float32')
        self.num_rmse_sum += paddle.sum(paddle.square(num_errors))
        
        # Process each sample for position and height metrics
        for i in range(batch_size):
            n_true = int(true_nums[i])
            n_pred = int(pred_nums[i])
            
            if n_true > 0 and n_pred > 0:
                n_match = min(n_true, n_pred)
                
                # Position errors (only on matched peaks)
                pos_true = targets['peak_position'][i, :n_match].astype('int64')
                pos_pred = predictions['peak_position'][i, :n_match, :].argmax(axis=1)
                pos_errors = (pos_pred - pos_true).astype('float32')
                self.pos_rmse_sum += paddle.sum(paddle.square(pos_errors))
                self.pos_count += _to_f32(n_match)
                
                # Height errors if enabled
                if self.use_height_prediction and 'peak_height' in predictions:
                    height_true = targets['peak_height'][i, :n_match].astype('float32')
                    height_pred = predictions['peak_height'][i, :n_match].reshape([-1])
                    height_errors = height_true - height_pred
                    self.height_rmse_sum += paddle.sum(paddle.square(height_errors))
                    self.height_count += _to_f32(n_match)
        
        self.num_samples += _to_f32(batch_size)
    
    def accumulate(self) -> Dict[str, float]:
        """
        Compute accumulated metrics.
        
        Returns:
            dict: Dictionary containing all metrics
        """
        # Distributed reduction
        num_rmse_sum = _all_reduce_sum_(self.num_rmse_sum.clone())
        pos_rmse_sum = _all_reduce_sum_(self.pos_rmse_sum.clone())
        height_rmse_sum = _all_reduce_sum_(self.height_rmse_sum.clone())
        pos_count = _all_reduce_sum_(self.pos_count.clone())
        height_count = _all_reduce_sum_(self.height_count.clone())
        num_samples = _all_reduce_sum_(self.num_samples.clone())
        
        # Compute final metrics
        num_rmse = paddle.sqrt(num_rmse_sum / paddle.maximum(num_samples, _to_f32(1.0))).item()
        pos_rmse = paddle.sqrt(pos_rmse_sum / paddle.maximum(pos_count, _to_f32(1.0))).item()
        
        metrics = {
            'num_rmse': num_rmse,
            'pos_rmse': pos_rmse,
        }
        
        if self.use_height_prediction:
            height_rmse = paddle.sqrt(
                height_rmse_sum / paddle.maximum(height_count, _to_f32(1.0))
            ).item()
            metrics['height_rmse'] = height_rmse
        
        return metrics
    
    def forward(self, predictions: Dict[str, paddle.Tensor], targets: Dict[str, paddle.Tensor]) -> Dict[str, float]:
        """
        Compute metrics for a single batch (non-streaming version).
        
        Args:
            predictions: dict from model forward
            targets: dict from dataloader
        
        Returns:
            dict: Dictionary containing all metrics for this batch
        """
        self.reset()
        self.update(predictions, targets)
        return self.accumulate()