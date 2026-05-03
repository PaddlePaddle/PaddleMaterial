# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AUC Metric Wrapper for binary classification.

This wrapper makes paddle.metric.Auc callable like a loss function,
allowing it to be used in the trainer's metric computation.
"""

import paddle


class AucMetric:
    """Wrapper for paddle.metric.Auc to make it callable.

    This wrapper adapts the paddle.metric.Auc interface to be compatible
    with the trainer's metric computation, which expects a callable function
    that takes (pred, label) and returns a metric value.

    Args:
        curve (str): Specifies the mode of the curve to be computed,
            'ROC' or 'PR' for the Precision-Recall-curve. Default is 'ROC'.
        num_thresholds (int): The number of thresholds to use when
            discretizing the roc curve. Default is 4095.
        name (str, optional): String name of the metric instance. Default is 'auc'.
    """

    def __init__(
        self, curve: str = "ROC", num_thresholds: int = 4095, name: str = "auc"
    ):
        self.auc = paddle.metric.Auc(
            curve=curve, num_thresholds=num_thresholds, name=name
        )

    def __call__(self, pred: paddle.Tensor, label: paddle.Tensor) -> float:
        """Compute AUC for the given predictions and labels.

        Args:
            pred (paddle.Tensor): Predictions, shape [batch_size] or [batch_size, 1].
                Values should be probabilities (after sigmoid) for binary classification.
            label (paddle.Tensor): Labels, shape [batch_size] or [batch_size, 1].
                Values should be 0 or 1.

        Returns:
            float: AUC value.
        """
        # Reset the metric for this batch
        self.auc.reset()

        # Ensure pred and label have correct shapes
        if pred.ndim == 1:
            pred = pred.unsqueeze(1)
        if label.ndim == 1:
            label = label.unsqueeze(1)

        # For binary classification, Auc expects predictions as [batch_size, 2]
        # where pred[:, 0] is probability of class 0, pred[:, 1] is probability of class 1
        # Since pred is probability of class 1 (after sigmoid), we need to construct:
        # [1 - pred, pred]
        pred_probs = paddle.concat([1 - pred, pred], axis=1)

        # Update the metric
        self.auc.update(preds=pred_probs, labels=label)

        # Compute and return the AUC
        return self.auc.accumulate()
