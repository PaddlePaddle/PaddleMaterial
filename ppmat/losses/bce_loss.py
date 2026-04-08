# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing_extensions import Literal


class BCEWithLogitsLoss(nn.Layer):
    r"""Binary cross-entropy loss with logits (sigmoid applied internally).

    This is useful for classification tasks in polymer-chemprop and similar models
    where the model outputs raw logits instead of probabilities.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction should be 'mean', 'sum', or 'none', but got {reduction}"
            )
        self.reduction = reduction

    def forward(self, pred, label) -> paddle.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction=self.reduction)
        return loss


class MaskedMSELoss(nn.Layer):
    r"""MSE loss with support for a target mask.

    Computes MSE only on valid (masked) entries, useful when some targets
    are missing in multi-task prediction.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, label, mask=None) -> paddle.Tensor:
        if mask is not None:
            loss = F.mse_loss(pred * mask, label * mask, reduction='sum') / mask.sum()
        else:
            loss = F.mse_loss(pred, label)
        return loss
