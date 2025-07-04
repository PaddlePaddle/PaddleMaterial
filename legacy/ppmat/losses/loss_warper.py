# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn


class LossWarper(nn.Layer):
    r"""Class for loss warp."""

    def __init__(
        self,
        loss_fn: Dict[str, nn.Layer],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, pred, label) -> paddle.Tensor:
        losses = {}
        for key in self.loss_fn:
            losses[key] = self.loss_fn[key](pred[key], label[key])

        loss = 0
        for key in losses:
            if self.weights is not None:
                loss += self.weights[key] * losses[key]
        losses["loss"] = loss
        return losses
