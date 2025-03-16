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

import copy
from typing import List
from typing import Optional

import paddle
import paddle.nn as nn


class LossWarper(nn.Layer):
    """This class is used to wrap the loss function, supporting multi-loss calculation.

    Args:
        loss_fns (List[nn.Layer]): The loss functions that need to be wrapped.
        apply_keys (List[str]): The keys of the input data corresponding to the loss
            functions.
        weights (Optional[List[float]], optional): The weight of each loss.
            Defaults to None.
    """

    def __init__(
        self,
        loss_fns: List[nn.Layer],
        apply_keys: List[str],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]

        if len(loss_fns) == 1 and len(apply_keys) > 1:
            loss_fns = [copy.deepcopy(loss_fns[0]) for _ in range(len(apply_keys))]

        self.loss_fns = loss_fns
        self.apply_keys = apply_keys

        assert len(loss_fns) == len(
            apply_keys
        ), "loss_fns and apply_keys must have the same length."
        self.weights = weights
        if self.weights is None:
            self.weights = [1.0] * len(loss_fns)
        assert len(self.weights) == len(
            loss_fns
        ), "weights must have the same length as loss_fns."

    def forward(self, pred, label) -> paddle.Tensor:
        losses = {}
        loss = 0.0
        for i, key in enumerate(self.apply_keys):
            losses[key] = self.loss_fns[i](pred[key], label[key])
            loss += self.weights[i] * losses[key]
        losses["loss"] = loss
        return losses
