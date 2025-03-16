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

from typing import Callable
from typing import Optional

import paddle
import paddle.nn.functional as F


class MAEMetric:
    """Mean absolute error."""

    def __init__(self, callback_fn: Optional[Callable] = None):
        self.callback_fn = callback_fn

    def __call__(self, pred, label):
        if self.callback_fn is not None:
            pred, label = self.callback_fn(pred, label)

        mask = paddle.isnan(label) is False
        if not mask.any():
            return 0.0

        mae = F.l1_loss(pred[mask], label[mask], "none")
        mae = mae.mean().item()
        return mae
