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

from __future__ import annotations

import paddle
import paddle.nn as nn


class RMSEMetric(nn.Layer):
    """Root Mean Squared Error metric.

    Args:
        eps (float): Numerical epsilon before sqrt. Defaults to 1e-12.
        ignore_nan (bool): Ignore NaN values in pred/label. Defaults to True.
    """

    def __init__(self, eps: float = 1e-12, ignore_nan: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.ignore_nan = bool(ignore_nan)

    def forward(self, pred: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
        pred = paddle.cast(pred, "float32")
        label = paddle.cast(label, "float32")

        if self.ignore_nan:
            valid = paddle.isfinite(pred) & paddle.isfinite(label)
            valid_count = int(paddle.sum(valid.astype("int32")).item())
            if valid_count == 0:
                return paddle.to_tensor(float("nan"), dtype="float32")
            pred = paddle.masked_select(pred, valid)
            label = paddle.masked_select(label, valid)

        mse = paddle.mean((pred - label) ** 2)
        return paddle.sqrt(mse + self.eps)


class RelativeErrorMetric(nn.Layer):
    """Mean relative error metric.

    Formula:
        mean(abs(pred - label) / max(abs(label), eps))

    Args:
        eps (float): Denominator clamp epsilon. Defaults to 1e-8.
        percentage (bool): Return percentage value (*100). Defaults to False.
        ignore_nan (bool): Ignore NaN values in pred/label. Defaults to True.
        ignore_small_label (bool): Ignore labels whose abs is below `small_label_tol`.
            Defaults to False.
        small_label_tol (float): Threshold used by `ignore_small_label`.
            Defaults to 1e-12.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        percentage: bool = False,
        ignore_nan: bool = True,
        ignore_small_label: bool = False,
        small_label_tol: float = 1e-12,
    ):
        super().__init__()
        self.eps = float(eps)
        self.percentage = bool(percentage)
        self.ignore_nan = bool(ignore_nan)
        self.ignore_small_label = bool(ignore_small_label)
        self.small_label_tol = float(small_label_tol)

    def forward(self, pred: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
        pred = paddle.cast(pred, "float32")
        label = paddle.cast(label, "float32")

        valid = paddle.ones_like(label, dtype="bool")
        if self.ignore_nan:
            valid = valid & paddle.isfinite(pred) & paddle.isfinite(label)
        if self.ignore_small_label:
            valid = valid & (paddle.abs(label) > self.small_label_tol)

        valid_count = int(paddle.sum(valid.astype("int32")).item())
        if valid_count == 0:
            return paddle.to_tensor(float("nan"), dtype="float32")

        pred = paddle.masked_select(pred, valid)
        label = paddle.masked_select(label, valid)

        denominator = paddle.maximum(
            paddle.abs(label),
            paddle.full_like(label, self.eps),
        )
        rel = paddle.abs(pred - label) / denominator
        if self.percentage:
            rel = rel * 100.0
        return paddle.mean(rel)
