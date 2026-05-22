# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Sigma noise schedule sigma(s) for one-sided variance-exploding interpolants."""

from math import log

import paddle

from .abstracts import Sigma


class GeometricSigma(Sigma):
    """sigma(s) = sigma_min * (sigma_max / sigma_min)^s."""

    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        super().__init__()
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive.")
        if sigma_max <= 0.0:
            raise ValueError("sigma_max must be positive.")
        if sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min.")
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._ratio = sigma_max / sigma_min
        self._log_ratio = log(self._ratio)

    def sigma(self, s: paddle.Tensor) -> paddle.Tensor:
        self._check_t(s)
        return self._sigma_min * self._ratio**s

    def sigma_dot(self, s: paddle.Tensor) -> paddle.Tensor:
        self._check_t(s)
        return self._sigma_min * self._log_ratio * self._ratio**s
