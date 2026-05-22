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

"""Tau functions tau(t) for one-sided variance-preserving interpolants."""

from math import exp, pi, sin

import paddle

from .abstracts import Tau


class TauConstantSchedule(Tau):
    """tau(t) = t, corresponding to constant noise schedule beta(s) = 2."""

    def __init__(self) -> None:
        super().__init__()

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return t.clone()

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.ones_like(t)


class TauLinearSchedule(Tau):
    """tau(t) for linear noise schedule beta(s) = beta_min + (beta_max - beta_min) * s."""

    def __init__(self, beta_min: float, beta_max: float) -> None:
        super().__init__()
        if beta_min <= 0.0:
            raise ValueError("beta_min must be positive.")
        if beta_max <= 0.0:
            raise ValueError("beta_max must be positive.")
        if beta_max <= beta_min:
            raise ValueError("beta_max must be greater than beta_min.")
        self._beta_min = beta_min
        self._beta_max = beta_max

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        log_t = paddle.log(t)
        return paddle.exp(
            0.5 * self._beta_min * log_t
            - 0.25 * (self._beta_max - self._beta_min) * log_t**2
        )

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        log_t = paddle.log(t)
        exp_factor = paddle.exp(
            0.5 * self._beta_min * log_t
            - 0.25 * (self._beta_max - self._beta_min) * log_t**2
        )
        return exp_factor * (
            0.5 * self._beta_min / t
            - 0.5 * (self._beta_max - self._beta_min) * log_t / t
        )


class TauCosineSchedule(Tau):
    """tau(t) for cosine noise schedule."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        if offset < 0.0:
            raise ValueError("offset must be non-negative.")
        self._offset = offset
        self._offset_factor = pi / (2.0 + 2.0 * self._offset)
        self._csc_prefactor = 1.0 / sin(self._offset_factor)
        self._one_over_e = 1.0 / exp(1.0)

    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.where(
            t > self._one_over_e,
            self._csc_prefactor
            * paddle.sin(self._offset_factor * (1.0 + paddle.log(t))),
            paddle.zeros_like(t),
        )

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.where(
            t > self._one_over_e,
            self._csc_prefactor
            * self._offset_factor
            * paddle.cos(self._offset_factor * (1.0 + paddle.log(t)))
            / t,
            paddle.zeros_like(t),
        )
