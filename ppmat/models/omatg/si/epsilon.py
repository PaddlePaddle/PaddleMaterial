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

"""Epsilon functions epsilon(t) for SDE stochastic interpolants."""

import paddle

from .abstracts import Epsilon


class ConstantEpsilon(Epsilon):
    """Epsilon(t) = c, constant in time."""

    def __init__(self, c: float) -> None:
        super().__init__()
        self._c = c

    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.full_like(t, self._c)


class VanishingEpsilon(Epsilon):
    """Epsilon(t) = c * f1 * f2, product of Fermi functions vanishing at endpoints."""

    def __init__(self, c: float = 1.0, sigma: float = 0.01, mu: float = 0.075) -> None:
        super().__init__()
        self._c = c
        self._sigma = sigma
        self._mu = mu

    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        f1 = 1 / (1 + paddle.exp(-(t - self._mu) / self._sigma))
        f2 = 1 / (1 + paddle.exp(-(1 - self._mu - t) / self._sigma))
        return self._c * f1 * f2
