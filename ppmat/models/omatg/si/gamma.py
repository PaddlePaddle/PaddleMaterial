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

"""Gamma functions for the latent variable gamma(t) * z in stochastic interpolants."""

import paddle

from .abstracts import LatentGamma


class LatentGammaSqrt(LatentGamma):
    """Gamma(t) = sqrt(a * t * (1 - t)). Requires antithetic sampling."""

    def __init__(self, a: float) -> None:
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        self._a = a

    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.sqrt(self._a * t * (1.0 - t))

    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return self._a * (1.0 - 2.0 * t) / (2.0 * paddle.sqrt(self._a * t * (1.0 - t)))

    def requires_antithetic(self) -> bool:
        return True


class LatentGammaEncoderDecoder(LatentGamma):
    """Encoder-decoder gamma. For a=1,p=1,switch=0.5 -> sin^2(pi*t)."""

    def __init__(
        self, a: float = 1.0, switch_time: float = 0.5, power: float = 1.0
    ) -> None:
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        if switch_time <= 0.0 or switch_time >= 1.0:
            raise ValueError("Switch time must be in (0,1).")
        if power < 0.5:
            raise ValueError("Power must be at least 0.5.")
        self._sqrt_a = a**0.5
        self._switch_time = switch_time
        self._power = power

    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return self._sqrt_a * paddle.sin(paddle.pi * a / b) ** 2

    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        assert paddle.all((0.001 <= t) & (t <= 1.0 - 0.001))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = paddle.sin(2.0 * paddle.pi * a / (a + b))
        return (
            -self._sqrt_a
            * self._power
            * paddle.pi
            * a
            * b
            * c
            / (t * (t - 1.0) * (a + b) ** 2)
        )

    def requires_antithetic(self) -> bool:
        return False
