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

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math


@paddle.jit.to_static
def gaussian(x: paddle.Tensor, mean, std) -> paddle.Tensor:
    a = (2 * math.pi) ** 0.5
    return paddle.exp(-0.5 * ((x - mean) / std) ** 2) / (a * std)


class PolynomialEnvelope(paddle.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    """

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: paddle.Tensor) -> paddle.Tensor:
        env_val = 1 + d_scaled**self.p * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return paddle.where(d_scaled < 1, env_val, paddle.zeros_like(env_val))


class GaussianSmearing(paddle.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> paddle.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class RadialMLP(paddle.nn.Module):
    """
    Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """

    def __init__(self, channels_list) -> None:
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue
            modules.append(
                paddle.compat.nn.Linear(input_channels, channels_list[i], bias=True)
            )
            input_channels = channels_list[i]
            if i == len(channels_list) - 1:
                break
            modules.append(paddle.nn.LayerNorm(channels_list[i]))
            modules.append(paddle.nn.SiLU())
        self.net = paddle.nn.Sequential(*modules)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return self.net(inputs)
