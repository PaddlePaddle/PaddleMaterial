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

from ..paddle_utils import *

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


class ScaledSiLU(paddle.nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418

    def forward(self, inputs):
        return (
            paddle.nn.functional.silu(inputs, inplace=self.inplace) * self.scale_factor
        )

    def extra_repr(self):
        str = f"scale_factor={self.scale_factor}"
        if self.inplace:
            str = str + ", inplace=True"
        return str


class ScaledSwiGLU(paddle.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = paddle.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = ScaledSiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        return w_1 * w_2


class SwiGLU(paddle.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = paddle.nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = paddle.nn.SiLU()

    def forward(self, inputs):
        w = self.w(inputs)
        w_1 = w.narrow(-1, 0, self.out_channels)
        w_1 = self.act(w_1)
        w_2 = w.narrow(-1, self.out_channels, self.out_channels)
        return w_1 * w_2


class SmoothLeakyReLU(paddle.nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = (1 + self.alpha) / 2 * x
        x2 = (1 - self.alpha) / 2 * x * (2 * paddle.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return f"negative_slope={self.alpha}"


class ScaledSmoothLeakyReLU(paddle.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = SmoothLeakyReLU(0.2)
        self.scale_factor = 1.531320475574866

    def forward(self, x):
        return self.act(x) * self.scale_factor

    def extra_repr(self):
        return f"negative_slope={self.act.alpha}, scale_factor={self.scale_factor}"


class ScaledSigmoid(paddle.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.sigmoid(x) * self.scale_factor


class GateActivation(paddle.nn.Module):
    def __init__(
        self, lmax: int, mmax: int, num_channels: int, m_prime: bool = False
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min(2 * lval + 1, 2 * self.mmax + 1)
            num_components = num_components + num_m_components
        expand_index = paddle.zeros([num_components]).long()
        self.m_prime = m_prime
        if self.m_prime:
            start_idx = 0
            length = self.lmax
            expand_index[start_idx : start_idx + length] = paddle.arange(self.lmax)
            start_idx = start_idx + length
            for mval in range(1, self.mmax + 1):
                length = 2 * (self.lmax + 1 - mval)
                expand_index[start_idx : start_idx + length] = paddle.cat(
                    [
                        paddle.arange(mval - 1, self.lmax),
                        paddle.arange(mval - 1, self.lmax),
                    ]
                )
                start_idx = start_idx + length
        else:
            start_idx = 0
            for lval in range(1, self.lmax + 1):
                length = min(2 * lval + 1, 2 * self.mmax + 1)
                expand_index[start_idx : start_idx + length] = lval - 1
                start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index, persistent=False)
        self.scalar_act = paddle.nn.SiLU()
        self.gate_act = paddle.nn.Sigmoid()

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """
        gating_scalars = self.gate_act(gating_scalars).view(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = paddle.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )
        input_tensors_scalars, input_tensors_vectors = input_tensors.split(
            (1, input_tensors.shape[1] - 1), 1
        )
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)
        input_tensors_vectors = input_tensors_vectors * gating_scalars
        return paddle.cat((input_tensors_scalars, input_tensors_vectors), dim=1)


class S2Activation(paddle.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = paddle.nn.SiLU()
        self.SO3_grid = SO3_grid

    def forward(self, inputs):
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        x_grid = paddle.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        return paddle.einsum("bai, zbac -> zic", from_grid_mat, x_grid)


class SeparableS2Activation(paddle.nn.Module):
    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = paddle.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax, SO3_grid)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return paddle.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )


class S2Activation_M(paddle.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int, SO3_grid, to_m: paddle.Tensor) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = paddle.nn.SiLU()
        self.SO3_grid = SO3_grid
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        to_grid_mat_m = paddle.einsum("ji,bai->jba", to_m, to_grid_mat)
        self.register_buffer("to_grid_mat_m", to_grid_mat_m, persistent=False)
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        from_grid_mat_m = paddle.einsum("ji,bai->baj", to_m, from_grid_mat)
        self.register_buffer("from_grid_mat_m", from_grid_mat_m, persistent=False)

    def forward(self, inputs):
        x_grid = paddle.einsum("iba, zic -> zbac", self.to_grid_mat_m, inputs)
        x_grid = self.act(x_grid)
        return paddle.einsum("bai, zbac -> zic", self.from_grid_mat_m, x_grid)


class SeparableS2Activation_M(paddle.nn.Module):
    def __init__(self, lmax: int, mmax: int, SO3_grid, to_m) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = paddle.nn.SiLU()
        self.s2_act = S2Activation_M(self.lmax, self.mmax, SO3_grid, to_m)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return paddle.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )
