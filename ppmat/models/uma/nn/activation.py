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


class GateActivation(paddle.nn.Layer):
    """Apply scalar SiLU and equivariant sigmoid gates."""

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        m_prime: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels

        num_components = sum(
            min(2 * degree + 1, 2 * mmax + 1) for degree in range(1, lmax + 1)
        )
        expand_index = paddle.zeros([num_components], dtype="int64")
        start = 0
        if m_prime:
            expand_index[:lmax] = paddle.arange(lmax)
            start = lmax
            for order in range(1, mmax + 1):
                length = 2 * (lmax + 1 - order)
                degree_index = paddle.arange(order - 1, lmax)
                expand_index[start : start + length] = paddle.concat(
                    [degree_index, degree_index]
                )
                start += length
        else:
            for degree in range(1, lmax + 1):
                length = min(2 * degree + 1, 2 * mmax + 1)
                expand_index[start : start + length] = degree - 1
                start += length
        self.register_buffer("expand_index", expand_index, persistable=False)

    def forward(
        self,
        gates: paddle.Tensor,
        features: paddle.Tensor,
    ) -> paddle.Tensor:
        gates = paddle.nn.functional.sigmoid(gates).reshape(
            [gates.shape[0], self.lmax, self.num_channels]
        )
        gates = paddle.index_select(gates, self.expand_index, axis=1)
        scalar = paddle.nn.functional.silu(features[:, :1, :])
        vectors = features[:, 1:, :] * gates
        return paddle.concat([scalar, vectors], axis=1)
