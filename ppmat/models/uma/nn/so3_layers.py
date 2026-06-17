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


class SO3_Linear(paddle.nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.weight = paddle.nn.Parameter(
            paddle.randn((self.lmax + 1, out_features, in_features))
        )
        bound = 1 / math.sqrt(self.in_features)
        paddle.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = paddle.nn.Parameter(paddle.zeros(out_features))
        expand_index = paddle.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : start_idx + length] = lval
        self.register_buffer("expand_index", expand_index, persistent=False)

    def forward(self, input_embedding):
        weight = paddle.index_select(self.weight, dim=0, index=self.expand_index)
        out = paddle.einsum("bmi, moi -> bmo", input_embedding, weight).contiguous()
        bias = self.bias.view(1, 1, self.out_features)
        first = out.narrow(1, 0, 1) + bias
        if out.shape[1] > 1:
            rest = out.narrow(1, 1, out.shape[1] - 1)
            out = paddle.concat([first, rest], axis=1)
        else:
            out = first
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"
