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

import paddle
from paddle import Tensor, nn


class GaussianExpansion(nn.Layer):
    """Expand distance by Gaussian basis functions."""

    def __init__(
        self,
        min: float = 0,
        max: float = 5,
        step: float = 0.5,
        var: float | None = None,
    ) -> None:
        super().__init__()
        assert min < max
        assert max - min > step
        self.register_buffer(
            "gaussian_centers", paddle.arange(min, max + step, step)
        )
        if var is None:
            var = step
        self.var = var

    def expand(self, features: Tensor) -> Tensor:
        return paddle.exp(
            -((features.reshape([-1, 1]) - self.gaussian_centers) ** 2) / self.var**2
        )
