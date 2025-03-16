# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import numpy as np
import paddle


class RBFExpansion(paddle.nn.Layer):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            name="centers",
            tensor=paddle.linspace(start=self.vmin, stop=self.vmax, num=self.bins),
        )
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / lengthscale**2

    def forward(self, distance: paddle.Tensor) -> paddle.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return paddle.exp(
            x=-self.gamma * (distance.unsqueeze(axis=1) - self.centers) ** 2
        )
