# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from ppmat.models.common.e3nn import o3


class Identity(paddle.nn.Layer):
    """Identity operation

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`

    irreps_out : `e3nn.o3.Irreps`
    """

    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        assert self.irreps_in == self.irreps_out
        output_mask = paddle.concat(
            x=[paddle.ones(shape=mul * (2 * l + 1)) for mul, (l, _p) in self.irreps_out]
        )
        self.register_buffer(name="output_mask", tensor=output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        """evaluate"""
        return features
