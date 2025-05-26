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
import paddle


class ScaledSiLU(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = paddle.nn.Silu()

    def forward(self, x: paddle.Tensor):
        return self._activation(x) * self.scale_factor


class SiQU(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._activation = paddle.nn.Silu()

    def forward(self, x: paddle.Tensor):
        return x * self._activation(x)
