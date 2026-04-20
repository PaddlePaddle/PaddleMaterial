# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

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


def get_activation_by_string(key):
    if key == "swish":
        activation = paddle.nn.SiLU()
    elif key == "silu":
        activation = paddle.nn.SiLU()
    elif key == "relu":
        activation = paddle.nn.ReLU()
    elif key == "elu":
        activation = paddle.nn.ELU()
    elif key == "leaky_relu":
        activation = paddle.nn.LeakyReLU()
    elif key == "tanh":
        activation = paddle.nn.Tanh()
    elif key == "sigmoid":
        activation = paddle.nn.Sigmoid()
    elif key == "softplus":
        activation = paddle.nn.Softplus()
    elif key == "gelu":
        activation = paddle.nn.GELU()
    elif key == "ssp":
        activation = ShiftedSoftplus()
    elif key == "swiglu":
        activation = SwiGLU()
    else:
        raise NotImplementedError("The activation function '%s' is unknown." % str(key))
    return activation


class ShiftedSoftplus(paddle.nn.Module):
    """
    Compute shifted soft-plus activation function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack
    under the MIT License.

    Notes:
        y = ln(1 + e^(-x)) - ln(2)
    """

    def __init__(self):
        super().__init__()
        self.softplus = paddle.nn.Softplus()
        self.shift = paddle.log(paddle.tensor(2.0))

    def forward(self, x):
        return self.softplus(x) - self.shift


class SwiGLU(paddle.nn.Module):
    """
    Compute swish-gated activation function.

    Notes:
        y = gate(x) * out(x) = swish(linear(x)) * linear(x)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = paddle.compat.nn.Linear(in_features, out_features)
        self.linear2 = paddle.compat.nn.Linear(in_features, out_features)
        self.gate = paddle.nn.SiLU()

    def forward(self, x):
        return self.gate(self.linear1(x)) * self.linear2(x)
