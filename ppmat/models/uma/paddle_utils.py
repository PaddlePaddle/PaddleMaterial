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

import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _Tensor_min(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_min", _Tensor_min)

def _Tensor_split(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, "split", _Tensor_split)

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)
############################## 相关utils函数，如上 ##############################

# Torch-style Linear compatibility.
if not hasattr(paddle.nn.Linear, "in_features"):
    setattr(
        paddle.nn.Linear,
        "in_features",
        property(lambda self: int(self.weight.shape[0])),
    )
if not hasattr(paddle.nn.Linear, "out_features"):
    setattr(
        paddle.nn.Linear,
        "out_features",
        property(lambda self: int(self.weight.shape[1])),
    )

# Accept torch-style `bias=` kwarg in Paddle Linear.
_orig_linear_init = paddle.nn.Linear.__init__


def _linear_init_compat(self, *args, **kwargs):
    if "bias" in kwargs and "bias_attr" not in kwargs:
        kwargs["bias_attr"] = kwargs.pop("bias")
    return _orig_linear_init(self, *args, **kwargs)


paddle.nn.Linear.__init__ = _linear_init_compat


def _tensor_mul_(self, other):
    if not paddle.is_tensor(other):
        other = paddle.to_tensor(other, dtype=self.dtype)
    return self.multiply_(other)


setattr(paddle.Tensor, "mul_", _tensor_mul_)


def _drop_place_device(kwargs):
    kwargs.pop("place", None)
    kwargs.pop("device", None)
    return kwargs


_orig_zeros = paddle.zeros
_orig_ones = paddle.ones
_orig_full = paddle.full
_orig_arange = paddle.arange
_orig_tensor = paddle.tensor


def _zeros_compat(*args, **kwargs):
    return _orig_zeros(*args, **_drop_place_device(kwargs))


def _ones_compat(*args, **kwargs):
    return _orig_ones(*args, **_drop_place_device(kwargs))


def _full_compat(*args, **kwargs):
    return _orig_full(*args, **_drop_place_device(kwargs))


def _arange_compat(*args, **kwargs):
    return _orig_arange(*args, **_drop_place_device(kwargs))


def _tensor_compat(*args, **kwargs):
    return _orig_tensor(*args, **_drop_place_device(kwargs))


paddle.zeros = _zeros_compat
paddle.ones = _ones_compat
paddle.full = _full_compat
paddle.arange = _arange_compat
paddle.tensor = _tensor_compat
