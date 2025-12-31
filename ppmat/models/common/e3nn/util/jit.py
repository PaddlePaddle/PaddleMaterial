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

import inspect

import paddle

_E3NN_COMPILE_MODE = "__e3nn_compile_mode__"
_VALID_MODES = "trace", "script", "unsupported", None


def compile_mode(mode: str):
    if mode not in _VALID_MODES:
        raise ValueError("Invalid compile mode")

    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, paddle.nn.Layer)):
            raise TypeError(
                "@e3nn.util.jit.compile_mode can only decorate classes derived from paddle.nn.Layer"
            )
        setattr(obj, _E3NN_COMPILE_MODE, mode)
        return obj

    return decorator


def get_compile_mode(mod: paddle.nn.Layer) -> str:
    if hasattr(mod, _E3NN_COMPILE_MODE):
        mode = getattr(mod, _E3NN_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _E3NN_COMPILE_MODE, None)
    assert mode in _VALID_MODES, "Invalid compile mode `%r`" % mode
    return mode


def compile(mod: paddle.nn.Layer, **kwargs):
    return mod


def script(mod: paddle.nn.Layer, **kwargs):
    return mod


def trace(mod: paddle.nn.Layer, **kwargs):
    return mod