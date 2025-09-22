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
from paddle import nn


# helper functions
def exists(val):
    return val is not None


def l2norm(t):
    return nn.functional.normalize(x=t, axis=-1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def log(t, eps=1e-12):
    return paddle.log(t.clamp(min=eps))


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.stop_gradient = not requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)
