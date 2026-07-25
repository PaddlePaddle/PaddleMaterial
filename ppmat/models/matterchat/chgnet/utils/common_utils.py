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

from __future__ import annotations

import os

import paddle

from ppmat.utils.io import read_json, write_json
from ppmat.utils.misc import AverageMeter


def mae(prediction: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Computes the mean absolute error between prediction and target."""
    return paddle.mean(paddle.abs(target - prediction))


def mkdir(path: str):
    """Make directory."""
    os.makedirs(path, exist_ok=True)
    return path
