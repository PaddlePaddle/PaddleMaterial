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

from __future__ import annotations

from typing import Tuple
from typing import Union

import numpy as np


class Normalize:
    """Normalize data class."""

    def __init__(
        self,
        mean: Union[np.ndarray, Tuple[float, ...]],
        std: Union[np.ndarray, Tuple[float, ...]],
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        self.mean = mean
        self.std = std
        self.apply_keys = [apply_keys] if isinstance(apply_keys, str) else apply_keys

    def __call__(self, data):
        for key in self.apply_keys:
            assert key in data, f"Key {key} does not exist in data."
            data[key] = (data[key] - self.mean) / self.std
        return data
