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

from typing import Optional
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


class ClipData:
    """Normalize data class."""

    def __init__(
        self,
        min: Optional[float] = None,
        max: Optional[float] = None,
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        assert min is not None or max is not None
        self.min = min
        self.max = max
        self.apply_keys = [apply_keys] if isinstance(apply_keys, str) else apply_keys

    def __call__(self, data):
        for key in self.apply_keys:
            assert key in data, f"Key {key} does not exist in data."
            if self.min is not None and data[key] < self.min:
                if isinstance(data[key], np.ndarray):
                    data[key] = np.full_like(data[key], self.min)
                else:
                    data[key] = self.min
            elif self.max is not None and data[key] > self.max:
                if isinstance(data[key], np.ndarray):
                    data[key] = np.full_like(data[key], self.max)
                else:
                    data[key] = self.max
        return data


class SelecTargetTransform:
    """Dynamically select specific dimensions or targets from the data."""

    def __init__(
        self,
        target_indices: Union[int, Tuple[int, ...]],
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        if isinstance(target_indices, int):
            target_indices = (target_indices,)
        self.target_indices = target_indices
        self.apply_keys = apply_keys

    def __call__(self, data):
        for key in self.apply_keys:
            assert key in data, f"Key {key} does not exist in data."
            target = data[key]
            if isinstance(target, np.ndarray):
                data[key] = target[..., self.target_indices]
        return data


class RemoveYTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = np.zeros((1, 0), dtype="float32")
        return data


class SelectMuTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data
