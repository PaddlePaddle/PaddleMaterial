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

from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "UnNormalize",
    "PowerData",
]

class UnNormalize:
    """UnNormalize the data.

    Args:
        mean (Union[np.ndarray, Tuple[float, ...]]): Mean of the data
        std (Union[np.ndarray, Tuple[float, ...]]): Standard deviation of the data
        apply_keys (Optional[Tuple[str, ...]]): Keys that need to be denormalized.
            If `None`, all keys will be normalized. Defaults to None.
    """

    def __init__(
        self,
        mean: Union[np.ndarray, Tuple[float, ...]],
        std: Union[np.ndarray, Tuple[float, ...]],
        apply_keys: Optional[Tuple[str, ...]] = None,
    ):

        self.mean = mean
        self.std = std
        if apply_keys is not None:
            self.apply_keys = (
                [apply_keys] if isinstance(apply_keys, str) else apply_keys
            )
        else:
            self.apply_keys = None

    def __call__(self, data):
        if self.apply_keys is None:
            apply_keys = data.keys()
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            if key not in data:
                continue
            data[key] = data[key] * self.std + self.mean
        return data


class PowerData:
    """Power Data.

    Args:
        base (Optional[int]): Base of the power. Defaults to None.
        exp (Optional[int]): Exponent of the power. Defaults to None.
        apply_keys (Optional[Tuple[str, ...]]): Keys that need to be powered.
            If `None`, all keys will be powered. Defaults to None.
    """

    def __init__(
        self,
        base: Optional[int] = None,
        exp: Optional[int] = None,
        apply_keys: Optional[Tuple[str, ...]] = None,
    ):
        assert (
            base is not None or exp is not None
        ), "Base or exponent must be specified."
        assert base is None or exp is None, "Base and exponent must be specified."

        self.base = base
        self.exp = exp
        if apply_keys is not None:
            self.apply_keys = (
                [apply_keys] if isinstance(apply_keys, str) else apply_keys
            )
        else:
            self.apply_keys = None

    def __call__(self, data):
        if self.apply_keys is None:
            apply_keys = data.keys()
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            if key not in data:
                continue
            if self.base is not None:
                data[key] = self.base ** data[key]
            else:
                data[key] = data[key] ** self.exp
        return data
