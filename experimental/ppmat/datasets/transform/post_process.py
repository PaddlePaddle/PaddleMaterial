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

    def __call__(self, pred_data, batch_data=None):

        apply_keys = (
            self.apply_keys if self.apply_keys is not None else pred_data.keys()
        )

        for key in apply_keys:
            if key not in pred_data:
                continue
            pred_data[key] = pred_data[key] * self.std + self.mean

        if batch_data is not None:
            apply_keys = (
                self.apply_keys if self.apply_keys is not None else batch_data.keys()
            )
            for key in apply_keys:
                if key not in batch_data:
                    continue
                batch_data[key] = batch_data[key] * self.std + self.mean

        return pred_data, batch_data
