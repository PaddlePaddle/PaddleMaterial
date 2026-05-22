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

"""Corrector classes for Stochastic Interpolants.
"""

import paddle
from .abstracts import Corrector


class IdentityCorrector(Corrector):
    """Corrector that does nothing."""

    def __init__(self):
        super().__init__()

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        """Return input unchanged."""
        return x

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        """Return x_1 unchanged."""
        return x_1.clone()


class PeriodicBoundaryConditionsCorrector(Corrector):
    """Wrap coordinates to interval [min, max] with periodic boundary conditions."""

    def __init__(self, min_value: float, max_value: float) -> None:
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        if self._min_value >= self._max_value:
            raise ValueError("Minimum value must be less than maximum value.")

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        """Wrap x to [min, max] interval."""
        range_val = self._max_value - self._min_value
        return paddle.remainder(x - self._min_value, range_val) + self._min_value

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        """Return x_1 image closest to x_0 in PBC."""
        separation_vector = x_1 - x_0
        length_over_two = (self._max_value - self._min_value) / 2.0
        range_val = self._max_value - self._min_value
        shortest_separation_vector = (
            paddle.remainder(separation_vector + length_over_two, range_val)
            - length_over_two
        )
        return x_0 + shortest_separation_vector
