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

"""Abstract classes for Sampler module.
"""

from abc import ABC, abstractmethod

import numpy as np
import paddle


class SpeciesDistribution(ABC):
    """Abstract base class for species distributions."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """Sample species from base distribution. Returns numpy array."""
        raise NotImplementedError


class CellDistribution(ABC):
    """Abstract base class for cell distributions."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, cell: paddle.Tensor) -> np.ndarray:
        """Sample cell from base distribution. Returns numpy array."""
        raise NotImplementedError


class PositionDistribution(ABC):
    """Abstract base class for position distributions (fractional or Cartesian)."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(
        self, pos: paddle.Tensor, pos_is_fractional: bool
    ) -> tuple[np.ndarray, bool]:
        """Sample positions from base distribution. Returns (positions, is_fractional)."""
        raise NotImplementedError
