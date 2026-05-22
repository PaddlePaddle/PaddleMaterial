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

"""Position distribution classes for Sampler module.
"""

import numpy as np
import paddle

from .abstracts import PositionDistribution


class MirrorPosition(PositionDistribution):
    """Base distribution that mirrors the given positions."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, pos: paddle.Tensor, pos_is_fractional: bool
    ) -> tuple[np.ndarray, bool]:
        """Return clone of input positions."""
        return pos.detach().clone().cpu().numpy(), pos_is_fractional


class NormalPositionDistribution(PositionDistribution):
    """Sample fractional positions from normal distribution with given mean and std."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError("Scale must be greater than 0.")
        self._loc = loc
        self._scale = scale

    def __call__(
        self, pos: paddle.Tensor, pos_is_fractional: bool
    ) -> tuple[np.ndarray, bool]:
        """
        Sample fractional positions from the base distribution given the atomic positions of a single structure.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: paddle.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        return paddle.normal(mean=self._loc, std=self._scale, shape=pos.shape).numpy(), True


class UniformPositionDistribution(PositionDistribution):
    """
    Position base distribution that samples fractional coordinates between 0 and 1
    using a uniform distribution.
    """

    def __init__(self) -> None:
        """Constructor of the UniformPositionDistribution class."""
        super().__init__()

    def __call__(
        self, pos: paddle.Tensor, pos_is_fractional: bool
    ) -> tuple[np.ndarray, bool]:
        """
        Sample fractional positions from the base distribution given the atomic positions of a single structure.

        :param pos:
            A tensor of shape (number_atoms, 3) containing the positions of the atoms in the structure.
        :type pos: paddle.Tensor
        :param pos_is_fractional:
            Whether the input positions are in fractional coordinates.
        :type pos_is_fractional: bool

        :return:
            (A sample of positions from the base distribution in a tensor of shape (number_atoms, 3),
             Whether the sampled positions are in fractional coordinates.)
        :rtype: tuple[np.ndarray, bool]
        """
        return paddle.rand(shape=pos.shape).cast(pos.dtype).numpy(), True
