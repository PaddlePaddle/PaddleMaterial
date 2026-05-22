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

"""Species distribution classes for Sampler module.
"""

import numpy as np
import paddle

from .abstracts import SpeciesDistribution


class MirrorSpecies(SpeciesDistribution):
    """Base distribution that mirrors the given species."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """Return clone of input species."""
        return species.detach().clone().cpu().numpy()


class UniformSpeciesDistribution(SpeciesDistribution):
    """Sample species uniformly over a range of atomic numbers."""

    def __init__(self, num_species: int) -> None:
        """Constructor of the UniformSpeciesDistribution class."""
        super().__init__()
        if num_species <= 0:
            raise ValueError("Number of species must be greater than 0.")
        self._num_species = num_species

    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """
        Sample species uniformly from the base distribution.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: paddle.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        # Sample uniformly from [0, num_species) range
        return paddle.randint(low=0, high=self._num_species, shape=species.shape).numpy()


class WeightedSpeciesDistribution(SpeciesDistribution):
    """
    Base distribution that samples species according to a learned weight distribution.

    The weights are a dict mapping from the atomic number to its weight. All other species will have
    a weight of zero. The weights will be normalized.

    :param weight_dict:
        A dictionary mapping from atomic number to weight.
    :type weight_dict: dict[int, float]
    :param num_species:
        Number of possible species.
    :type num_species: int

    :raises ValueError:
        If the number of species is less than or equal to 0.
        If there are less than 2 species with non-zero weights.
    """

    def __init__(self, weight_dict: dict[int, float], num_species: int) -> None:
        """Constructor of the WeightedSpeciesDistribution class."""
        super().__init__()
        if num_species <= 0:
            raise ValueError("Number of species must be greater than 0.")
        self._num_species = num_species

        # Convert to paddle tensor and normalize
        weights = paddle.zeros([num_species], dtype="float32")
        for atomic_number, weight in weight_dict.items():
            if atomic_number < 0 or atomic_number >= num_species:
                raise ValueError(
                    f"Atomic number {atomic_number} is out of bounds for {num_species} species."
                )
            weights[atomic_number] = weight

        if paddle.sum(weights) == 0:
            raise ValueError("Sum of weights must be greater than 0.")
        weights = weights / paddle.sum(weights)

        self._weights = weights

    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """
        Sample species according to the weighted base distribution.

        :param species:
            The atomic numbers of all atoms in the structure in a tensor of shape (number_atoms, ).
        :type species: paddle.Tensor

        :return:
            A sample of species from the base distribution in a tensor of shape (number_atoms, ).
        :rtype: np.ndarray
        """
        return paddle.multinomial(
            self._weights, num_samples=int(species.numel()), replacement=True
        ).numpy()


class MaskSpeciesDistribution(SpeciesDistribution):
    """Base distribution returning the masking token for all atoms."""

    def __init__(self, token: int = 0) -> None:
        super().__init__()
        self._token = token

    def __call__(self, species: paddle.Tensor) -> np.ndarray:
        """Return array of masking token with same length as input species."""
        return np.ones(len(species), dtype=np.int64) * self._token
