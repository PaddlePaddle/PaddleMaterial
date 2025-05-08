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
            data[key] = (data[key] - self.mean) / self.std
        return data


class Log10:
    """Calculates the base-10 logarithm of the data, element-wise."""

    def __init__(
        self,
        apply_keys: Optional[Tuple[str, ...]] = None,
    ):
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
            data[key] = np.log10(data[key])
        return data


class LatticePolarDecomposition:
    """Lattice Polar Decomposition"""

    def __init__(self):
        pass

    def __call__(self, data):
        lattice = data["structure_array"]["lattice"]
        lattice_symm = self.compute_lattice_polar_decomposition(lattice)
        data["structure_array"]["lattice"] = lattice_symm
        return data

    def compute_lattice_polar_decomposition(
        self, lattice_matrix: np.ndarray
    ) -> np.ndarray:

        U, S, Vh = np.linalg.svd(lattice_matrix, full_matrices=True)
        S_square = np.diag(S.squeeze())

        V = Vh.transpose(0, 2, 1)
        U = U @ Vh

        P = V @ S_square @ Vh
        P_prime = U @ P @ U.transpose(0, 2, 1)

        return P_prime
