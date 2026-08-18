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

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from cvve import GridSpec

from ppmat.utils.crystal import normalize_coordinate_unit


@dataclass(frozen=True)
class BuildGrid:
    """Build a uniform three-dimensional grid.

    Args:
        format: Input format. ``"array"`` builds a grid from a mapping with
            ``shape`` and ``voxel_vectors``. ``"bounding_box"`` builds a grid
            around Cartesian coordinates.
        shape: Grid shape used by ``"bounding_box"``.
        padding: Padding added on each side of a bounding box.
        coordinate_unit: Unit used by the grid geometry.
    """

    format: str
    shape: Sequence[int] | None = None
    padding: float = 0.0
    coordinate_unit: str = "angstrom"

    def __call__(self, data: Any) -> GridSpec:
        coordinate_unit = normalize_coordinate_unit(self.coordinate_unit)

        if self.format == "array":
            if not isinstance(data, Mapping):
                raise TypeError("Array grid data must be a mapping.")
            return GridSpec(
                shape=data["shape"],
                origin=data.get("origin", np.zeros(3)),
                vectors=data["voxel_vectors"],
                length_unit=coordinate_unit,
                value_unit=data.get("value_unit", "unknown"),
                periodic=data.get("periodic", (False, False, False)),
                cell=data.get("cell"),
            )

        if self.format == "bounding_box":
            if self.shape is None:
                raise ValueError("shape is required for a bounding-box grid.")
            shape = tuple(int(size) for size in self.shape)
            if len(shape) != 3 or min(shape) <= 0:
                raise ValueError(f"Invalid grid shape: {self.shape}")
            if self.padding < 0:
                raise ValueError("padding must be non-negative.")

            coordinates = np.asarray(data, dtype=np.float32)
            if (
                coordinates.ndim != 2
                or coordinates.shape[0] == 0
                or coordinates.shape[1] != 3
            ):
                raise ValueError("Coordinates must have shape [num_atoms, 3].")
            axis_len = np.maximum(np.ptp(coordinates, axis=0), 1e-3)
            axis_len += 2 * self.padding
            origin = (coordinates.min(axis=0) + coordinates.max(axis=0) - axis_len) / 2
            return GridSpec(
                shape=shape,
                origin=origin,
                vectors=np.diag(axis_len / np.asarray(shape)),
                length_unit=coordinate_unit,
            )

        raise ValueError(f"Unsupported grid format: {self.format}")
