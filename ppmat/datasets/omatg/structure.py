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

"""Structure class for representing crystalline structures.
"""

from typing import Any, Optional

import paddle
from ase import Atoms
from ase.build.tools import niggli_reduce
from ase.symbols import Symbols


class Structure:
    """Storage for crystalline structure with cell, atomic numbers, coordinates, properties, and metadata.
    Supports Cartesian/fractional coordinate conversion and Niggli reduction.
    """

    def __init__(
        self,
        cell: paddle.Tensor,
        atomic_numbers: paddle.Tensor,
        pos: paddle.Tensor,
        property_dict: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        pos_is_fractional: bool = False,
    ) -> None:
        """Constructor for the Structure class."""
        assert cell.shape == (3, 3), f"cell must be 3x3, got {cell.shape}"
        assert (
            atomic_numbers.dim() == 1
        ), f"atomic_numbers must be 1D, got {atomic_numbers.dim()}D"
        assert (
            pos.shape[0] == len(atomic_numbers) and pos.shape[1] == 3
        ), f"pos must be (N, 3), got {pos.shape}"

        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._pos = pos
        self._property_dict = property_dict if property_dict is not None else {}
        self._metadata = metadata if metadata is not None else {}
        self._fractional = pos_is_fractional

    @property
    def cell(self) -> paddle.Tensor:
        """Return the cell vectors of the structure (3x3 lattice matrix)."""
        return self._cell

    @property
    def atomic_numbers(self) -> paddle.Tensor:
        """Return the atomic numbers of the atoms in the structure."""
        return self._atomic_numbers

    @property
    def symbols(self) -> list[str]:
        """Return the chemical symbols of the atoms in the structure."""
        return list(Symbols(self._atomic_numbers.numpy()))

    @property
    def pos(self) -> paddle.Tensor:
        """
        Return the Cartesian or fractional coordinates of the atoms.

        If convert_to_fractional() has been called, these coordinates will be
        fractional coordinates. Otherwise, they will be Cartesian coordinates.
        """
        return self._pos

    @property
    def pos_is_fractional(self) -> bool:
        """Return whether the atomic positions are in fractional coordinates."""
        return self._fractional

    @property
    def property_dict(self) -> dict[str, Any]:
        """Return the property dictionary of the structure."""
        return self._property_dict

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the structure."""
        return self._metadata

    def to(self, floating_point_precision: str) -> None:
        """
        Convert the floating point precision of the structure.

        :param floating_point_precision: The paddle dtype to convert to.
            Should be one of "float32", "float64", "float16", "bfloat16".
        :raises ValueError: If the given floating point precision is not supported.
        """
        valid_precisions = ["float32", "float64", "float16", "bfloat16"]
        if floating_point_precision not in valid_precisions:
            raise ValueError(
                f"Unsupported floating point precision: {floating_point_precision}. "
                f"Supported precisions are {valid_precisions}."
            )
        self._cell = self._cell.cast(floating_point_precision)
        self._pos = self._pos.cast(floating_point_precision)
        for key, value in self._property_dict.items():
            if paddle.is_tensor(value) and value.dtype in [
                paddle.float32,
                paddle.float64,
                paddle.float16,
                paddle.bfloat16,
            ]:
                self._property_dict[key] = value.cast(floating_point_precision)

    def get_ase_atoms(self) -> Atoms:
        """
        Convert the structure to an ASE Atoms object.

        :return: The ASE Atoms object.
        """
        if self._fractional:
            return Atoms(
                numbers=self.atomic_numbers.tolist(),
                scaled_positions=self.pos.numpy(),
                cell=self.cell.numpy(),
                pbc=True,
                info=self.property_dict | self.metadata,
            )
        else:
            return Atoms(
                numbers=self.atomic_numbers.tolist(),
                positions=self.pos.numpy(),
                cell=self.cell.numpy(),
                pbc=True,
                info=self.property_dict | self.metadata,
            )

    def niggli_reduce(self) -> None:
        """Niggli reduce the structure."""
        atoms = self.get_ase_atoms()
        niggli_reduce(atoms)
        self._cell = paddle.to_tensor(atoms.cell[:], dtype=self._cell.dtype)
        if self._fractional:
            self._pos = paddle.to_tensor(
                atoms.get_scaled_positions(), dtype=self._pos.dtype
            )
        else:
            self._pos = paddle.to_tensor(atoms.positions, dtype=self._pos.dtype)

    def convert_to_fractional(self) -> None:
        """
        Convert the atomic positions to fractional coordinates.
        """
        if not self._fractional:
            # Solve r = f * cell for f: f = r * cell^(-1)
            with paddle.no_grad():
                self._pos = paddle.remainder(
                    paddle.linalg.solve(self._cell, self._pos.T).T,
                    1.0,
                )
            self._fractional = True

    def convert_to_cartesian(self) -> None:
        """
        Convert the atomic positions to Cartesian coordinates.
        """
        if self._fractional:
            with paddle.no_grad():
                self._pos = paddle.matmul(self._pos, self._cell)
            self._fractional = False

    def __repr__(self) -> str:
        """Return a string representation of the structure."""
        return (
            f"Structure(cell={self._cell.shape}, "
            f"atomic_numbers={self._atomic_numbers.shape}, "
            f"pos={self._pos.shape}, "
            f"fractional={self._fractional})"
        )
