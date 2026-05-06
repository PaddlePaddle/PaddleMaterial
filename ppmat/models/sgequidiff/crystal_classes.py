# Copyright (C) 2026 Suzhou National Laboratory and Baidu PaddlePaddle team
# This code was jointly developed by Suzhou National Laboratory and Baidu PaddlePaddle team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import paddle

from ppmat.models.sgequidiff.constants import NUM_ELEMENTS, chemical_symbols


@dataclass
class CartesianAtom:
    wyckoff_letter: paddle.Tensor   # zero-indexed
    element: paddle.Tensor          # zero-indexed
    cartesian_cart_coords: paddle.Tensor  # shape (1, 3)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CartesianAtom):
            return NotImplemented
        return (
            self.wyckoff_letter == other.wyckoff_letter
            and self.element == other.element
            and paddle.allclose(
                self.cartesian_cart_coords,
                other.cartesian_cart_coords,
                atol=0.1,
                rtol=0.0,
            )
        )

    def __str__(self) -> str:
        string_id = (
            f"{int(self.wyckoff_letter)}_{int(self.element)}_"
            f"{self.cartesian_cart_coords.detach().cpu().numpy().round(decimals=1)}"
        )
        return string_id

class ASUCrystal:
    """
    Dataclass containing raw attributes of a crystal in the asymmetric unit (ASU).

    """

    __hash__ = None

    def __init__(
        self,
        space_group_number: paddle.Tensor,
        conventional_lattice_lengths: paddle.Tensor,
        conventional_lattice_angles: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        conventional_frac_coords: paddle.Tensor,
        device: Union[str, paddle.CUDAPlace, paddle.CPUPlace] = "cpu",
        wyckoff_shape_indices: Optional[paddle.Tensor] = None,
        composition_space: Optional[paddle.Tensor] = None,
        cartesian_coords: Optional[paddle.Tensor] = None,
    ):
        assert (
            wyckoff_indices.shape[0]
            == element_indices.shape[0]
            == conventional_frac_coords.shape[0]
        )
        self.space_group_number = space_group_number
        self.conventional_lattice_lengths = conventional_lattice_lengths
        self.conventional_lattice_angles = conventional_lattice_angles
        self.element_indices = element_indices
        self.wyckoff_indices = wyckoff_indices
        self.conventional_frac_coords = conventional_frac_coords
        self.device = device
        self.wyckoff_shape_indices = wyckoff_shape_indices
        self.cartesian_coords: Optional[paddle.Tensor] = cartesian_coords
        self.composition_space = composition_space

    @classmethod
    def from_flat(cls, flat_crystal: np.ndarray):
        num_atoms: int = int(flat_crystal[0])
        space_group_number = paddle.to_tensor(
            flat_crystal[1].astype("int64"), dtype=paddle.int64
        )
        composition_space = paddle.to_tensor(
            flat_crystal[2 : 2 + NUM_ELEMENTS], dtype=paddle.float32
        )
        conventional_lattice_lengths = paddle.to_tensor(
            flat_crystal[2 + NUM_ELEMENTS : 5 + NUM_ELEMENTS], dtype=paddle.float32
        )
        conventional_lattice_angles = paddle.to_tensor(
            flat_crystal[5 + NUM_ELEMENTS : 8 + NUM_ELEMENTS], dtype=paddle.float32
        )
        element_indices = paddle.to_tensor(
            flat_crystal[8 + NUM_ELEMENTS : 8 + NUM_ELEMENTS + num_atoms],
            dtype=paddle.int64,
        )
        wyckoff_indices = paddle.to_tensor(
            flat_crystal[
                8 + NUM_ELEMENTS + num_atoms : 8 + NUM_ELEMENTS + (2 * num_atoms)
            ],
            dtype=paddle.int64,
        )
        conventional_frac_coords = paddle.to_tensor(
            flat_crystal[
                8 + NUM_ELEMENTS + (2 * num_atoms) :
                8 + NUM_ELEMENTS + (5 * num_atoms)
            ].reshape(num_atoms, 3),
            dtype=paddle.float32,
        )
        if len(flat_crystal) > 8 + NUM_ELEMENTS + (5 * num_atoms):
            wyckoff_shape_indices = paddle.to_tensor(
                flat_crystal[
                    8 + NUM_ELEMENTS + (5 * num_atoms) :
                    8 + NUM_ELEMENTS + (6 * num_atoms)
                ],
                dtype=paddle.int64,
            )
        else:
            wyckoff_shape_indices = None

        return ASUCrystal(
            space_group_number=space_group_number,
            composition_space=composition_space,
            conventional_lattice_lengths=conventional_lattice_lengths,
            conventional_lattice_angles=conventional_lattice_angles,
            element_indices=element_indices,
            wyckoff_indices=wyckoff_indices,
            conventional_frac_coords=conventional_frac_coords,
            wyckoff_shape_indices=wyckoff_shape_indices,
        )

    @property
    def num_atoms(self) -> int:
        return int(self.conventional_frac_coords.shape[0])

    def flatten(self) -> np.ndarray:
        if self.composition_space is None:
            composition_space = paddle.zeros([NUM_ELEMENTS], dtype=paddle.float32)
            composition_space = paddle.scatter(
                composition_space,
                self.element_indices,
                paddle.ones_like(self.element_indices, dtype=paddle.float32),
            )
        else:
            composition_space = self.composition_space

        flat_crystal = [
            np.array([float(len(self.element_indices))]),
            np.array([float(self.space_group_number)]),
            composition_space.numpy(),
            self.conventional_lattice_lengths.numpy(),
            self.conventional_lattice_angles.numpy(),
            self.element_indices.numpy(),
            self.wyckoff_indices.numpy(),
            self.conventional_frac_coords.numpy().ravel(),
        ]
        if self.wyckoff_shape_indices is not None:
            flat_crystal.append(self.wyckoff_shape_indices.numpy().ravel())

        return np.concatenate(flat_crystal)

    def to(self, device: Union[str, paddle.CUDAPlace, paddle.CPUPlace] = "cpu"):
        if isinstance(self.space_group_number, paddle.Tensor):
            space_group_number = self.space_group_number
        else:
            space_group_number = self.space_group_number

        cartesian_coords = (
            self.cartesian_coords if self.cartesian_coords is None
            else self.cartesian_coords
        )
        composition_space = (
            self.composition_space if self.composition_space is None
            else self.composition_space
        )
        wyckoff_shape_indices = (
            self.wyckoff_shape_indices if self.wyckoff_shape_indices is None
            else self.wyckoff_shape_indices
        )

        return ASUCrystal(
            space_group_number=space_group_number,
            conventional_lattice_lengths=self.conventional_lattice_lengths,
            conventional_lattice_angles=self.conventional_lattice_angles,
            element_indices=self.element_indices,
            wyckoff_indices=self.wyckoff_indices,
            conventional_frac_coords=self.conventional_frac_coords,
            wyckoff_shape_indices=wyckoff_shape_indices,
            composition_space=composition_space,
            cartesian_coords=cartesian_coords,
            device=device,
        )

    def to_ImmutableASUCrystal(self):
        return ImmutableASUCrystal(
            self.space_group_number,
            self.conventional_lattice_lengths,
            self.conventional_lattice_angles,
            self.element_indices,
            self.wyckoff_indices,
            self.conventional_frac_coords,
            self.device,
            self.wyckoff_shape_indices,
            self.composition_space,
            self.cartesian_coords,
        )

    def __str__(self):
        from ppmat.models.sgequidiff import global_vars as sgequidiff_global_vars
        str_representation = (
            f"----- ASUCrystal -----\n"
            f"Space group {int(self.space_group_number)}\n"
            f"(a={float(self.conventional_lattice_lengths[0]):0.2f},"
            f"b={float(self.conventional_lattice_lengths[1]):0.2f},"
            f"c={float(self.conventional_lattice_lengths[2]):0.2f},"
            f"alpha={float(self.conventional_lattice_angles[0]):0.0f},"
            f"beta={float(self.conventional_lattice_angles[1]):0.0f},"
            f"gamma={float(self.conventional_lattice_angles[2]):0.0f})\n"
            f"-- Atoms:\n"
        )
        wyckoff_letters = [
            chr(97 + int(self.wyckoff_indices[i]))
            if int(self.wyckoff_indices[i]) <= 25
            else chr(39 + int(self.wyckoff_indices[i]))
            for i in range(self.num_atoms)
        ]
        wyckoff_dims = [
            str(
                sgequidiff_global_vars.asu_wyckoff_dict[
                    str(int(self.space_group_number))
                ][letter]["dim"]
            )
            for letter in wyckoff_letters
        ]
        atom_strings = "\n".join(
            [
                "".join(
                    [
                        chemical_symbols[int(self.element_indices[i]) + 1],
                        "\t",
                        wyckoff_letters[i] + " (" + wyckoff_dims[i] + "D)",
                        "\t",
                        str(self.conventional_frac_coords[i].numpy().tolist()),
                    ]
                )
                for i in range(self.num_atoms)
            ]
        )
        return "".join((str_representation, atom_strings, "\n"))

class ImmutableASUCrystal:
    """Immutable version of ASUCrystal for hashable operations.
    
    """

    def __init__(
        self,
        space_group_number: paddle.Tensor,
        conventional_lattice_lengths: paddle.Tensor,
        conventional_lattice_angles: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        conventional_frac_coords: paddle.Tensor,
        device: Union[str, paddle.CUDAPlace, paddle.CPUPlace] = "cpu",
        wyckoff_shape_indices: Optional[paddle.Tensor] = None,
        composition_space: Optional[paddle.Tensor] = None,
        cartesian_coords: Optional[paddle.Tensor] = None,
    ):
        assert (
            wyckoff_indices.shape[0]
            == element_indices.shape[0]
            == conventional_frac_coords.shape[0]
        )
        self._space_group_number = space_group_number
        self._conventional_lattice_lengths = conventional_lattice_lengths
        self._conventional_lattice_angles = conventional_lattice_angles
        self._element_indices = element_indices
        self._wyckoff_indices = wyckoff_indices
        self._conventional_frac_coords = conventional_frac_coords
        self._device = device
        self._num_atoms = int(self._conventional_frac_coords.shape[0])
        self._wyckoff_shape_indices = wyckoff_shape_indices
        self._composition_space = composition_space
        self._cartesian_coords: Optional[paddle.Tensor] = cartesian_coords

        if isinstance(cartesian_coords, paddle.Tensor):
            self._atoms: List[CartesianAtom] = [
                CartesianAtom(wyckoff, etype, cart_coord.unsqueeze(0))
                for wyckoff, etype, cart_coord in zip(
                    wyckoff_indices.detach(),
                    element_indices.detach(),
                    cartesian_coords.detach(),
                )
            ]
        else:
            self._atoms = None

    def to_ASUCrystal(self):
        return ASUCrystal(
            self._space_group_number.clone(),
            self._conventional_lattice_lengths.clone(),
            self._conventional_lattice_angles.clone(),
            self._element_indices.clone(),
            self._wyckoff_indices.clone(),
            self._conventional_frac_coords.clone(),
            self._device,
            self._wyckoff_shape_indices.clone()
            if self._wyckoff_shape_indices is not None
            else None,
            self._composition_space.clone()
            if self._composition_space is not None
            else None,
            self._cartesian_coords,
        )

    @property
    def space_group_number(self) -> paddle.Tensor:
        return self._space_group_number

    @property
    def conventional_lattice_lengths(self) -> paddle.Tensor:
        return self._conventional_lattice_lengths

    @property
    def conventional_lattice_angles(self) -> paddle.Tensor:
        return self._conventional_lattice_angles

    @property
    def element_indices(self) -> paddle.Tensor:
        return self._element_indices

    @property
    def wyckoff_indices(self) -> paddle.Tensor:
        return self._wyckoff_indices

    @property
    def conventional_frac_coords(self) -> paddle.Tensor:
        return self._conventional_frac_coords

    @property
    def wyckoff_shape_indices(self) -> Optional[paddle.Tensor]:
        return self._wyckoff_shape_indices

    @property
    def num_atoms(self) -> int:
        return self._num_atoms

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableASUCrystal):
            return NotImplemented
        if other is self:
            return True
        if self._space_group_number != other._space_group_number:
            return False
        if self._num_atoms != other._num_atoms:
            return False
        if not (
            paddle.allclose(
                self._conventional_lattice_lengths,
                other._conventional_lattice_lengths,
                atol=1e-6,
                rtol=0.0,
            )
            and paddle.allclose(
                self._conventional_lattice_angles,
                other._conventional_lattice_angles,
                atol=1e-6,
                rtol=0.0,
            )
        ):
            return False
        assert self._atoms is not None
        return all(atom in other._atoms for atom in self._atoms) and all(
            atom in self._atoms for atom in other._atoms
        )

    def __hash__(self) -> str:
        assert self._atoms is not None
        crystal_str = (
            f"{int(self._space_group_number)}_"
            f"{self._conventional_lattice_lengths.detach().numpy().round(1)}_"
            f"{self._conventional_lattice_angles.detach().numpy().round(0)}_"
            f"{sorted([str(atom) for atom in self._atoms])}"
        )
        return crystal_str

class CrystalDict(dict):
    """Dictionary of batched crystal tensors.
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @paddle.no_grad()
    def repeat_(self, num_repeats: int):
        for key, crystal_attribute_tensor in self.items():
            if isinstance(crystal_attribute_tensor, paddle.Tensor):
                self[key] = paddle.repeat_interleave(
                    crystal_attribute_tensor, num_repeats, axis=0
                )
        return self

    def __len__(self) -> int:
        return self["space_group_indices"].shape[0]

    def to_(self, device: str):
        self["device"] = device
        tensor_keys = [
            "space_group_indices", "batch_chemistries", "lattice_lengths",
            "lattice_angles", "n_atoms_per_asu", "element_indices", "wyckoff_indices",
        ]
        for k in tensor_keys:
            if k in self and isinstance(self[k], paddle.Tensor):
                self[k] = self[k]
        if "lattice_matrices" in self and isinstance(self["lattice_matrices"], paddle.Tensor):
            pass
        if "wyckoff_shape_indices" in self and isinstance(
            self["wyckoff_shape_indices"], paddle.Tensor
        ):
            pass
        if "frac_coords" in self and isinstance(self["frac_coords"], paddle.Tensor):
            pass
        return self
