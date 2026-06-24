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

"""ASU (Asymmetric Unit) crystal data structures and utilities."""

from typing import List, Optional, Tuple, Union

import numpy as np
import paddle

NUM_ELEMENTS = 98  # SGEquiDiff flat crystal format offset


class CartesianAtom:
    """Immutable atom data for hash-based comparison."""
    __slots__ = ('wyckoff_letter', 'element', 'cartesian_cart_coords')

    def __init__(self, wyckoff_letter, element, cartesian_cart_coords):
        self.wyckoff_letter = wyckoff_letter
        self.element = element
        self.cartesian_cart_coords = cartesian_cart_coords

    def __eq__(self, other):
        if not isinstance(other, CartesianAtom):
            return NotImplemented
        return (self.wyckoff_letter == other.wyckoff_letter and self.element == other.element
                and paddle.allclose(self.cartesian_cart_coords, other.cartesian_cart_coords, atol=0.1, rtol=0.0))

    def __str__(self):
        return f"{int(self.wyckoff_letter)}_{int(self.element)}_{self.cartesian_cart_coords.detach().cpu().numpy().round(decimals=1)}"


class ASUCrystal:
    """ASU crystal data with optional immutability for hashable operations."""

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
        _immutable: bool = False,
    ):
        assert wyckoff_indices.shape[0] == element_indices.shape[0] == conventional_frac_coords.shape[0]
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
        self._immutable = _immutable

    @classmethod
    def from_flat(cls, flat_crystal: np.ndarray):
        num_atoms: int = int(flat_crystal[0])
        space_group_number = paddle.to_tensor(flat_crystal[1].astype("int64"), dtype=paddle.int64)
        composition_space = paddle.to_tensor(flat_crystal[2: 2 + NUM_ELEMENTS], dtype=paddle.float32)
        conventional_lattice_lengths = paddle.to_tensor(flat_crystal[2 + NUM_ELEMENTS: 5 + NUM_ELEMENTS], dtype=paddle.float32)
        conventional_lattice_angles = paddle.to_tensor(flat_crystal[5 + NUM_ELEMENTS: 8 + NUM_ELEMENTS], dtype=paddle.float32)
        element_indices = paddle.to_tensor(flat_crystal[8 + NUM_ELEMENTS: 8 + NUM_ELEMENTS + num_atoms], dtype=paddle.int64)
        wyckoff_indices = paddle.to_tensor(flat_crystal[8 + NUM_ELEMENTS + num_atoms: 8 + NUM_ELEMENTS + (2 * num_atoms)], dtype=paddle.int64)
        conventional_frac_coords = paddle.to_tensor(flat_crystal[8 + NUM_ELEMENTS + (2 * num_atoms): 8 + NUM_ELEMENTS + (5 * num_atoms)].reshape(num_atoms, 3), dtype=paddle.float32)
        wyckoff_shape_indices = paddle.to_tensor(flat_crystal[8 + NUM_ELEMENTS + (5 * num_atoms): 8 + NUM_ELEMENTS + (6 * num_atoms)], dtype=paddle.int64) if len(flat_crystal) > 8 + NUM_ELEMENTS + (5 * num_atoms) else None
        return cls(space_group_number=space_group_number, composition_space=composition_space, conventional_lattice_lengths=conventional_lattice_lengths, conventional_lattice_angles=conventional_lattice_angles, element_indices=element_indices, wyckoff_indices=wyckoff_indices, conventional_frac_coords=conventional_frac_coords, wyckoff_shape_indices=wyckoff_shape_indices)

    @property
    def num_atoms(self) -> int:
        return int(self.conventional_frac_coords.shape[0])

    def to_ImmutableASUCrystal(self):
        return ImmutableASUCrystal(
            self.space_group_number, self.conventional_lattice_lengths, self.conventional_lattice_angles,
            self.element_indices, self.wyckoff_indices, self.conventional_frac_coords, self.device,
            self.wyckoff_shape_indices, self.composition_space, self.cartesian_coords,
        )


class ImmutableASUCrystal(ASUCrystal):
    """Hashable version of ASUCrystal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _immutable=True, **kwargs)
        self._num_atoms = int(self.conventional_frac_coords.shape[0])
        if isinstance(self.cartesian_coords, paddle.Tensor):
            self._atoms = [CartesianAtom(w, e, c.unsqueeze(0)) for w, e, c in zip(self.wyckoff_indices.detach(), self.element_indices.detach(), self.cartesian_coords.detach())]
        else:
            self._atoms = None

    def to_ASUCrystal(self):
        return ASUCrystal(self.space_group_number.clone(), self.conventional_lattice_lengths.clone(), self.conventional_lattice_angles.clone(), self.element_indices.clone(), self.wyckoff_indices.clone(), self.conventional_frac_coords.clone(), self.device, self.wyckoff_shape_indices.clone() if self.wyckoff_shape_indices is not None else None, self.composition_space.clone() if self.composition_space is not None else None, self.cartesian_coords)

    @property
    def num_atoms(self) -> int:
        return self._num_atoms

    def __eq__(self, other):
        if not isinstance(other, ImmutableASUCrystal):
            return NotImplemented
        if other is self:
            return True
        if int(self.space_group_number) != int(other.space_group_number) or self._num_atoms != other._num_atoms:
            return False
        if not (paddle.allclose(self.conventional_lattice_lengths, other.conventional_lattice_lengths, atol=1e-6, rtol=0.0) and paddle.allclose(self.conventional_lattice_angles, other.conventional_lattice_angles, atol=1e-6, rtol=0.0)):
            return False
        assert self._atoms is not None
        return all(a in other._atoms for a in self._atoms) and all(a in self._atoms for a in other._atoms)

    def __hash__(self):
        assert self._atoms is not None
        return hash(f"{int(self.space_group_number)}_{self.conventional_lattice_lengths.detach().numpy().round(1)}_{self.conventional_lattice_angles.detach().numpy().round(0)}_{sorted([str(a) for a in self._atoms])}")


def is_inside(
    xs: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = 1e-6,
) -> paddle.Tensor:
    """Check if points are inside a convex polytope."""
    results = xs @ hull_equations[:, :3].T < -hull_equations[:, 3][None, :] - epsilon
    return results.all(axis=-1)


def uniformly_sample_point_in_asu_wyckoff_site(
    space_group_numbers: List[str],
    wyckoff_letters: List[str],
    dictionary_of_wyckoffs_in_asu: dict,
    dictionary_of_wyckoff_shape_decompositions: dict,
    device=None,
    finished_sampling_mask: paddle.Tensor = None,
    n_samples_per_wyckoff: int = 1,
    return_sampled_wyckoff_shape_indices: bool = False,
    hull_equations_3d: Optional[paddle.Tensor] = None,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    """Uniformly sample points in ASU Wyckoff sites."""
    random_samples_in_wyckoffs = []
    sampled_wyckoff_shape_indices = []
    for i, (space_group_number, wyckoff_letter) in enumerate(
        zip(space_group_numbers, wyckoff_letters)
    ):
        wyckoff_position_dict = dictionary_of_wyckoffs_in_asu[space_group_number][wyckoff_letter]
        wyckoff_dof = int(wyckoff_position_dict["dim"])

        if wyckoff_dof == 1:
            lengths = paddle.to_tensor(
                dictionary_of_wyckoff_shape_decompositions[space_group_number][wyckoff_letter]["volumes"],
                dtype=paddle.float32,
            )
            vertices = paddle.to_tensor(wyckoff_position_dict["vertices"].astype("float32"))
            sampled_line_index = paddle.multinomial(
                lengths, num_samples=n_samples_per_wyckoff, replacement=True
            )
            vertices = vertices[sampled_line_index]
            sampled_wyckoff_shape_index = sampled_line_index

        elif wyckoff_dof == 2:
            wyckoff_shapes_decomp_dict = dictionary_of_wyckoff_shape_decompositions[space_group_number][wyckoff_letter]
            facet_areas = paddle.to_tensor(wyckoff_shapes_decomp_dict["volumes"], dtype=paddle.float32)
            sampled_facet_idxs = paddle.multinomial(facet_areas, num_samples=n_samples_per_wyckoff, replacement=True)
            max_num_triangles_per_facet = wyckoff_shapes_decomp_dict["max_triangles_per_facet"]
            _sampled_facet_triangle_areas = [
                paddle.to_tensor(wyckoff_shapes_decomp_dict["facet_triangle_areas"][int(fid)], dtype=paddle.float32)
                for fid in sampled_facet_idxs
            ]
            sampled_facet_triangle_areas = paddle.zeros([n_samples_per_wyckoff, max_num_triangles_per_facet])
            for j in range(n_samples_per_wyckoff):
                n = _sampled_facet_triangle_areas[j].shape[0]
                sampled_facet_triangle_areas[j, :n] = _sampled_facet_triangle_areas[j]
            sampled_triangle_idxs = paddle.multinomial(sampled_facet_triangle_areas, num_samples=1).squeeze(axis=1)
            vertices = paddle.stack([
                paddle.to_tensor(wyckoff_shapes_decomp_dict["facet_triangles"][int(fid)][int(tid)], dtype=paddle.float32)
                for fid, tid in zip(sampled_facet_idxs, sampled_triangle_idxs)
            ], axis=0)
            sampled_wyckoff_shape_index = sampled_facet_idxs

        else:
            vertices = wyckoff_position_dict["vertices_tensor"]
            sampled_wyckoff_shape_index = paddle.to_tensor([0], dtype=paddle.int64).expand([n_samples_per_wyckoff])

        if finished_sampling_mask is not None and finished_sampling_mask[i]:
            sample_in_wyckoff = paddle.full([n_samples_per_wyckoff, 3], -1.0, dtype=paddle.float32)
            sampled_wyckoff_shape_index = paddle.full([n_samples_per_wyckoff], -1, dtype=paddle.int64)
        else:
            h_eq = hull_equations_3d[int(space_group_number) - 1] if hull_equations_3d is not None else None
            sample_in_wyckoff = uniformly_sample_point_in_convex_shape(
                vertices=vertices, wyckoff_site_dimensionality=wyckoff_dof,
                n_samples=n_samples_per_wyckoff, hull_equations=h_eq,
            )
            assert sample_in_wyckoff is not None

        random_samples_in_wyckoffs.append(sample_in_wyckoff)
        sampled_wyckoff_shape_indices.append(sampled_wyckoff_shape_index)

    random_samples_in_wyckoffs = paddle.stack(random_samples_in_wyckoffs, axis=0)
    sampled_wyckoff_shape_indices = paddle.stack(sampled_wyckoff_shape_indices, axis=0)
    if return_sampled_wyckoff_shape_indices:
        return random_samples_in_wyckoffs, sampled_wyckoff_shape_indices
    return random_samples_in_wyckoffs


def uniformly_sample_point_in_convex_shape(
    vertices: paddle.Tensor,
    wyckoff_site_dimensionality: int,
    n_samples: int = 1,
    hull_equations: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """Uniformly sample points inside a convex shape."""
    if wyckoff_site_dimensionality == 0:
        assert vertices.shape == [1, 3]
        return vertices.expand([n_samples, 3])
    elif wyckoff_site_dimensionality == 1:
        assert list(vertices.shape) == [n_samples, 2, 3]
        samples = paddle.rand([n_samples, 1])
        ep1, ep2 = vertices[:, 0], vertices[:, 1]
        return samples * (ep2 - ep1) + ep1
    elif wyckoff_site_dimensionality == 2:
        assert list(vertices.shape) == [n_samples, 3, 3]
        r1_sqrt, r2 = paddle.rand([n_samples, 1]).sqrt(), paddle.rand([n_samples, 1])
        return (1.0 - r1_sqrt) * vertices[:, 0] + r1_sqrt * (1.0 - r2) * vertices[:, 1] + r1_sqrt * r2 * vertices[:, 2]
    elif wyckoff_site_dimensionality == 3:
        assert hull_equations is not None, "hull_equations required for 3D sampling"
        box_lower_left = paddle.min(vertices, axis=0)
        box_top_right = paddle.max(vertices, axis=0)
        accepted = []
        total = 0
        while True:
            candidate = paddle.rand([3 * n_samples, 3]) * (box_top_right - box_lower_left) + box_lower_left
            mask = is_inside(candidate, hull_equations)
            accepted.append(candidate[mask])
            total += mask.cast(paddle.int64).sum().item()
            if total >= n_samples:
                return paddle.concat(accepted, axis=0)[:n_samples]
    raise AttributeError(f"Invalid dimensionality: {wyckoff_site_dimensionality}")
