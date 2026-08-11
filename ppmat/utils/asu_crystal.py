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

import paddle


class ASUCrystal:
    """ASU crystal data."""

    __hash__ = None

    def __init__(
        self,
        space_group_number: paddle.Tensor,
        conventional_lattice_lengths: paddle.Tensor,
        conventional_lattice_angles: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        conventional_frac_coords: paddle.Tensor,
    ):
        assert wyckoff_indices.shape[0] == element_indices.shape[0] == conventional_frac_coords.shape[0]
        self.space_group_number = space_group_number
        self.conventional_lattice_lengths = conventional_lattice_lengths
        self.conventional_lattice_angles = conventional_lattice_angles
        self.element_indices = element_indices
        self.wyckoff_indices = wyckoff_indices
        self.conventional_frac_coords = conventional_frac_coords

    @property
    def num_atoms(self) -> int:
        return int(self.conventional_frac_coords.shape[0])


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
            sampled_triangle_idxs = paddle.multinomial(
                sampled_facet_triangle_areas, num_samples=1
            ).squeeze(axis=1)
            vertices = paddle.stack([
                paddle.to_tensor(
                    wyckoff_shapes_decomp_dict["facet_triangles"][int(fid)][int(tid)],
                    dtype=paddle.float32,
                )
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
        max_iters = 1000
        batch_size = max(3 * n_samples, 1)
        for _ in range(max_iters):
            candidate = (
                paddle.rand([batch_size, 3]) * (box_top_right - box_lower_left)
                + box_lower_left
            )
            mask = is_inside(candidate, hull_equations)
            accepted.append(candidate[mask])
            total += mask.cast(paddle.int64).sum().item()
            if total >= n_samples:
                return paddle.concat(accepted, axis=0)[:n_samples]
            batch_size *= 2
        raise RuntimeError(
            f"Rejection sampling failed to collect {n_samples} points inside the "
            f"3D hull after {max_iters} iterations (collected {total}); "
            "check hull_equations / bounding box validity"
        )
    raise AttributeError(f"Invalid dimensionality: {wyckoff_site_dimensionality}")
