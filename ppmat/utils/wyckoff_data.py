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

"""Wyckoff site / space group precomputed data for the asymmetric unit (ASU)."""

import json
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
from pymatgen.symmetry.groups import SpaceGroup as _PymatgenSpaceGroup
from scipy.spatial import ConvexHull

from ppmat.utils.asu_data import resolve_asu_data_dir
from ppmat.utils.crystal import MAX_WYCKOFF_POSITIONS
from ppmat.utils.crystal import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.utils.crystal import spgroup_data

_DATA_DIRECTORY = None


def _resolve_data_dir() -> Path:
    global _DATA_DIRECTORY
    if _DATA_DIRECTORY is None:
        _DATA_DIRECTORY = resolve_asu_data_dir()
    return _DATA_DIRECTORY


def _ensure_wyckoff_shape_decomp() -> None:
    """Ensure wyckoff_shape_decomposition.pkl exists."""
    shape_decomp_path = _resolve_data_dir() / "wyckoff_shape_decomposition.pkl"
    if shape_decomp_path.exists():
        return
    from ppmat.utils.wyckoff_shape_decomp import (
        build_wyckoff_shape_decomposition_dict,
    )
    asu_dict_path = (
        _resolve_data_dir() / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
    ).as_posix()
    build_wyckoff_shape_decomposition_dict(str(shape_decomp_path), asu_dict_path)


def string_to_fraction(string: str) -> Fraction:
    return Fraction(string)


def load_dictionary_of_wyckoff_sites_in_asus(
    json_filepath: Optional[str] = None,
) -> dict:
    """Load Wyckoff site dictionary within ASU."""
    if json_filepath is None:
        json_filepath = (
            _resolve_data_dir() / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
        ).as_posix()
    try:
        with open(json_filepath) as file:
            wyckoffs_dict = json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON format error in file {json_filepath}:\n{str(e)}\n"
            f"Please check if data file is complete.",
            doc=e.doc,
            pos=e.pos,
        ) from e

    convert_string_array_to_fractions = np.vectorize(string_to_fraction)
    for space_group_number in wyckoffs_dict.keys():
        for wyckoff_letter in wyckoffs_dict[space_group_number]["ordered_wyckoff_letters"]:
            wyckoff_site_dict = wyckoffs_dict[space_group_number][wyckoff_letter]
            wyckoff_dof = int(wyckoff_site_dict["dim"])

            if wyckoff_dof != 2:
                wyckoff_site_dict["vertices"] = convert_string_array_to_fractions(
                    wyckoff_site_dict["vertices"]
                )
                wyckoff_site_dict["vertices_tensor"] = paddle.to_tensor(
                    wyckoff_site_dict["vertices"].astype("float32"),
                    dtype=paddle.float32,
                )
            else:
                all_faces: List[np.ndarray] = []
                all_faces_tensors = []
                for face in wyckoff_site_dict["vertices"]:
                    face_array = convert_string_array_to_fractions(face)  # (n_face_vertices, 3)
                    all_faces.append(face_array)
                    all_faces_tensors.append(
                        paddle.to_tensor(face_array.astype("float32"), dtype=paddle.float32)
                    )
                wyckoff_site_dict["vertices"] = all_faces
                wyckoff_site_dict["vertices_tensors"] = all_faces_tensors
            wyckoff_site_dict["dim"] = int(wyckoff_site_dict["dim"])

            if wyckoff_site_dict["dim"] == 2:
                wyckoff_site_dict["plane_coefficients"] = convert_string_array_to_fractions(
                    wyckoff_site_dict["plane_coefficients"]
                ).tolist()

    return wyckoffs_dict


def _project_onto_1d_subspace(line: paddle.Tensor) -> paddle.Tensor:
    """Project to 1D subspace, return (3,3) projection matrix."""
    projection_matrix = (line.T @ line) / (line ** 2).sum()
    return projection_matrix


def _project_onto_2d_subspace(
    facet_vertices: paddle.Tensor, return_plane_normal: bool = False
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    """Project to 2D subspace, return (3,3) projection matrix."""
    ab = facet_vertices[0] - facet_vertices[1]
    bc = facet_vertices[1] - facet_vertices[2]
    plane_normal = paddle.linalg.cross(ab, bc)

    v1 = paddle.linalg.cross(ab, plane_normal).reshape([1, 3])
    v2 = ab.reshape([1, 3])

    projection_matrix = (
        (v1.T @ v1) / (v1 ** 2).sum() + (v2.T @ v2) / (v2 ** 2).sum()
    )
    if return_plane_normal:
        return projection_matrix, plane_normal
    else:
        return projection_matrix


# Max number of geometric shapes a single Wyckoff site is decomposed into
# (0D point / 1D line / 2D facet); used to pad shape-related tensors.
max_shapes_per_wyckoff = 4
# Max number of simplicial facets of an ASU convex hull; used to pad hull
# equation tensors to a fixed shape across all space groups.
max_simplicial_hull_facets = 16

_eye3 = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32)
_P_identity = _eye3.clone()
_invP_identity = _eye3.clone()

_body_centered_face = {
    "P": paddle.to_tensor(
        [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]], dtype=paddle.float32
    ),
    "invP": paddle.to_tensor(
        [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=paddle.float32
    ),
}
_body_centered = {
    "P": paddle.to_tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=paddle.float32
    ),
    "invP": paddle.to_tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=paddle.float32
    ),
}

conventional_to_primitive_transforms: dict = {
    "cP": {"P": _P_identity, "invP": _invP_identity},
    "tP": {"P": _P_identity, "invP": _invP_identity},
    "hP": {"P": _P_identity, "invP": _invP_identity},
    "oP": {"P": _P_identity, "invP": _invP_identity},
    "mP": {"P": _P_identity, "invP": _invP_identity},
    "aP": {"P": _P_identity, "invP": _invP_identity},
    "cF": _body_centered_face,
    "oF": _body_centered_face,
    "cI": _body_centered,
    "tI": _body_centered,
    "oI": _body_centered,
    "hR": {
        "P": (1.0 / 3.0)
        * paddle.to_tensor(
            [[-3.0, -3.0, 0.0], [-3.0, 0.0, 0.0], [-2.0, -1.0, -1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[0.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -3.0]], dtype=paddle.float32
        ),
    },
    "oC": {
        "P": paddle.to_tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, -1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=paddle.float32
        ),
    },
    "oA": {
        "P": paddle.to_tensor(
            [[0.0, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.5, -0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]], dtype=paddle.float32
        ),
    },
    "mC": {
        "P": paddle.to_tensor(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.5, -0.5, 0.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[0.0, -1.0, 0.0], [0.0, -1.0, -2.0], [1.0, 0.0, 0.0]], dtype=paddle.float32
        ),
    },
}

_lazy_vars = {
    'asu_wyckoff_dict', 'padded_general_wyckoff_matrices',
    'padded_inverse_general_wyckoff_matrices', 'padded_general_wyckoff_translations',
    'padded_general_wyckoff_ops_mask', 'conventional_to_primitive_P_matrices',
    'conventional_to_primitive_invP_matrices', 'wyckoff_dimension_tensor',
    'noise_projection_matrices', 'wyckoff_shape_volumes',
    'point_per_1d_wyckoff_line', 'line_directions_of_1d_wyckoffs',
    'point_per_2d_wyckoff_plane', 'plane_normals_of_2d_wyckoffs',
    'asu_hull_equations',
}
_initialized = False


def _lazy_init():
    global _initialized, asu_wyckoff_dict
    global padded_general_wyckoff_matrices, padded_inverse_general_wyckoff_matrices
    global padded_general_wyckoff_translations, padded_general_wyckoff_ops_mask
    global conventional_to_primitive_P_matrices, conventional_to_primitive_invP_matrices
    global wyckoff_dimension_tensor
    global noise_projection_matrices, wyckoff_shape_volumes
    global point_per_1d_wyckoff_line, line_directions_of_1d_wyckoffs
    global point_per_2d_wyckoff_plane, plane_normals_of_2d_wyckoffs
    global asu_hull_equations
    if _initialized:
        return
    try:
        _lazy_init_once()
    except BaseException:
        _initialized = False
        raise
    _initialized = True


def _lazy_init_once():
    global asu_wyckoff_dict
    global padded_general_wyckoff_matrices, padded_inverse_general_wyckoff_matrices
    global padded_general_wyckoff_translations, padded_general_wyckoff_ops_mask
    global conventional_to_primitive_P_matrices, conventional_to_primitive_invP_matrices
    global wyckoff_dimension_tensor
    global noise_projection_matrices, wyckoff_shape_volumes
    global point_per_1d_wyckoff_line, line_directions_of_1d_wyckoffs
    global point_per_2d_wyckoff_plane, plane_normals_of_2d_wyckoffs
    global asu_hull_equations

    asu_wyckoff_dict = load_dictionary_of_wyckoff_sites_in_asus(
        (_resolve_data_dir() / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json").as_posix()
    )

    padded_general_wyckoff_matrices = paddle.zeros([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 192, 3, 3])
    padded_inverse_general_wyckoff_matrices = paddle.zeros([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 192, 3, 3])
    padded_general_wyckoff_translations = paddle.zeros([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 192, 1, 3])
    padded_general_wyckoff_ops_mask = paddle.zeros([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 192], dtype=paddle.bool)

    for space_group_number in range(1, 231):
        _sg = _PymatgenSpaceGroup.from_int_number(space_group_number)
        _ops = list(_sg.symmetry_ops)

        _identity_idx = next(
            (
                i for i, op in enumerate(_ops)
                if np.allclose(op.rotation_matrix, np.eye(3), atol=1e-6)
                and np.allclose(op.translation_vector, np.zeros(3), atol=1e-6)
            ),
            0,
        )
        if _identity_idx != 0:
            _ops = [_ops[_identity_idx]] + [_ops[j] for j in range(len(_ops)) if j != _identity_idx]

        tensor_wyckoff_rotations = []
        tensor_wyckoff_inv_rotations = []
        tensor_wyckoff_translations = []
        for symmetry_rep in _ops:
            rotation = paddle.to_tensor(
                symmetry_rep.rotation_matrix, dtype=paddle.float32
            )
            tensor_wyckoff_rotations.append(rotation.T)
            tensor_wyckoff_inv_rotations.append(paddle.linalg.inv(rotation).T)
            tensor_wyckoff_translations.append(
                paddle.to_tensor(
                    symmetry_rep.translation_vector, dtype=paddle.float32
                ).unsqueeze(0)
            )

        tensor_wyckoff_rotations = paddle.stack(tensor_wyckoff_rotations, axis=0)
        tensor_wyckoff_inv_rotations = paddle.stack(tensor_wyckoff_inv_rotations, axis=0)
        tensor_wyckoff_translations = paddle.stack(tensor_wyckoff_translations, axis=0)

        n_ops = tensor_wyckoff_rotations.shape[0]
        padded_general_wyckoff_matrices[space_group_number - 1, :n_ops] = (
            tensor_wyckoff_rotations
        )
        padded_inverse_general_wyckoff_matrices[space_group_number - 1, :n_ops] = (
            tensor_wyckoff_inv_rotations
        )
        padded_general_wyckoff_translations[space_group_number - 1, :n_ops] = (
            tensor_wyckoff_translations
        )
        padded_general_wyckoff_ops_mask[space_group_number - 1, :n_ops] = True

        assert (
            paddle.equal_all(
                padded_general_wyckoff_matrices[space_group_number - 1, 0],
                paddle.eye(3)
            ).item()
            and
            paddle.equal_all(
                padded_general_wyckoff_translations[space_group_number - 1, 0],
                paddle.zeros([1, 3])
            ).item()
            and
            padded_general_wyckoff_ops_mask[space_group_number - 1][0].item() is True
        )

    conventional_to_primitive_P_matrices = []
    conventional_to_primitive_invP_matrices = []
    for space_group_number in range(1, 231):
        bravais_lattice_string = spgroup_data[space_group_number]
        conventional_to_primitive_P_matrices.append(
            conventional_to_primitive_transforms[bravais_lattice_string]["P"]
        )
        conventional_to_primitive_invP_matrices.append(
            conventional_to_primitive_transforms[bravais_lattice_string]["invP"]
        )
    conventional_to_primitive_P_matrices = paddle.stack(
        conventional_to_primitive_P_matrices, axis=0
    )
    conventional_to_primitive_invP_matrices = paddle.stack(
        conventional_to_primitive_invP_matrices, axis=0
    )

    wyckoff_dimension_tensor = -1 * paddle.ones(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS],
        dtype=paddle.int64,
    )

    for space_group_number in range(1, 231):
        sorted_wyckoff_letters = asu_wyckoff_dict[str(space_group_number)][
            "ordered_wyckoff_letters"
        ]
        for wyckoff_index, wyckoff_letter in enumerate(sorted_wyckoff_letters):
            wyckoff_dim = asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]["dim"]
            wyckoff_dimension_tensor[space_group_number - 1, wyckoff_index] = wyckoff_dim

    noise_projection_matrices = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff, 3, 3]
    )
    wyckoff_shape_volumes = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff]
    )
    point_per_1d_wyckoff_line = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff, 3]
    )
    line_directions_of_1d_wyckoffs = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff, 3]
    )
    point_per_2d_wyckoff_plane = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff, 3]
    )
    plane_normals_of_2d_wyckoffs = paddle.zeros(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, max_shapes_per_wyckoff, 3]
    )

    for sg_num in range(1, 231):
        sg_dict = asu_wyckoff_dict[str(sg_num)]
        wyckoff_projection_matrices = paddle.zeros([max_shapes_per_wyckoff, 3, 3])
        for i, wyckoff_letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
            wyckoff_dict = sg_dict[wyckoff_letter]

            shape_volumes: List[float] = wyckoff_dict["volumes"]
            wyckoff_shape_volumes[sg_num - 1, i, :len(shape_volumes)] = paddle.to_tensor(
                shape_volumes, dtype=paddle.float32
            )

            dim = int(wyckoff_dict["dim"])
            if dim == 0:
                wyckoff_projection_matrices[0] = paddle.zeros([3, 3])
            elif dim == 1:
                for j, line_segment in enumerate(wyckoff_dict["vertices"]):
                    line_segment = line_segment.astype("float32")
                    line = paddle.to_tensor(line_segment[1] - line_segment[0]).reshape([1, 3])
                    projection_matrix = _project_onto_1d_subspace(line)
                    wyckoff_projection_matrices[j] = projection_matrix

                    point_per_1d_wyckoff_line[sg_num - 1, i, j] = paddle.to_tensor(
                        line_segment[1]
                    ).reshape([3])
                    line_directions_of_1d_wyckoffs[sg_num - 1, i, j] = line.reshape([3])
            elif dim == 2:
                for j, facet_vertices in enumerate(wyckoff_dict["vertices"]):
                    projection_matrix, plane_normal = _project_onto_2d_subspace(
                        paddle.to_tensor(facet_vertices.astype("float32")),
                        return_plane_normal=True,
                    )
                    wyckoff_projection_matrices[j] = projection_matrix

                    point_per_2d_wyckoff_plane[sg_num - 1, i, j] = paddle.to_tensor(
                        facet_vertices.astype("float32")[0]
                    )
                    plane_normals_of_2d_wyckoffs[sg_num - 1, i, j] = plane_normal
            elif dim == 3:
                wyckoff_projection_matrices[0] = paddle.eye(3)
            else:
                raise AttributeError
            noise_projection_matrices[sg_num - 1, i] = wyckoff_projection_matrices

    asu_hull_equations = paddle.full(
        [NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, max_simplicial_hull_facets, 4],
        fill_value=0.0,
        dtype=paddle.float32,
    )
    asu_hull_equations[..., 3] = -1.0

    for sg in range(1, 231):
        sg_dict = asu_wyckoff_dict[str(sg)]
        general_wyckoff_letter = sg_dict["ordered_wyckoff_letters"][-1]
        general_wyckoff_dict = sg_dict[general_wyckoff_letter]

        assert general_wyckoff_dict["dim"] == 3
        asu_vertices: np.ndarray = general_wyckoff_dict["vertices"]

        hull = ConvexHull(asu_vertices.astype("float64"))
        equations = hull.equations
        n_simplicial_facets = equations.shape[0]

        asu_hull_equations[sg - 1, :n_simplicial_facets] = paddle.to_tensor(
            equations, dtype=paddle.float32
        )


def __getattr__(name):
    if name == "DATA_DIRECTORY":
        return _resolve_data_dir()
    if name == "ASU_DICT_PATH":
        return (
            _resolve_data_dir() / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
        ).as_posix()
    if name == "SHAPE_DECOMP_DICT_PATH":
        return _resolve_data_dir() / "wyckoff_shape_decomposition.pkl"
    if name in _lazy_vars:
        _lazy_init()
        try:
            return globals()[name]
        except KeyError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazily-initialized variables to IDEs / static analysis."""
    return sorted(set(globals().keys()) | _lazy_vars)
