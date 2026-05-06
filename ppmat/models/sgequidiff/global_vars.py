"""空间群全局预计算变量模块。"""
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import paddle
from pymatgen.symmetry.groups import SpaceGroup as _PymatgenSpaceGroup
from scipy.spatial import ConvexHull

from ppmat.models.sgequidiff.constants import MAX_WYCKOFF_SITES
from ppmat.models.sgequidiff.spacegroup_data import spgroup_data

_THIS_FILE = Path(__file__).resolve()

_MODULE_DIR = _THIS_FILE.parent

_ENV_DATA_DIR_STR = os.getenv("SGEQUI_DATA_DIR")
if _ENV_DATA_DIR_STR:
    _ENV_DATA_DIR = Path(_ENV_DATA_DIR_STR)
    if _ENV_DATA_DIR.exists():
        DATA_DIRECTORY = _ENV_DATA_DIR
    else:
        print(f"Warning: SGEQUI_DATA_DIR '{_ENV_DATA_DIR}' does not exist, trying fallback paths")
else:
    _CANDIDATE_DIRS = [
        _MODULE_DIR / "resources",
    ]
    
    _TARGET_FILE = "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"
    DATA_DIRECTORY = None

    for _d in _CANDIDATE_DIRS:
        if (_d / _TARGET_FILE).exists():
            DATA_DIRECTORY = _d
            break

    if DATA_DIRECTORY is None:
        _candidate_paths = "\n".join([f"  • {d}" for d in _CANDIDATE_DIRS])
        raise FileNotFoundError(
            f"Cannot find SGEQUI data directory.\n"
            f"\nTried paths:\n{_candidate_paths}\n"
            f"\nSolutions:\n"
            f"  1. Run setup script: python ppmat/models/sgequidiff/setup_data.py\n"
            f"  2. Copy data to module directory: {str(_CANDIDATE_DIRS[0])}\n"
            f"  3. Set environment variable: export SGEQUI_DATA_DIR=/your/data/path\n"
            f"  4. Check if data exists in the directories above\n"
            f"\nRequired file: {_TARGET_FILE}"
        )

from pathlib import Path as _Path

def _resolve_data_dir() -> _Path:
    if DATA_DIRECTORY is None:
        raise RuntimeError("DATA_DIRECTORY was not initialized properly")
    return DATA_DIRECTORY

ASU_DICT_PATH: str = (_resolve_data_dir() / "wyckoff_positions/clean_wyckoffs_in_asu_v6.json").as_posix()

SHAPE_DECOMP_DICT_PATH: Path = _resolve_data_dir() / "wyckoff_shape_decomposition.pkl"

def _ensure_wyckoff_shape_decomp() -> None:
    """确保 wyckoff_shape_decomposition.pkl 存在。"""
    if SHAPE_DECOMP_DICT_PATH.exists():
        return
    from ppmat.models.sgequidiff.wyckoff_shape_decomp_builder import (
        build_wyckoff_shape_decomposition_dict,
    )
    build_wyckoff_shape_decomposition_dict(str(SHAPE_DECOMP_DICT_PATH), ASU_DICT_PATH)

embedding_tools = None

def string_to_fraction(string: str) -> Fraction:
    return Fraction(string)

def load_dictionary_of_wyckoff_sites_in_asus(
    json_filepath: str = ASU_DICT_PATH,
) -> dict:
    """加载非对称单元内 Wyckoff 位置字典。"""
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

asu_wyckoff_dict = load_dictionary_of_wyckoff_sites_in_asus(ASU_DICT_PATH)

padded_general_wyckoff_matrices = paddle.zeros([230, 192, 3, 3])
padded_inverse_general_wyckoff_matrices = paddle.zeros([230, 192, 3, 3])
padded_general_wyckoff_translations = paddle.zeros([230, 192, 1, 3])
padded_general_wyckoff_ops_mask = paddle.zeros([230, 192], dtype=paddle.bool)

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
        tensor_wyckoff_rotations.append(rotation.T)  # (3, 3)
        tensor_wyckoff_inv_rotations.append(paddle.linalg.inv(rotation).T)
        tensor_wyckoff_translations.append(
            paddle.to_tensor(
                symmetry_rep.translation_vector, dtype=paddle.float32
            ).unsqueeze(0)
        )  # (1, 3)

    tensor_wyckoff_rotations = paddle.stack(tensor_wyckoff_rotations, axis=0)
    # (multiplicity, 3, 3)
    tensor_wyckoff_inv_rotations = paddle.stack(tensor_wyckoff_inv_rotations, axis=0)
    tensor_wyckoff_translations = paddle.stack(tensor_wyckoff_translations, axis=0)
    # (multiplicity, 1, 3)

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
        padded_general_wyckoff_ops_mask[space_group_number - 1][0].item() == True
    )

conventional_to_primitive_transforms: dict = {
    "cP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
    "tP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
    "hP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
    "oP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
    "mP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
    "cF": {
        "P": paddle.to_tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=paddle.float32
        ),
    },
    "oF": {
        "P": paddle.to_tensor(
            [[-0.5, -0.5, 0.0], [-0.5, 0.0, -0.5], [0.0, -0.5, -0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]], dtype=paddle.float32
        ),
    },
    "cI": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=paddle.float32
        ),
    },
    "tI": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=paddle.float32
        ),
    },
    "oI": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, -0.5, 0.5]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 2.0]], dtype=paddle.float32
        ),
    },
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
    "aP": {
        "P": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
        "invP": paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32
        ),
    },
}

conventional_to_primitive_P_matrices = []
conventional_to_primitive_invP_matrices = []
for space_group_number in range(1, 231):
    bravais_lattice_string = spgroup_data[space_group_number][0]
    conventional_to_primitive_P_matrices.append(
        conventional_to_primitive_transforms[bravais_lattice_string]["P"]
    )
    conventional_to_primitive_invP_matrices.append(
        conventional_to_primitive_transforms[bravais_lattice_string]["invP"]
    )
conventional_to_primitive_P_matrices = paddle.stack(
    conventional_to_primitive_P_matrices, axis=0
)  # (230, 3, 3)
conventional_to_primitive_invP_matrices = paddle.stack(
    conventional_to_primitive_invP_matrices, axis=0
)  # (230, 3, 3)

wyckoff_dimension_tensor = -1 * paddle.ones([230, 27], dtype=paddle.int64)
for space_group_number in range(1, 231):
    sorted_wyckoff_letters = asu_wyckoff_dict[str(space_group_number)][
        "ordered_wyckoff_letters"
    ]
    for wyckoff_index, wyckoff_letter in enumerate(sorted_wyckoff_letters):
        wyckoff_dim = asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]["dim"]
        wyckoff_dimension_tensor[space_group_number - 1, wyckoff_index] = wyckoff_dim

def _project_onto_1d_subspace(line: paddle.Tensor) -> paddle.Tensor:
    """投影到 1D 子空间，返回 (3,3) 投影矩阵。"""
    projection_matrix = (line.T @ line) / (line ** 2).sum()
    return projection_matrix

def _project_onto_2d_subspace(
    facet_vertices: paddle.Tensor, return_plane_normal: bool = False
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    """投影到 2D 子空间，返回 (3,3) 投影矩阵。"""
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

num_space_groups = 230
max_wyckoffs_per_space_group = 27
max_shapes_per_wyckoff = 4

noise_projection_matrices = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3, 3]
)
wyckoff_shape_volumes = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff]
)
point_per_1d_wyckoff_line = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3]
)
line_directions_of_1d_wyckoffs = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3]
)
point_per_2d_wyckoff_plane = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3]
)
plane_normals_of_2d_wyckoffs = paddle.zeros(
    [num_space_groups, max_wyckoffs_per_space_group, max_shapes_per_wyckoff, 3]
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
                projection_matrix = _project_onto_1d_subspace(line)  # (3, 3)
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

max_simplicial_hull_facets = 16

# 初始化使 inside test 永远为 True 的 padding
asu_hull_equations = -1.0 * paddle.nn.functional.one_hot(
    paddle.to_tensor(3), num_classes=4
).unsqueeze(0).unsqueeze(0).expand([230, max_simplicial_hull_facets, 4]).cast(paddle.float32)
asu_hull_equations_mask = paddle.zeros([230, max_simplicial_hull_facets], dtype=paddle.bool)

for sg in range(1, 231):
    sg_dict = asu_wyckoff_dict[str(sg)]
    general_wyckoff_letter = sg_dict["ordered_wyckoff_letters"][-1]
    general_wyckoff_dict = sg_dict[general_wyckoff_letter]

    assert general_wyckoff_dict["dim"] == 3
    asu_vertices: np.ndarray = general_wyckoff_dict["vertices"]

    hull = ConvexHull(asu_vertices.astype("float64"))
    equations = hull.equations  # (n_facet, ndim+1)
    n_simplicial_facets = equations.shape[0]

    asu_hull_equations[sg - 1, :n_simplicial_facets] = paddle.to_tensor(
        equations, dtype=paddle.float32
    )
    asu_hull_equations_mask[sg - 1, :n_simplicial_facets] = True

asu_hull_equations_numpy = asu_hull_equations.numpy()

max_vertices_per_wyckoff_shape = 10

wyckoff_shape_vertices = paddle.zeros(
    [230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, max_vertices_per_wyckoff_shape, 3],
    dtype=paddle.float32,
)
mask_wyckoff_shape_vertices = paddle.zeros(
    [230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff, max_vertices_per_wyckoff_shape],
    dtype=paddle.bool,
)
n_vertices_per_wyckoff_shape = paddle.zeros(
    [230, MAX_WYCKOFF_SITES, max_shapes_per_wyckoff], dtype=paddle.int64
)
n_shapes_per_wyckoff = paddle.zeros([230, MAX_WYCKOFF_SITES], dtype=paddle.int64)

for sg in range(1, 231):
    sg_dict = asu_wyckoff_dict[str(sg)]
    for wp_idx, wyckoff_letter in zip(
        range(MAX_WYCKOFF_SITES), sg_dict["ordered_wyckoff_letters"]
    ):
        wyck_dict = sg_dict[wyckoff_letter]
        wyck_dof = int(wyck_dict["dim"])
        if wyck_dof not in [1, 2]:
            n_vertices = wyck_dict["vertices_tensor"].shape[0]
            n_vertices_per_wyckoff_shape[sg - 1, wp_idx, 0] = n_vertices
            n_shapes_per_wyckoff[sg - 1, wp_idx] = 1
            wyckoff_shape_vertices[sg - 1, wp_idx, 0, :wyck_dict["vertices"].shape[0], :] = (
                wyck_dict["vertices_tensor"]
            )
            mask_wyckoff_shape_vertices[sg - 1, wp_idx, 0, :n_vertices] = True
        else:
            n_shapes = len(wyck_dict["vertices"])
            n_shapes_per_wyckoff[sg - 1, wp_idx] = n_shapes
            assert 0 <= n_shapes <= max_shapes_per_wyckoff
            if wyck_dof == 1:
                n_vertices = 2
                n_vertices_per_wyckoff_shape[sg - 1, wp_idx, :n_shapes] = n_vertices
                wyckoff_shape_vertices[sg - 1, wp_idx, :n_shapes, :n_vertices, :] = (
                    wyck_dict["vertices_tensor"]
                )
                mask_wyckoff_shape_vertices[sg - 1, wp_idx, :n_shapes, :n_vertices] = True
            elif wyck_dof == 2:
                for shape_idx in range(n_shapes):
                    n_vertices = len(wyck_dict["vertices"][shape_idx])
                    n_vertices_per_wyckoff_shape[sg - 1, wp_idx, shape_idx] = n_vertices
                    wyckoff_shape_vertices[sg - 1, wp_idx, shape_idx, :n_vertices, :] = (
                        wyck_dict["vertices_tensors"][shape_idx]
                    )
                    mask_wyckoff_shape_vertices[sg - 1, wp_idx, shape_idx, :n_vertices] = True
