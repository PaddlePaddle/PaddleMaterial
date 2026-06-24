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

"""Space group global precomputed variables and EmbeddingTools."""
import json
import os
from ppmat.utils import logger
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from pymatgen.symmetry.groups import SpaceGroup as _PymatgenSpaceGroup
from scipy.spatial import ConvexHull

from ppmat.models.sgequidiff.constants import (
    MAX_WYCKOFF_SITES,
    NUM_ELEMENTS,
    NUM_SPACE_GROUPS,
)
from ppmat.models.sgequidiff.constants import spgroup_data

_THIS_FILE = Path(__file__).resolve()

_MODULE_DIR = _THIS_FILE.parent

_ENV_DATA_DIR_STR = os.getenv("SGEQUI_DATA_DIR")
if _ENV_DATA_DIR_STR:
    _ENV_DATA_DIR = Path(_ENV_DATA_DIR_STR)
    if _ENV_DATA_DIR.exists():
        DATA_DIRECTORY = _ENV_DATA_DIR
    else:
        logger.info(f"Warning: SGEQUI_DATA_DIR '{_ENV_DATA_DIR}' does not exist, trying fallback paths")
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
        _candidate_paths = "\n".join([f"  \u2022 {d}" for d in _CANDIDATE_DIRS])
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
    """Ensure wyckoff_shape_decomposition.pkl exists."""
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
    """Load Wyckoff site dictionary within ASU."""
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
        padded_general_wyckoff_ops_mask[space_group_number - 1][0].item() == True
    )

_eye3 = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=paddle.float32)
_P_identity = _eye3.clone()
_invP_identity = _eye3.clone()

conventional_to_primitive_transforms: dict = {
    "cP": {"P": _P_identity, "invP": _invP_identity},
    "tP": {"P": _P_identity, "invP": _invP_identity},
    "hP": {"P": _P_identity, "invP": _invP_identity},
    "oP": {"P": _P_identity, "invP": _invP_identity},
    "mP": {"P": _P_identity, "invP": _invP_identity},
    "aP": {"P": _P_identity, "invP": _invP_identity},
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
)
conventional_to_primitive_invP_matrices = paddle.stack(
    conventional_to_primitive_invP_matrices, axis=0
)

wyckoff_dimension_tensor = -1 * paddle.ones([230, 27], dtype=paddle.int64)
for space_group_number in range(1, 231):
    sorted_wyckoff_letters = asu_wyckoff_dict[str(space_group_number)][
        "ordered_wyckoff_letters"
    ]
    for wyckoff_index, wyckoff_letter in enumerate(sorted_wyckoff_letters):
        wyckoff_dim = asu_wyckoff_dict[str(space_group_number)][wyckoff_letter]["dim"]
        wyckoff_dimension_tensor[space_group_number - 1, wyckoff_index] = wyckoff_dim

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

# initialize padding so inside test is always True
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
    equations = hull.equations
    n_simplicial_facets = equations.shape[0]

    asu_hull_equations[sg - 1, :n_simplicial_facets] = paddle.to_tensor(
        equations, dtype=paddle.float32
    )
    asu_hull_equations_mask[sg - 1, :n_simplicial_facets] = True

max_vertices_per_wyckoff_shape = 10


class EmbeddingTools:
    """Element/space-group/Wyckoff embedding tools. Singleton, accessed via embedding_tools."""

    @paddle.no_grad()
    def __init__(
        self,
        space_group_embedding_json_path: Optional[str] = None,
        element_embedding_json_path: Optional[str] = None,
        wyckoff_embedding_json_path: Optional[str] = None,
        chemistry_embedding_type: str = "identity",
        device: str = "cpu",
    ):
        self.space_group_embedding_dict = None
        self.element_embedding_dict = None
        self.wyckoff_embedding_dict = None
        self.chemistry_embedding_type = chemistry_embedding_type
        self.device = device

        data_directory = DATA_DIRECTORY

        if space_group_embedding_json_path is not None:
            fp = Path(data_directory / space_group_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.space_group_embedding_dict = json.load(file)
            self.space_group_embedding_length = len(
                self.space_group_embedding_dict["1"]
            )
            self.space_group_embedding_tensor = paddle.to_tensor(
                [
                    self.space_group_embedding_dict[str(sg_num)]
                    for sg_num in range(1, 231)
                ],
                dtype=paddle.float32,
            )
        else:
            self.space_group_embedding_length = NUM_SPACE_GROUPS

        if element_embedding_json_path is not None:
            fp = Path(data_directory / element_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.element_embedding_dict = json.load(file)
            self.element_embedding_length = len(self.element_embedding_dict["0"])
            self.element_embedding_tensor = paddle.to_tensor(
                [
                    self.element_embedding_dict[str(atomic_number)]
                    for atomic_number in range(NUM_ELEMENTS + 1)
                ],
                dtype=paddle.float32,
            )
        else:
            self.element_embedding_length = NUM_ELEMENTS

        if wyckoff_embedding_json_path is not None:
            fp = Path(data_directory / wyckoff_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.wyckoff_embedding_dict = json.load(file)
            self.wyckoff_embedding_length = len(
                self.wyckoff_embedding_dict["1"]["a"]
            )

            wyckoff_emb_list = []
            n_wyckoffs_list = []
            for sg_num in range(1, 231):
                wyckoff_dict_of_sg = self.wyckoff_embedding_dict[str(sg_num)]
                letters = list(wyckoff_dict_of_sg.keys())

                wyckoff_ascii = [ord(l) for l in letters]
                wyckoff_idxs = [
                    ai - 97 if ai >= 97 else ai - 65 + 26 for ai in wyckoff_ascii
                ]
                sorted_letters = [
                    l
                    for l, _ in sorted(
                        zip(letters, wyckoff_idxs), key=lambda pair: pair[1]
                    )
                ]

                emb_array = paddle.to_tensor(
                    [wyckoff_dict_of_sg[l] for l in sorted_letters],
                    dtype=paddle.float32,
                )
                padding = paddle.zeros(
                    [MAX_WYCKOFF_SITES - len(letters), self.wyckoff_embedding_length]
                )
                wyckoff_emb_list.append(paddle.concat([emb_array, padding], axis=0))
                n_wyckoffs_list.append(len(letters))

            self.wyckoff_embedding_tensor = paddle.stack(wyckoff_emb_list, axis=0)
            self.n_wyckoffs_per_space_group = paddle.to_tensor(
                n_wyckoffs_list, dtype=paddle.int64
            )
        else:
            self.wyckoff_embedding_length = MAX_WYCKOFF_SITES

    def get_space_group_embedding(self, space_group_index: paddle.Tensor) -> paddle.Tensor:
        """Get space group embedding."""
        assert space_group_index.dtype == paddle.int64
        if self.space_group_embedding_dict is None:
            return F.one_hot(space_group_index, NUM_SPACE_GROUPS).cast(paddle.float32)
        else:
            return self.space_group_embedding_tensor[space_group_index]

    @paddle.no_grad()
    def get_element_embedding(self, atomic_number: paddle.Tensor) -> paddle.Tensor:
        """Get element embedding."""
        assert atomic_number.dtype == paddle.int64
        if self.element_embedding_dict is None:
            return F.one_hot(
                atomic_number - 1, NUM_ELEMENTS
            ).cast(paddle.float32)
        else:
            return self.element_embedding_tensor[atomic_number]

    @paddle.no_grad()
    def get_wyckoff_embedding(
        self,
        wyckoff_index: paddle.Tensor,
        space_group_index: paddle.Tensor,
    ) -> paddle.Tensor:
        """Get Wyckoff embedding by index and space group."""
        if self.wyckoff_embedding_dict is None:
            return F.one_hot(wyckoff_index, MAX_WYCKOFF_SITES).cast(paddle.float32)
        else:
            valid_mask = self.n_wyckoffs_per_space_group[space_group_index] > wyckoff_index
            assert valid_mask.all().item(), "Invalid space group-Wyckoff index pairs"
            return self.wyckoff_embedding_tensor[space_group_index, wyckoff_index, :]

def set_global_embedding_tools(
    space_group_embedding_json_path: Optional[str] = None,
    element_embedding_json_path: Optional[str] = None,
    wyckoff_embedding_json_path: Optional[str] = None,
    chemistry_embedding_type: str = "identity",
    device: str = "cpu",
) -> None:
    """Initialize and set global embedding_tools."""
    global embedding_tools
    embedding_tools = EmbeddingTools(
        space_group_embedding_json_path=space_group_embedding_json_path,
        element_embedding_json_path=element_embedding_json_path,
        wyckoff_embedding_json_path=wyckoff_embedding_json_path,
        chemistry_embedding_type=chemistry_embedding_type,
        device=device,
    )