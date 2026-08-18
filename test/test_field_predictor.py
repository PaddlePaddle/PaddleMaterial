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

import numpy as np

from ppmat.datasets.build_field import BuildField
from ppmat.predictor.field_predictor import read_cube_density
from ppmat.utils.io import write_cube


def test_read_cube_density_normalizes_geometry_to_angstrom(tmp_path):
    shape = np.asarray([2, 3, 4])
    cell = np.diag([4.0, 6.0, 8.0])
    origin = np.asarray([-1.0, -2.0, -3.0])
    atom_numbers = np.asarray([6, 1, 1])
    atom_coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5541, 0.7996, 0.4965],
            [-0.7782, -0.3735, 0.6692],
        ]
    )
    density = np.arange(np.prod(shape), dtype=np.float32)
    cube_path = tmp_path / "density.cube"
    write_cube(
        cube_path,
        atom_numbers,
        atom_coord,
        density,
        {
            "shape": shape,
            "cell": cell,
            "origin": origin,
            "coordinate_unit": "angstrom",
        },
    )

    actual_density, actual_grid_coord, info = read_cube_density(
        cube_path,
        BuildField(format="cube", name="density"),
    )
    expected_grid = BuildField.build_grid_one(
        {
            "shape": shape,
            "voxel_vectors": cell / shape[:, None],
            "origin": origin,
        },
        "angstrom",
    )

    assert info["coordinate_unit"] == "angstrom"
    np.testing.assert_array_equal(info["atom_numbers"], atom_numbers)
    np.testing.assert_allclose(info["atom_coord_ref"], atom_coord, atol=1e-5)
    np.testing.assert_allclose(info["cell"], cell, atol=1e-5)
    np.testing.assert_allclose(info["origin"], origin, atol=1e-5)
    np.testing.assert_allclose(
        actual_grid_coord,
        expected_grid.cartesian_coordinates(),
        atol=1e-5,
    )
    np.testing.assert_allclose(actual_density, density, atol=1e-5)
