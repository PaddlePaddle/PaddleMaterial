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
import paddle

from ppmat.utils.scatter import scatter_argmax
from ppmat.utils.scatter import scatter_argmin
from ppmat.utils.scatter import scatter_min_indices
from ppmat.utils.scatter import scatter_min_with_argmin


def test_scatter_argmin_handles_unsorted_and_empty_groups():
    values = paddle.to_tensor([3.0, -2.0, 4.0, -5.0, 1.0])
    groups = paddle.to_tensor([2, 0, 2, 0, 2], dtype="int64")

    result = scatter_argmin(values, groups, dim_size=4)

    np.testing.assert_array_equal(result.numpy(), [3, -1, 4, -1])


def test_scatter_argmin_selects_first_value_on_ties():
    values = paddle.to_tensor([2.0, 1.0, 1.0, 3.0])
    groups = paddle.to_tensor([0, 0, 0, 1], dtype="int64")

    result = scatter_argmin(values, groups)

    np.testing.assert_array_equal(result.numpy(), [1, 3])


def test_scatter_argmin_handles_empty_input():
    values = paddle.empty([0], dtype="float32")
    groups = paddle.empty([0], dtype="int64")

    result = scatter_argmin(values, groups, dim_size=3)

    np.testing.assert_array_equal(result.numpy(), [-1, -1, -1])


def test_scatter_argmax_selects_last_value_on_ties():
    values = paddle.to_tensor([1.0, 3.0, 3.0, 2.0])
    groups = paddle.to_tensor([0, 0, 0, 1], dtype="int64")

    result = scatter_argmax(values, groups)

    # ties resolved by last occurrence: group 0 max 3.0 at src index 2
    np.testing.assert_array_equal(result.numpy(), [2, 3])


def test_scatter_argmax_handles_empty_groups():
    values = paddle.to_tensor([5.0, 1.0])
    groups = paddle.to_tensor([0, 2], dtype="int64")

    result = scatter_argmax(values, groups, dim_size=4)

    np.testing.assert_array_equal(result.numpy(), [0, 0, 1, 0])


def test_scatter_min_with_argmin_returns_min_and_index():
    values = paddle.to_tensor([4.0, 1.0, 3.0, 2.0, 0.5])
    groups = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64")

    min_values, argmin = scatter_min_with_argmin(values, groups, dim_size=2)

    np.testing.assert_allclose(min_values.numpy(), [1.0, 0.5])
    np.testing.assert_array_equal(argmin.numpy(), [1, 4])


def test_scatter_min_with_argmin_pads_to_dim_size():
    values = paddle.to_tensor([4.0, 1.0])
    groups = paddle.to_tensor([0, 0], dtype="int64")

    min_values, argmin = scatter_min_with_argmin(values, groups, dim_size=4)

    # empty groups get +inf min value and dim_size sentinel argmin
    np.testing.assert_allclose(min_values.numpy(), [1.0, float("inf"), float("inf"), float("inf")])
    np.testing.assert_array_equal(argmin.numpy(), [1, 4, 4, 4])


def test_scatter_min_with_argmin_handles_empty_input():
    values = paddle.empty([0], dtype="float32")
    groups = paddle.empty([0], dtype="int64")

    min_values, argmin = scatter_min_with_argmin(values, groups, dim_size=3)

    np.testing.assert_allclose(min_values.numpy(), [float("inf")] * 3)
    np.testing.assert_array_equal(argmin.numpy(), [3, 3, 3])


def test_scatter_min_indices_maps_rows_to_min_column():
    # row 0 has overlapping cols {2, 1}; row 2 has {4, 3}
    ov_row = paddle.to_tensor([0, 0, 2, 2], dtype="int64")
    ov_col = paddle.to_tensor([2, 1, 4, 3], dtype="int64")

    result = scatter_min_indices(ov_row, ov_col, n_total=4)

    # rows without overlaps keep their own index (0, 1, 3 -> 3 keeps itself)
    np.testing.assert_array_equal(result.numpy(), [1, 1, 3, 3])


def test_scatter_min_indices_handles_empty_rows():
    ov_row = paddle.empty([0], dtype="int64")
    ov_col = paddle.empty([0], dtype="int64")

    result = scatter_min_indices(ov_row, ov_col, n_total=3)

    np.testing.assert_array_equal(result.numpy(), [0, 1, 2])
