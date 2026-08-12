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

from ppmat.utils.scatter import scatter_argmin


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
