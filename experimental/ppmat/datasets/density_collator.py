# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import paddle

from ppmat.models.infgcn.paddle_utils import * # noqa: F403

from .paddle_geometric_data import Batch


def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_len = max([s.shape[0] for s in sequences])
    trailing_dims = tuple(sequences[0].shape[1:])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)

    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


class DensityCollator:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def __call__(self, batch):
        g, densities, grid_coord, infos = zip(*batch)
        g = Batch.from_data_list(g)
        if self.n_samples is None:
            densities = pad_sequence(densities, batch_first=True, padding_value=-1)
            grid_coord = pad_sequence(grid_coord, batch_first=True, padding_value=0.0)
            return g, densities, grid_coord, infos
        sampled_density, sampled_grid = [], []
        for d, coord in zip(densities, grid_coord):
            idx = random.sample(range(d.shape[0]), self.n_samples)
            sampled_density.append(d[idx])
            sampled_grid.append(coord[idx])
        sampled_density = paddle.stack(x=sampled_density, axis=0)
        sampled_grid = paddle.stack(x=sampled_grid, axis=0)
        return g, sampled_density, sampled_grid, infos


class DensityVoxelCollator:
    def __call__(self, batch):
        g, densities, grid_coord, infos = zip(*batch)
        g = Batch.from_data_list(g)
        shapes = [info["shape"] for info in infos]
        max_shape = np.array(shapes).max(0)
        padded_density, padded_grid = [], []
        for den, grid, shape in zip(densities, grid_coord, shapes):
            padded_density.append(
                paddle.nn.functional.pad(
                    x=den.view(*shape),
                    pad=(
                        0,
                        max_shape[2] - shape[2],
                        0,
                        max_shape[1] - shape[1],
                        0,
                        max_shape[0] - shape[0],
                    ),
                    value=-1,
                    pad_from_left_axis=False,
                )
            )
            padded_grid.append(
                paddle.nn.functional.pad(
                    x=grid.view(*shape, 3),
                    pad=(
                        0,
                        0,
                        0,
                        max_shape[2] - shape[2],
                        0,
                        max_shape[1] - shape[1],
                        0,
                        max_shape[0] - shape[0],
                    ),
                    value=0.0,
                    pad_from_left_axis=False,
                )
            )
        densities = paddle.stack(x=padded_density, axis=0)
        grid_coord = paddle.stack(x=padded_grid, axis=0)
        return g, densities, grid_coord, infos
