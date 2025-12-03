# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import numbers
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import List

import numpy as np
import paddle
import pgl

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.datasets.geometric_data_type.batch import Batch


class DefaultCollator(object):
    def __call__(self, batch: List[Any]) -> Any:
        """Default_collate_fn for paddle dataloader.

        NOTE: This `default_collate_fn` is different from official `default_collate_fn`
        which specially adapt case where sample is `None` and `pgl.Graph`.

        ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/io/dataloader/collate.py#L25

        Args:
            batch (List[Any]): Batch of samples to be collated.

        Returns:
            Any: Collated batch data.
        """
        sample = batch[0]
        if sample is None:
            return None
        elif isinstance(sample, ConcatNumpyWarper):
            batch = np.concatenate(batch, axis=0)
            return batch
        elif isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, (paddle.Tensor, paddle.framework.core.eager.Tensor)):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Mapping):
            return {key: self([d[key] for d in batch]) for key in sample}
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError("Fields number not same among samples in a batch")
            return [self(fields) for fields in zip(*batch)]
        elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
            # use str(type()) instead of isinstance() in case of pgl is not installed.
            graphs = pgl.Graph.batch(batch)
            # NOTE: when num_works >1, graphs.tensor() will convert numpy.ndarray to
            # CPU Tensor, which will cause error in model training.
            # graphs.tensor()
            return graphs
        elif isinstance(sample, ConcatData):
            return ConcatData.batch(batch)
        raise TypeError(
            "batch data can only contains: paddle.Tensor, numpy.ndarray, "
            f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
        )


class DensityCollator:
    def __init__(self, n_samples=None, padding_value=-1.0):
        self.n_samples = n_samples
        self.padding_value = padding_value

    def __call__(self, batch):
        g, densities, grid_coord, infos = zip(*batch)
        g = Batch.from_data_list(g)
        infos = list(infos)
        if self.n_samples is None:
            densities = pad_sequence(
                densities, batch_first=True, padding_value=self.padding_value
            )
            grid_coord = pad_sequence(grid_coord, batch_first=True, padding_value=0.0)
            mask = (densities != self.padding_value).astype("float32")
        else:
            sampled_density, sampled_grid, mask = [], [], []
            target_samples = int(self.n_samples)
            for d, coord in zip(densities, grid_coord):
                total = int(d.shape[0])
                replace = total < target_samples
                indices = np.random.choice(total, target_samples, replace=replace)
                indices.sort()
                sampled_density.append(d[indices])
                sampled_grid.append(coord[indices])
                mask.append(
                    paddle.ones_like(x=sampled_density[-1], dtype="float32")
                )
            densities = paddle.stack(x=sampled_density, axis=0)
            grid_coord = paddle.stack(x=sampled_grid, axis=0)
            mask = paddle.stack(x=mask, axis=0)
        densities = densities * mask
        return {
            "density": densities,
            "density_mask": mask,
            "grid_coord": grid_coord,
            "graph": g,
            "infos": infos,
        }


class DensityVoxelCollator:
    def __init__(self, padding_value=-1.0):
        self.padding_value = padding_value

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
        mask = (densities != self.padding_value).astype("float32")
        densities = densities * mask
        return {
            "density": densities,
            "density_mask": mask,
            "grid_coord": grid_coord,
            "graph": g,
            "infos": list(infos),
        }

# utils DensityCollator
def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_len = max([int(s.shape[0]) for s in sequences])  # 确保转换为Python整数
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
