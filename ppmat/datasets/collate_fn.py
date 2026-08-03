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
import warnings

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data


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
        elif isinstance(sample, Data):
            # Geometric `Data` objects: batch them into a single `Batch`
            return Batch.from_data_list(batch)
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


class RadiusGraphCollator:
    """Collate PGL radius graphs and offset their edge-based triplet indices."""

    def __call__(self, batch):
        graphs = [sample["graph"] for sample in batch]
        num_edges = [np.asarray(graph.edges).shape[0] for graph in graphs]
        edge_offsets = np.cumsum([0] + num_edges[:-1])

        triplet_fields = {key: [] for key in ("idx_kj", "idx_ji")}
        for index, graph in enumerate(graphs):
            for key in triplet_fields:
                triplet_fields[key].append(
                    np.asarray(
                        graph.edge_feat[f"ti_{key}"], dtype=np.int64
                    )
                    + edge_offsets[index]
                )

        graph = pgl.Graph.batch(graphs)
        graph.edge_feat.update(
            {
                f"ti_{key}": np.concatenate(values)
                for key, values in triplet_fields.items()
            }
        )

        result = DefaultCollator()(
            [
                {key: value for key, value in sample.items() if key != "graph"}
                for sample in batch
            ]
        )
        result["graph"] = graph
        return result


class DensityCollator:
    def __init__(
        self,
        n_samples=None,
        padding_value=-1.0,
        sampling_mode: str = "uniform",  # "uniform" or "random"
        uniform_random_offset: bool = False,
        sampling_seed: int | None = None,
        clip_max: float | None = None,
        importance_sampling: bool = False,
        importance_threshold: float = 1e-5,
        importance_ratio: float = 0.8,
        extreme_threshold: float | None = None,
        extreme_ratio: float = 0.05,
    ):
        self.n_samples = n_samples
        self.padding_value = padding_value
        self.sampling_mode = sampling_mode.lower()
        self.uniform_random_offset = bool(uniform_random_offset)
        self.sampling_seed = sampling_seed
        self._rng = np.random.default_rng(sampling_seed) if sampling_seed is not None else None
        self.clip_max = clip_max
        self.importance_sampling = bool(importance_sampling)
        self.importance_threshold = importance_threshold
        self.importance_ratio = float(importance_ratio)
        self.extreme_threshold = extreme_threshold
        self.extreme_ratio = float(extreme_ratio)
        self._warned_length_mismatch = False

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
                total_d = int(d.shape[0])
                total_coord = int(coord.shape[0])
                total = min(total_d, total_coord)
                if total_d != total_coord and not self._warned_length_mismatch:
                    warnings.warn(
                        f"Density length ({total_d}) and grid length "
                        f"({total_coord}) differ; truncating to {total}."
                    )
                    self._warned_length_mismatch = True
                if total == 0:
                    raise ValueError("Empty density/grid pair encountered in batch.")
                if self.importance_sampling:
                    total_idx = np.arange(total)
                    dense_vals = np.abs(d.numpy().reshape(-1))
                    threshold = float(self.importance_threshold)
                    high_mask = dense_vals >= threshold
                    high_idx = total_idx[high_mask]

                    extreme_idx = np.array([], dtype=int)
                    if self.extreme_threshold is not None:
                        extreme_mask = dense_vals >= self.extreme_threshold
                        extreme_idx = total_idx[extreme_mask]
                        # ensure extreme is subset of high
                        extreme_idx = np.intersect1d(extreme_idx, high_idx, assume_unique=True)
                    mid_idx = np.setdiff1d(high_idx, extreme_idx, assume_unique=True)

                    high_quota = min(target_samples, max(0, int(target_samples * self.importance_ratio)))
                    extreme_quota = min(target_samples, max(0, int(target_samples * self.extreme_ratio)))

                    extreme_take = min(len(extreme_idx), extreme_quota)
                    indices_extreme = (
                        np.random.choice(extreme_idx, extreme_take, replace=False)
                        if extreme_take > 0
                        else np.array([], dtype=int)
                    )

                    remaining_high = high_quota - len(indices_extreme)
                    mid_take = min(len(mid_idx), remaining_high)
                    indices_mid = (
                        np.random.choice(mid_idx, mid_take, replace=False)
                        if mid_take > 0
                        else np.array([], dtype=int)
                    )

                    selected = np.concatenate([indices_extreme, indices_mid])
                    remaining = target_samples - len(selected)
                    if remaining > 0:
                        low_candidates = np.setdiff1d(total_idx, selected, assume_unique=False)
                        if len(low_candidates) == 0:
                            low_candidates = total_idx
                        replace_low = remaining > len(low_candidates)
                        indices_low = np.random.choice(low_candidates, remaining, replace=replace_low)
                        indices = np.concatenate([selected, indices_low])
                    else:
                        indices = selected
                else:
                    if self.sampling_mode == "uniform":
                        if self.uniform_random_offset:
                            if self._rng is None:
                                self._rng = np.random.default_rng()
                            step = (total - 1) / max(target_samples - 1, 1)
                            offset = float(self._rng.uniform(0, max(step, 1.0))) if step > 0 else 0.0
                            idx = offset + step * np.arange(target_samples)
                            indices = np.clip(np.round(idx).astype(int), 0, total - 1)
                        else:
                            indices = np.linspace(0, total - 1, num=target_samples, dtype=int)
                    elif self.sampling_mode == "random":
                        replace = target_samples > total
                        indices = np.random.choice(total, target_samples, replace=replace)
                    else:
                        raise ValueError(
                            f"Unsupported sampling_mode '{self.sampling_mode}'. "
                            "Use 'uniform' or 'random'."
                        )
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
        if self.clip_max is not None:
            densities = paddle.clip(densities, min=self.padding_value, max=self.clip_max)
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
