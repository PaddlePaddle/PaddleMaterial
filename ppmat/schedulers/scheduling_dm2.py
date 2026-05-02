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

import math
from typing import Optional

import paddle


class DM2DenoisingScheduler:
    """Noise and denoising schedule used by DM2.

    The official DM2 demos train the denoiser by applying Gaussian rattle noise to
    periodic structures and sampling by repeatedly denoising noisy snapshots. This
    class keeps those diffusion-related pieces outside the model definition so DM2
    follows the scheduler layout used by the other generative models in PPMat.
    """

    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.75,
        train_sigma_distribution: str = "uniform",
        sample_schedule: str = "linear",
        default_num_inference_steps: int = 2900,
        default_final_relax_steps: int = 100,
        default_max_sigma_for_denoising: float = 1.0,
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.train_sigma_distribution = train_sigma_distribution
        self.sample_schedule = sample_schedule
        self.default_num_inference_steps = int(default_num_inference_steps)
        self.default_final_relax_steps = int(default_final_relax_steps)
        self.default_max_sigma_for_denoising = float(default_max_sigma_for_denoising)

    def _sample_sigma(self, shape, dtype):
        if self.sigma_min >= self.sigma_max:
            return paddle.full(shape, self.sigma_max, dtype=dtype)

        if self.train_sigma_distribution == "uniform":
            return paddle.empty(shape, dtype=dtype).uniform_(
                min=self.sigma_min,
                max=self.sigma_max,
            )

        if self.train_sigma_distribution in {"log_uniform", "loguniform"}:
            log_sigma = paddle.empty(shape, dtype=dtype).uniform_(
                min=math.log(self.sigma_min),
                max=math.log(self.sigma_max),
            )
            return paddle.exp(log_sigma)

        raise ValueError(
            "train_sigma_distribution must be 'uniform' or 'log_uniform', "
            f"got {self.train_sigma_distribution}."
        )

    def _expand_sigma_to_nodes(self, data, sigma: Optional[paddle.Tensor]):
        batch = getattr(data, "batch", None)
        num_nodes = data.pos.shape[0]

        if sigma is None:
            if batch is None:
                sigma = self._sample_sigma([1], data.pos.dtype)
                return sigma.expand([num_nodes]).unsqueeze(axis=-1)

            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
            sigma = self._sample_sigma([num_graphs], data.pos.dtype)
            return sigma[batch].unsqueeze(axis=-1)

        if not isinstance(sigma, paddle.Tensor):
            sigma = paddle.to_tensor(sigma, dtype=data.pos.dtype)
        sigma = sigma.astype(data.pos.dtype).reshape([-1])

        if sigma.numel() == 1:
            return sigma.expand([num_nodes]).unsqueeze(axis=-1)
        if sigma.shape[0] == num_nodes:
            return sigma.unsqueeze(axis=-1)
        if batch is not None:
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
            if sigma.shape[0] == num_graphs:
                return sigma[batch].unsqueeze(axis=-1)

        raise ValueError(
            "sigma must be a scalar, one value per graph, or one value per node."
        )

    def add_noise(self, data, sigma: Optional[paddle.Tensor] = None):
        """Apply Gaussian rattle noise and store ``dx``, ``eps`` and ``sigma``."""

        sigma_per_node = self._expand_sigma_to_nodes(data, sigma)
        eps = paddle.randn(data.pos.shape, dtype=data.pos.dtype)
        data.dx = sigma_per_node * eps
        data.pos = data.pos + data.dx

        if getattr(data, "edge_attr", None) is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            data.edge_attr = data.edge_attr + data.dx[dst] - data.dx[src]

        data.sigma = sigma_per_node
        data.eps = eps
        return data

    __call__ = add_noise

    def get_sampling_sigmas(
        self,
        num_inference_steps: Optional[int] = None,
        max_sigma_for_denoising: Optional[float] = None,
        dtype: str = "float32",
    ):
        num_steps = (
            self.default_num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        max_sigma = (
            self.default_max_sigma_for_denoising
            if max_sigma_for_denoising is None
            else float(max_sigma_for_denoising)
        )

        if self.sample_schedule == "linear":
            return paddle.linspace(max_sigma, self.sigma_min, num_steps, dtype=dtype)

        if self.sample_schedule in {"log", "log_linear"}:
            return paddle.exp(
                paddle.linspace(
                    math.log(max_sigma),
                    math.log(self.sigma_min),
                    num_steps,
                    dtype=dtype,
                )
            )

        raise ValueError(
            "sample_schedule must be 'linear' or 'log_linear', "
            f"got {self.sample_schedule}."
        )

    def denoise_step(self, noisy_pos: paddle.Tensor, pred_dx: paddle.Tensor):
        return noisy_pos - pred_dx


class RattleParticles(paddle.nn.Layer):
    """Compatibility wrapper around :class:`DM2DenoisingScheduler.add_noise`."""

    def __init__(self, sigma_max: float, sigma_min: float = 0.001):
        super().__init__()
        self.scheduler = DM2DenoisingScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def forward(self, data):
        return self.scheduler.add_noise(data)


class DownselectEdges(paddle.nn.Layer):
    """Keep only edges whose displacement length is within ``cutoff``."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, data):
        edge_length = paddle.linalg.norm(data.edge_attr[:, :3], axis=1)
        edge_ids = paddle.nonzero(edge_length <= self.cutoff).flatten()
        data.edge_index = paddle.index_select(data.edge_index, edge_ids, axis=1)
        data.edge_attr = paddle.index_select(data.edge_attr, edge_ids, axis=0)
        return data
