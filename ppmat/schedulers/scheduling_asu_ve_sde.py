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

"""ASU-wrapped VE-SDE scheduler with space-group-aware sigma norms."""

import math
import pickle

import numpy as np
import paddle
import paddle.nn as nn

import ppmat.utils.wyckoff_data as wyckoff_data
from ppmat.utils.wyckoff_data import _ensure_wyckoff_shape_decomp
from ppmat.utils import logger
from ppmat.utils.asu_crystal import uniformly_sample_point_in_asu_wyckoff_site
from ppmat.utils.crystal import MAX_WYCKOFF_POSITIONS, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.utils.asu_math import (
    get_space_group_ops_and_conventional_atoms,
    d_log_p_asu_wrapped_normal,
)


class ASUVESDEScheduler(nn.Layer):
    """VE-SDE scheduler with ASU-wrapped sigma norms for space-group-aware score matching."""

    def __init__(
        self,
        num_timesteps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        sigma_norm_type: str = "asu_wrapped",
        num_lattice_translations: int = 5,
        num_monte_carlo_samples: int = 10_000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_lattice_translations = num_lattice_translations

        sigmas = paddle.to_tensor(
            np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_timesteps)),
            dtype=paddle.float32,
        )

        if sigma_norm_type == "unwrapped":
            _sigma_norms = self._sigma_norm_unwrapped(sigmas)
        elif sigma_norm_type == "asu_wrapped":
            _sigma_norms = self._sigma_norm_asu_wrapped(sigmas, num_monte_carlo_samples)
        else:
            raise AttributeError(f"Unknown sigma_norm_type: {sigma_norm_type}")

        self.register_buffer(
            "sigmas",
            paddle.concat([paddle.zeros([1]), sigmas], axis=0),
        )
        self.register_buffer(
            "sigma_norms",
            paddle.concat(
                [paddle.ones([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, 1]), _sigma_norms],
                axis=-1,
            ),
        )

    def uniform_sample_timestep(self, batch_size: int) -> paddle.Tensor:
        """Uniformly sample integer timesteps in [1, num_timesteps]."""
        return paddle.randint(
            low=1,
            high=self.num_timesteps + 1,
            shape=[batch_size],
            dtype=paddle.int64,
        )

    @paddle.no_grad()
    def _sigma_norm_unwrapped(self, sigmas: paddle.Tensor) -> paddle.Tensor:
        sigma_norms = 1.0 / sigmas
        return sigma_norms[None, None, :].expand([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, -1])

    @paddle.no_grad()
    def _sigma_norm_asu_wrapped(
        self, sigmas: paddle.Tensor, num_monte_carlo_samples: int = 10_000,
    ) -> paddle.Tensor:
        """Monte Carlo estimate of expected score L2 norm for ASU-wrapped normal."""
        num_timesteps = sigmas.shape[0]
        num_lattice_translations = self.num_lattice_translations

        cache_name = (
            f"expected_score_norms_minSigma{float(sigmas[0]):0.3f}"
            f"_maxSigma{float(sigmas[-1]):0.3f}"
            f"_T{num_timesteps}_{num_monte_carlo_samples}MCsamples"
            f"_{num_lattice_translations}LatticeTranslations.pdparams"
        )
        cache_path = wyckoff_data.DATA_DIRECTORY / cache_name
        logger.info(f"Checking cache: {cache_path}")
        logger.info(f"Cache exists: {cache_path.exists()}")
        if cache_path.exists():
            logger.info(f"Loading sigma norms from: {cache_path}")
            sigma_norms = paddle.load(str(cache_path))
            logger.info(f"Successfully loaded sigma_norms with shape: {sigma_norms.shape}")
            return sigma_norms
        logger.info("Cache not found, computing sigma_norms...")

        _ensure_wyckoff_shape_decomp()
        with open(wyckoff_data.DATA_DIRECTORY / "wyckoff_shape_decomposition.pkl", "rb") as f:
            wyckoff_shape_decomp_dict = pickle.load(f)

        asu_wyckoff_dict = wyckoff_data.asu_wyckoff_dict
        sigma_norms = paddle.zeros([NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, MAX_WYCKOFF_POSITIONS, num_timesteps], dtype=paddle.float32)

        for sg_num in range(NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS, 0, -1):
            sg_dict = asu_wyckoff_dict[str(sg_num)]
            wyckoff_letters = sg_dict["ordered_wyckoff_letters"]

            x0s, wsi = uniformly_sample_point_in_asu_wyckoff_site(
                space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
                wyckoff_letters=wyckoff_letters,
                dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
                dictionary_of_wyckoff_shape_decompositions=wyckoff_shape_decomp_dict,
                hull_equations_3d=wyckoff_data.asu_hull_equations,
                n_samples_per_wyckoff=num_monte_carlo_samples,
                return_sampled_wyckoff_shape_indices=True,
            )

            space_group_idx = paddle.to_tensor([sg_num - 1], dtype=paddle.int64)

            for i, letter in enumerate(wyckoff_letters):
                _wyckoff_idx = paddle.to_tensor([i], dtype=paddle.int64)
                _wyckoff_idx_expanded = _wyckoff_idx.expand([num_monte_carlo_samples])

                sg_ops = get_space_group_ops_and_conventional_atoms(
                    x0s[i],
                    paddle.zeros_like(_wyckoff_idx_expanded),
                    _wyckoff_idx_expanded,
                    space_group_idx,
                    n_atoms_per_xtal=paddle.to_tensor(
                        [num_monte_carlo_samples], dtype=paddle.int64
                    ),
                )
                map_conv_to_asu = sg_ops.map_conventional_to_asu_atom
                orbited_x = sg_ops.conventional_frac_coords
                unique_indices = sg_ops.unique_non_overlapping_atom_indices
                map_unique_conv_to_asu = map_conv_to_asu[unique_indices]

                _chunks = min(200, num_timesteps)
                for sigma_idxs in paddle.chunk(
                    paddle.arange(num_timesteps), chunks=_chunks
                ):
                    batch_sigmas = sigmas[sigma_idxs]
                    norms = []
                    for sigma in batch_sigmas.tolist():
                        noise = paddle.randn([num_monte_carlo_samples, 3]) * sigma
                        xts = x0s[i] + noise

                        scores = d_log_p_asu_wrapped_normal(
                            xts, orbited_x, map_unique_conv_to_asu,
                            num_lattice_translations, sigma,
                        )
                        norm_t = ((scores ** 2).sum(axis=-1)).sqrt().mean()
                        norms.append(float(norm_t.item()))

                    sigma_norms[sg_num - 1, i, sigma_idxs] = paddle.to_tensor(
                        norms, dtype=paddle.float32
                    )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        paddle.save(sigma_norms, str(cache_path))
        logger.info(f"Saved sigma norms to: {cache_path}")
        return sigma_norms

    @paddle.no_grad()
    def step_pred(self, x: paddle.Tensor, score: paddle.Tensor, t: int, noise: paddle.Tensor) -> paddle.Tensor:
        """Predictor step: x_{t} = x_{t+1} + (sigma_{t+1}^2 - sigma_t^2) * score + sqrt(sigma_{t+1}^2 - sigma_t^2) * noise."""
        sigma_sq_diff = self.sigmas[t + 1] ** 2 - self.sigmas[t] ** 2
        return x + sigma_sq_diff * score + paddle.sqrt(sigma_sq_diff) * noise

    @paddle.no_grad()
    def step_correct(
        self, x: paddle.Tensor, score: paddle.Tensor, noise: paddle.Tensor,
        snr: float = 0.4, max_step_size: float = 1e6,
    ) -> paddle.Tensor:
        """Corrector (Langevin) step: x = x + step_size * score + sqrt(2 * step_size) * noise."""
        noise_norm = ((noise ** 2).sum(axis=-1)).sqrt().mean()
        grad_norm = ((score ** 2).sum(axis=-1)).sqrt().mean()
        step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2
        step_size = paddle.where(noise == 0.0, paddle.zeros_like(noise), step_size * paddle.ones_like(noise))
        step_size = paddle.nan_to_num(step_size, nan=0.0, posinf=max_step_size, neginf=-max_step_size)
        return x + step_size * score + paddle.sqrt(2 * step_size) * noise

    @paddle.no_grad()
    def get_interpolated_sigma_norm_t(self, t, space_group_indices, wyckoff_indices):
        """Interpolate sigma_norm at real time t."""
        t_val = float(t.item())
        assert 0.0 <= t_val <= self.num_timesteps

        t_below = int(math.floor(t_val))
        t_above = int(math.ceil(t_val))
        if t_below == t_above:
            if t_below == 0:
                t_below, t_above = 0, 1
            elif t_below == self.num_timesteps:
                t_above = self.num_timesteps
                t_below = self.num_timesteps - 1
            else:
                t_above = t_below + 1

        p = (t_val - t_below) / max(t_above - t_below, 1)
        sn_below = self.sigma_norms[space_group_indices, wyckoff_indices, t_below]
        sn_above = self.sigma_norms[space_group_indices, wyckoff_indices, t_above]
        return (1 - p) * sn_below + p * sn_above
