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


from typing import Optional
from typing import Tuple

import paddle
from paddle_scatter import scatter_add
from ppmat.schedulers.utils import maybe_expand

from ppmat.utils import logger


def wrap_at_boundary(x: paddle.Tensor, wrapping_boundary: float) -> paddle.Tensor:
    """Wrap x at the boundary given by wrapping_boundary.
    Args:
      x: tensor of shape (batch_size, dim)
      wrapping_boundary: float): wrap at [0, wrapping_boundary] in all dimensions.
    Returns:
      wrapped_x: tensor of shape (batch_size, dim)
    """
    return paddle.remainder(x=x, y=paddle.to_tensor(wrapping_boundary))


class NumAtomsVarianceAdjustedWrappedVESDE:
    """Variance-adjusted wrapped Variational Exponential SDE (VESDE) with atomic number
    scaling.

    This implementation modifies the standard VESDE by scaling the variance using the
    cubic root of the number of atoms. The goal is to reduce the influence by the cell
    size on the variance of the fractional coordinates.

    Args:
        wrapping_boundary (float | paddle.Tensor, optional): Defines the periodic
            boundary range [0, wrapping_boundary] applied uniformly across all
            dimensions. Defaults to 1.0.
        sigma_min (float, optional): Minimum noise scale for the diffusion process.
            This value should be tuned to match the data distribution's inherent noise
            level. Defaults to 0.01.

        sigma_max (float, optional): Maximum noise scale controlling the upper bound of
            the noise schedule. Larger values allow for more aggressive diffusion but
            may require smaller step sizes. Defaults to 5.0.

        snr (float): Signal-to-Noise Ratio parameter balancing deterministic vs
            stochastic components in the SDE. Lower values increase stochasticity.
            Defaults to 0.4.

        max_step_size (int): Maximum allowed integration step size to ensure numerical
            stability. Defaults to 1,000,000.
    """

    def __init__(
        self,
        wrapping_boundary: (float | paddle.Tensor) = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        snr: float = 0.4,
        max_step_size: int = 1000000,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.wrapping_boundary = wrapping_boundary

        self.snr = snr
        self.max_step_size = max_step_size

    def std_scaling(self, num_atoms) -> paddle.Tensor:
        return num_atoms ** (-1 / 3)

    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        mean = x
        std = maybe_expand(
            self.sigma_min * (self.sigma_max / self.sigma_min) ** t, batch_idx, x
        )
        std_scale = self.std_scaling(num_atoms)
        std = std * maybe_expand(std_scale, batch_idx, like=std)
        return mean, std

    def wrap(self, x):
        return wrap_at_boundary(x, self.wrapping_boundary)

    def sample_marginal(
        self,
        x: paddle.Tensor,
        noise: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> paddle.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t)
        """
        mean, std = self.marginal_prob(
            x=x, t=t, batch_idx=batch_idx, num_atoms=num_atoms
        )
        if noise is None:
            noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        return mean + std * noise

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
        batch_idx: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> paddle.Tensor:
        if (original_samples > self.wrapping_boundary).astype("bool").any() or (
            original_samples < 0
        ).astype("bool").any():
            logger.warning(
                "Wrapped SDE has received input outside of the wrapping boundary."
            )
        noisy_x = self.sample_marginal(
            x=original_samples,
            noise=noise,
            t=timesteps,
            batch_idx=batch_idx,
            num_atoms=num_atoms,
        )
        return self.wrap(noisy_x)

    def prior_sampling(
        self,
        shape: (list | tuple),
        num_atoms: paddle.Tensor,
        batch_idx: paddle.Tensor,
    ) -> paddle.Tensor:
        std_scale = self.std_scaling(num_atoms)
        prior_sample = paddle.randn(shape=shape) * self.sigma_max
        return self.wrap(
            prior_sample * maybe_expand(std_scale, batch_idx, like=prior_sample)
        )

    def step_correct(
        self,
        model_output: paddle.Tensor,
        timestep: paddle.Tensor,
        sample: paddle.Tensor,
        batch_idx: paddle.Tensor,
    ):

        prev_sample, prev_sample_mean = self.step_given_score(
            x=sample, score=model_output, t=timestep, batch_idx=batch_idx
        )
        return self.wrap(prev_sample), self.wrap(prev_sample_mean)

    def step_given_score(self, x, batch_idx, score, t):
        alpha = self.get_alpha(t)
        snr = self.snr
        noise = paddle.randn(shape=score.shape, dtype=score.dtype)
        grad_norm_square = (
            paddle.square(x=score).reshape(tuple(score.shape)[0], -1).sum(axis=1)
        )
        noise_norm_square = (
            paddle.square(x=noise).reshape(tuple(noise.shape)[0], -1).sum(axis=1)
        )
        if batch_idx is None:
            grad_norm = grad_norm_square.sqrt().mean()
            noise_norm = noise_norm_square.sqrt().mean()
        else:
            grad_norm = paddle.sqrt(
                x=scatter_add(grad_norm_square, dim=-1, index=batch_idx)
            ).mean()
            noise_norm = paddle.sqrt(
                x=scatter_add(noise_norm_square, dim=-1, index=batch_idx)
            ).mean()
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = paddle.minimum(
            x=step_size, y=paddle.to_tensor(self.max_step_size, dtype="float32")
        )
        if grad_norm == 0:
            step_size[:] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + paddle.sqrt(x=step_size * 2) * noise
        return x, mean

    def get_alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        alpha = paddle.ones_like(x=t)
        return alpha

    def step_pred(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        dt: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        num_atoms,
    ):
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x, t=t, dt=dt, batch_idx=batch_idx, num_atoms=num_atoms
        )
        z = paddle.randn(shape=x_coeff.shape, dtype=x_coeff.dtype)
        mean = x_coeff * x + score_coeff * score
        sample = mean + std * z
        return sample, mean

    def _get_coeffs(self, x, t, dt, batch_idx, num_atoms):
        """
        Compute coefficients for ancestral sampling.
        This is in a separate method to make it easier to test."""
        s = t + dt
        alpha_t, sigma_t = self.mean_coeff_and_std(
            x=x, t=t, batch_idx=batch_idx, num_atoms=num_atoms
        )
        if batch_idx is None:
            is_time_zero = s <= 0
        else:
            is_time_zero = s[batch_idx] <= 0
        alpha_s, sigma_s = self.mean_coeff_and_std(
            x=x, t=s, batch_idx=batch_idx, num_atoms=num_atoms
        )
        sigma_s[is_time_zero] = 0
        sigma2_t_given_s = sigma_t**2 - sigma_s**2 * alpha_t**2 / alpha_s**2
        sigma_t_given_s = paddle.sqrt(x=sigma2_t_given_s)
        std = sigma_t_given_s * sigma_s / sigma_t
        min_alpha_t_given_s = 0.001
        alpha_t_given_s = alpha_t / alpha_s
        if paddle.any(x=alpha_t_given_s < min_alpha_t_given_s):
            logger.warning(
                f"Clipping alpha_t_given_s to {min_alpha_t_given_s} to avoid "
                "divide-by-zero. You should probably change something else to avoid "
                "this."
            )
            alpha_t_given_s = paddle.clip(
                x=alpha_t_given_s, min=min_alpha_t_given_s, max=1
            )
        score_coeff = sigma2_t_given_s / alpha_t_given_s
        x_coeff = 1.0 / alpha_t_given_s
        std[is_time_zero] = 0
        return x_coeff, score_coeff, std

    def mean_coeff_and_std(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: Optional[paddle.Tensor] = None,
        num_atoms=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at
        time t.
        """
        return self.marginal_prob(paddle.ones_like(x=x), t, batch_idx, num_atoms)
