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


import paddle

from ppmat.utils import logger
from ppmat.utils.crystal import compute_lattice_polar_decomposition
from ppmat.utils.misc import expand
from ppmat.utils.misc import make_noise_symmetric_preserve_variance
from ppmat.utils.misc import maybe_expand


class LatticeVPSDEScheduler:
    """Lattice Variance Preserving SDE Scheduler.

    Args:
        beta_min (float, optional): Minimum beta value. Defaults to 0.1.
        beta_max (float, optional): Maximum beta value. Defaults to 20.
        limit_density (float, optional): Limit density. Defaults to 0.05.
        limit_var_scaling_constant (float, optional): Scaling constant for limiting
            variance. Defaults to 0.25.
        snr (float, optional): Signal-to-Noise ratio. Defaults to 0.2.
        max_step_size (int, optional): Maximum allowed integration step size to ensure
            numerical stability. Defaults to 1,000,000.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        limit_density: float = 0.05,
        limit_var_scaling_constant: float = 0.25,
        snr: float = 0.2,
        max_step_size: int = 1000000,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.limit_density = limit_density
        self.limit_var_scaling_constant = limit_var_scaling_constant

        self.T = 1.0
        self.snr = snr
        self.max_step_size = max_step_size

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _marginal_mean_coeff(self, t: paddle.Tensor) -> paddle.Tensor:
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        return paddle.exp(x=log_mean_coeff)

    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        mean_coeff = self._marginal_mean_coeff(t)
        limit_mean = self.get_limit_mean(x=x, num_atoms=num_atoms)
        limit_var = self.get_limit_var(x=x, num_atoms=num_atoms)
        mean_coeff_expanded = maybe_expand(mean_coeff, batch=None, like=x)
        mean = mean_coeff_expanded * x + (1 - mean_coeff_expanded) * limit_mean
        std = paddle.sqrt(x=(1.0 - mean_coeff_expanded**2) * limit_var)
        return mean, std

    def get_limit_mean(self, x: paddle.Tensor, num_atoms=None) -> paddle.Tensor:
        return paddle.pow(
            x=paddle.eye(num_rows=3).expand(shape=x.shape)
            * num_atoms[:, None, None].astype("float32")
            / self.limit_density,
            y=1.0 / 3,
        )

    def get_limit_var(self, x: paddle.Tensor, num_atoms) -> paddle.Tensor:
        """
        Returns the element-wise variance of the limit distribution.
        NOTE: even though we have a different limit variance per data
        dimension we still sample IID for each element per data point.
        We do NOT do any correlated sampling over data dimensions per
        data point.

        Return shape=x.shape
        """
        n_atoms_expanded = expand(num_atoms, tuple(x.shape))
        n_atoms_expanded = paddle.tile(x=n_atoms_expanded, repeat_times=(1, 3, 3)).cast(
            "float32"
        )
        out = (
            paddle.pow(x=n_atoms_expanded, y=2.0 / 3) * self.limit_var_scaling_constant
        )
        return out

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
        num_atoms: paddle.Tensor,
    ) -> paddle.Tensor:
        mean, std = self.marginal_prob(
            x=original_samples, t=timesteps, num_atoms=num_atoms
        )
        if noise is None:
            z = paddle.randn(shape=original_samples.shape, dtype=original_samples.dtype)
            noise = make_noise_symmetric_preserve_variance(z)
        return mean + expand(std, tuple(noise.shape)) * noise

    def prior_sampling(
        self,
        shape: (list | tuple),
        num_atoms: paddle.Tensor,
    ) -> paddle.Tensor:
        x_sample = paddle.randn(shape=shape)
        x_sample = make_noise_symmetric_preserve_variance(x_sample)
        limit_mean = self.get_limit_mean(x=x_sample, num_atoms=num_atoms)
        limit_var = self.get_limit_var(x=x_sample, num_atoms=num_atoms)
        return x_sample * limit_var.sqrt() + limit_mean

    def get_alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        alpha = 1 - self.beta(t) * self.T / 1000
        return alpha

    def step_correct(
        self,
        x: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        t: paddle.Tensor,
    ):
        alpha = self.get_alpha(t)
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        noise = make_noise_symmetric_preserve_variance(noise)
        grad_norm_square = (
            paddle.square(x=score).reshape(tuple(score.shape)[0], -1).sum(axis=1)
        )
        noise_norm_square = (
            paddle.square(x=noise).reshape(tuple(noise.shape)[0], -1).sum(axis=1)
        )
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = paddle.minimum(
            x=step_size, y=paddle.to_tensor(self.max_step_size, dtype="float32")
        )
        if grad_norm == 0:
            step_size[:] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + paddle.sqrt(x=step_size * 2) * noise
        x = compute_lattice_polar_decomposition(x)
        mean = compute_lattice_polar_decomposition(mean)
        return x, mean

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
        mean_coeff = 1 - x_coeff
        z = make_noise_symmetric_preserve_variance(
            paddle.randn(shape=x_coeff.shape, dtype=x_coeff.dtype)
        )
        mean = (
            x_coeff * x
            + score_coeff * score
            + mean_coeff * self.get_limit_mean(x=x, num_atoms=num_atoms)
        )
        sample = mean + std * z
        return sample, mean

    def mean_coeff_and_std(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        num_atoms=None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at
        time t."""
        mean_coeff = self._marginal_mean_coeff(t)
        std = self.marginal_prob(x, t, num_atoms)[1]
        return maybe_expand(mean_coeff, batch=None, like=x), std

    def _get_coeffs(self, x, t, dt, batch_idx, num_atoms):
        """
        Compute coefficients for ancestral sampling.
        This is in a separate method to make it easier to test."""

        s = t + dt
        alpha_t, sigma_t = self.mean_coeff_and_std(x=x, t=t, num_atoms=num_atoms)
        if batch_idx is None:
            is_time_zero = s <= 0
        else:
            is_time_zero = s[batch_idx] <= 0
        alpha_s, sigma_s = self.mean_coeff_and_std(x=x, t=s, num_atoms=num_atoms)
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
