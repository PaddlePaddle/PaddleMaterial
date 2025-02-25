import sys

import paddle
from omegaconf import DictConfig

from mattergen.diffusion.corruption.corruption import B
from mattergen.diffusion.corruption.corruption import BatchedData
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE as DiffSDE
from mattergen.diffusion.corruption.sde_lib import VESDE as DiffVESDE
from mattergen.diffusion.corruption.sde_lib import VPSDE
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE
from paddle_utils import *


def expand(a, x_shape, left=False):
    a_dim = len(tuple(a.shape))
    if left:
        return a.reshape(*((1,) * (len(x_shape) - a_dim) + tuple(a.shape)))
    else:
        return a.reshape(*(tuple(a.shape) + (1,) * (len(x_shape) - a_dim)))


def make_noise_symmetric_preserve_variance(noise: paddle.Tensor) -> paddle.Tensor:
    """Makes the noise matrix symmetric, preserving the variance. Assumes i.i.d. noise for each dimension.

    Args:
        noise (paddle.Tensor): Input noise matrix, must be a batched square matrix, i.e., have shape (batch_size, dim, dim).

    Returns:
        paddle.Tensor: The symmetric noise matrix, with the same variance as the input.
    """
    assert (
        len(tuple(noise.shape)) == 3 and tuple(noise.shape)[1] == tuple(noise.shape)[2]
    ), "Symmetric noise only works for square-matrix-shaped data."
    return (
        1
        / 2**0.5
        * (1 - paddle.eye(num_rows=3)[None])
        * (noise + noise.transpose(perm=dim2perm(noise.ndim, 1, 2)))
        + paddle.eye(num_rows=3)[None] * noise
    )


class LatticeVPSDE(VPSDE):
    @staticmethod
    def from_vpsde_config(vpsde_config: DictConfig):
        return LatticeVPSDE(**vpsde_config)

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        limit_density: (float | None) = 0.05,
        limit_var_scaling_constant: float = 0.25,
        **kwargs
    ):
        """Variance-preserving SDE with drift coefficient changing linearly over time."""
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.limit_density = limit_density
        self.limit_var_scaling_constant = limit_var_scaling_constant
        self._limit_info_key = "num_atoms"

    @property
    def limit_info_key(self) -> str:
        return self._limit_info_key

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: paddle.Tensor) -> paddle.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return paddle.exp(x=log_mean_coeff)

    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        assert batch is not None
        mean_coeff = self._marginal_mean_coeff(t)
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)
        mean_coeff_expanded = maybe_expand(mean_coeff, batch_idx, x)
        mean = mean_coeff_expanded * x + (1 - mean_coeff_expanded) * limit_mean
        std = paddle.sqrt(x=(1.0 - mean_coeff_expanded**2) * limit_var)
        return mean, std

    def mean_coeff_and_std(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at time t."""
        mean_coeff = self._marginal_mean_coeff(t)
        std = self.marginal_prob(x, t, batch_idx, batch)[1]
        return maybe_expand(mean_coeff, batch=None, like=x), std

    def get_limit_mean(self, x: paddle.Tensor, batch: BatchedData) -> paddle.Tensor:
        n_atoms = batch[self.limit_info_key]
        return paddle.pow(
            x=paddle.eye(num_rows=3).expand(shape=[len(n_atoms), 3, 3])
            * n_atoms[:, None, None].astype("float32")
            / self.limit_density,
            y=1.0 / 3,
        ).to(x.place)

    def get_limit_var(self, x: paddle.Tensor, batch: BatchedData) -> paddle.Tensor:
        """
        Returns the element-wise variance of the limit distribution.
        NOTE: even though we have a different limit variance per data
        dimension we still sample IID for each element per data point.
        We do NOT do any correlated sampling over data dimensions per
        data point.

        Return shape=x.shape
        """
        n_atoms = batch[self.limit_info_key]
        n_atoms_expanded = expand(n_atoms, tuple(x.shape))
        n_atoms_expanded = paddle.tile(x=n_atoms_expanded, repeat_times=(1, 3, 3)).cast("float32")
        out = (
            paddle.pow(x=n_atoms_expanded, y=2.0 / 3).to(x.place) * self.limit_var_scaling_constant
        )
        return out

    def sample_marginal(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> paddle.Tensor:
        mean, std = self.marginal_prob(x=x, t=t, batch=batch)
        z = paddle.randn(shape=x.shape, dtype=x.dtype)
        z = make_noise_symmetric_preserve_variance(z)
        return mean + expand(std, tuple(z.shape)) * z

    def prior_sampling(
        self,
        shape: (list | tuple),
        conditioning_data: (BatchedData | None) = None,
        batch_idx: B = None,
    ) -> paddle.Tensor:
        x_sample = paddle.randn(shape=shape)
        x_sample = make_noise_symmetric_preserve_variance(x_sample)
        assert conditioning_data is not None
        limit_info = conditioning_data[self.limit_info_key]
        x_sample = x_sample.to(limit_info.place)
        limit_mean = self.get_limit_mean(x=x_sample, batch=conditioning_data)
        limit_var = self.get_limit_var(x=x_sample, batch=conditioning_data)
        return x_sample * limit_var.sqrt() + limit_mean

    def sde(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        assert batch is not None
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)
        beta_t = self.beta(t)
        drift = -0.5 * expand(beta_t, tuple(x.shape)) * (x - limit_mean)
        diffusion = paddle.sqrt(x=expand(beta_t, tuple(limit_var.shape)) * limit_var)
        return maybe_expand(drift, batch_idx), maybe_expand(diffusion, batch_idx)


class NumAtomsVarianceAdjustedWrappedVESDE(WrappedVESDE):
    """Wrapped VESDE with variance adjusted by number of atoms. We divide the standard deviation by the cubic root of the number of atoms.
    The goal is to reduce the influence by the cell size on the variance of the fractional coordinates.
    """

    def __init__(
        self,
        wrapping_boundary: (float | paddle.Tensor) = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        limit_info_key: str = "num_atoms",
    ):
        super().__init__(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            wrapping_boundary=wrapping_boundary,
        )
        self.limit_info_key = limit_info_key

    def std_scaling(self, batch: BatchedData) -> paddle.Tensor:
        return batch[self.limit_info_key] ** (-1 / 3)

    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        mean, std = super().marginal_prob(x, t, batch_idx, batch)
        assert (
            batch is not None
        ), "batch must be provided when using NumAtomsVarianceAdjustedWrappedVESDEMixin"
        std_scale = self.std_scaling(batch)
        std = std * maybe_expand(std_scale, batch_idx, like=std)
        return mean, std

    def prior_sampling(
        self,
        shape: (list | tuple),
        conditioning_data: (BatchedData | None) = None,
        batch_idx=None,
    ) -> paddle.Tensor:
        _super = super()
        assert isinstance(self, DiffSDE) and hasattr(_super, "prior_sampling")
        assert (
            conditioning_data is not None
        ), "batch must be provided when using NumAtomsVarianceAdjustedWrappedVESDEMixin"
        num_atoms = conditioning_data[self.limit_info_key]
        batch_idx = paddle.repeat_interleave(
            x=paddle.arange(end=tuple(num_atoms.shape)[0]), repeats=num_atoms, axis=0
        )
        std_scale = self.std_scaling(conditioning_data)
        prior_sample = DiffVESDE.prior_sampling(self, shape=shape).to(num_atoms.place)
        return self.wrap(prior_sample * maybe_expand(std_scale, batch_idx, like=prior_sample))

    def sde(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: (BatchedData | None) = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        sigma = self.marginal_prob(x, t, batch_idx, batch)[1]
        sigma_min = self.marginal_prob(x, paddle.zeros_like(x=t), batch_idx, batch)[1]
        sigma_max = self.marginal_prob(x, paddle.ones_like(x=t), batch_idx, batch)[1]
        drift = paddle.zeros_like(x=x)
        diffusion = sigma * paddle.sqrt(x=2 * (sigma_max.log() - sigma_min.log()))
        return drift, diffusion
