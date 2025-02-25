import sys

import paddle

from mattergen.common.diffusion import corruption as sde_lib
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling import predictors_correctors as pc
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from paddle_utils import *

SampleAndMean = tuple[paddle.Tensor, paddle.Tensor]


class LatticeAncestralSamplingPredictor(AncestralSamplingPredictor):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

    def update_given_score(
        self,
        *,
        x: paddle.Tensor,
        t: paddle.Tensor,
        dt: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        batch: (BatchedData | None)
    ) -> SampleAndMean:
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x, t=t, dt=dt, batch_idx=batch_idx, batch=batch
        )
        mean_coeff = 1 - x_coeff
        z = sde_lib.make_noise_symmetric_preserve_variance(
            paddle.randn(shape=x_coeff.shape, dtype=x_coeff.dtype)
        )
        assert hasattr(self.corruption, "get_limit_mean")
        mean = (
            x_coeff * x
            + score_coeff * score
            + mean_coeff * self.corruption.get_limit_mean(x=x, batch=batch)
        )
        sample = mean + std * z
        return sample, mean


class LatticeLangevinDiffCorrector(pc.LangevinCorrector):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

    def step_given_score(
        self,
        *,
        x: paddle.Tensor,
        batch_idx,  #: (paddle.int64 | None), todo: fix this
        score: paddle.Tensor,
        t: paddle.Tensor
    ) -> SampleAndMean:
        assert isinstance(self.corruption, sde_lib.LatticeVPSDE)
        alpha = self.get_alpha(t)
        snr = self.snr
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        noise = sde_lib.make_noise_symmetric_preserve_variance(noise)
        grad_norm_square = paddle.square(x=score).reshape(tuple(score.shape)[0], -1).sum(axis=1)
        noise_norm_square = paddle.square(x=noise).reshape(tuple(noise.shape)[0], -1).sum(axis=1)
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = paddle.minimum(x=step_size, y=self.max_step_size)
        if grad_norm == 0:
            step_size[:] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + paddle.sqrt(x=step_size * 2) * noise
        x = compute_lattice_polar_decomposition(x)
        mean = compute_lattice_polar_decomposition(mean)
        return x, mean
