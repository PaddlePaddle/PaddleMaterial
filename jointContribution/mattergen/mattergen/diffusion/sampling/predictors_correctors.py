import sys


import paddle
from paddle_utils import *

"""Adapted from https://github.com/yang-song/score_sde_pytorch which is released under Apache license.

Key changes:
- Introduced batch_idx argument to work with graph-like data (e.g. molecules)
- Introduced `..._given_score` methods so that multiple fields can be sampled at once using a shared score model. See PredictorCorrector for how this is used.
"""
import abc

from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.sde_lib import (VESDE, VPSDE, BaseVPSDE,
                                                    Corruption, ScoreFunction)
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin
from paddle_scatter import scatter_add

SampleAndMean = tuple[paddle.Tensor, paddle.Tensor]


class Sampler(abc.ABC):
    def __init__(self, corruption: Corruption, score_fn: (ScoreFunction | None)):
        if not self.is_compatible(corruption):
            raise IncompatibleSampler(
                f"{self.__class__.__name__} is not compatible with {corruption}"
            )
        self.corruption = corruption
        self.score_fn = score_fn

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return True


class LangevinCorrector(Sampler):
    def __init__(
        self,
        corruption: Corruption,
        score_fn: (ScoreFunction | None),
        n_steps: int,
        snr: float = 0.2,
        max_step_size: float = 1.0,
    ):
        """The Langevin corrector.

        Args:
            corruption: corruption process
            score_fn: score function
            n_steps: number of Langevin steps at each noise level
            snr: signal-to-noise ratio
            max_step_size: largest coefficient that the score can be multiplied by for each Langevin step.
        """
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.n_steps = n_steps
        self.snr = snr
        self.max_step_size = paddle.to_tensor(data=max_step_size)

    @classmethod
    def is_compatible(cls, corruption: Corruption):
        return (
            isinstance(corruption, (VESDE, BaseVPSDE))
            and super().is_compatible(corruption)
            and not isinstance(corruption, WrappedSDEMixin)
        )

    def update_fn(self, *, x, t, batch_idx) -> SampleAndMean:
        assert self.score_fn is not None, "Did you mean to use step_given_score?"
        for _ in range(self.n_steps):
            score = self.score_fn(x, t, batch_idx)
            x, x_mean = self.step_given_score(
                x=x, batch_idx=batch_idx, score=score, t=t
            )
        return x, x_mean

    def get_alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        sde = self.corruption
        if isinstance(sde, VPSDE):
            alpha = 1 - sde.beta(t) * sde.T / 1000
        else:
            alpha = paddle.ones_like(x=t)
        return alpha

    def step_given_score(
        self, *, x, batch_idx, #: (paddle.int64 | None), todo: fix this
        score, t
    ) -> SampleAndMean:
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
        step_size = paddle.minimum(x=step_size, y=self.max_step_size)
        if grad_norm == 0:
            step_size[:] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + paddle.sqrt(x=step_size * 2) * noise
        return x, mean
