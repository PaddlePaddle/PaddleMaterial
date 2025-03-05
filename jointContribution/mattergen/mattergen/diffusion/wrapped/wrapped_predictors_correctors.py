from typing import Optional, Tuple

import mattergen.diffusion.sampling.predictors_correctors as pc
import paddle
from mattergen.diffusion.corruption import sde_lib
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.sampling import predictors
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin

SampleAndMean = Tuple[paddle.Tensor, paddle.Tensor]


class WrappedPredictorMixin:
    """A mixin for wrapping the predictor in a WrappedSDE."""

    def update_given_score(
        self,
        *,
        x: paddle.Tensor,
        t: paddle.Tensor,
        dt: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        batch: Optional[BatchedData],
    ) -> SampleAndMean:
        assert isinstance(self, predictors.Predictor)
        _super = super()
        assert hasattr(_super, "update_given_score")
        assert hasattr(self, "corruption")
        if not hasattr(self.corruption, "wrap"):
            raise IncompatibleSampler(
                f"{self.__class__.__name__} is not compatible with {self.corruption}."
            )
        sample, mean = _super.update_given_score(
            x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=batch
        )
        return self.corruption.wrap(sample), self.corruption.wrap(mean)


class WrappedCorrectorMixin:
    """A mixin for wrapping the corrector in a WrappedSDE."""

    def step_given_score(
        self,
        *,
        x: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        t: paddle.Tensor,
    ) -> SampleAndMean:
        assert isinstance(self, pc.LangevinCorrector)
        _super = super()
        assert hasattr(_super, "step_given_score")
        assert hasattr(self, "corruption") and hasattr(self.corruption, "wrap")
        if not hasattr(self.corruption, "wrap"):
            raise IncompatibleSampler(
                f"{self.__class__.__name__} is not compatible with {self.corruption}."
            )
        sample, mean = _super.step_given_score(
            x=x, score=score, t=t, batch_idx=batch_idx
        )
        return self.corruption.wrap(sample), self.corruption.wrap(mean)


class WrappedAncestralSamplingPredictor(
    WrappedPredictorMixin, predictors.AncestralSamplingPredictor
):
    @classmethod
    def is_compatible(cls, corruption: Corruption):
        return isinstance(corruption, (sde_lib.VPSDE, sde_lib.VESDE)) and isinstance(
            corruption, WrappedSDEMixin
        )


class WrappedLangevinCorrector(WrappedCorrectorMixin, pc.LangevinCorrector):
    @classmethod
    def is_compatible(cls, corruption: Corruption):
        return isinstance(corruption, (sde_lib.VPSDE, sde_lib.VESDE)) and isinstance(
            corruption, WrappedSDEMixin
        )
