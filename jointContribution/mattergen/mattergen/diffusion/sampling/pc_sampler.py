from __future__ import annotations

from typing import Generic, Mapping, Tuple, TypeVar

import paddle
from mattergen.diffusion.corruption.multi_corruption import (MultiCorruption,
                                                             apply)
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
# from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_partials import (CorrectorPartial,
                                                      PredictorPartial)
from tqdm.auto import tqdm

Diffusable = TypeVar("Diffusable", bound=BatchedData)
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: (dict[str, PredictorPartial] | None) = None,
        corrector_partials: (dict[str, CorrectorPartial] | None) = None,
        device: (paddle.CPUPlace, paddle.CUDAPlace, str),
        n_steps_corrector: int,
        N: int,
        eps_t: float = 0.001,
        max_t: (float | None) = None
    ):
        """
        Args:
            diffusion_module: diffusion module
            predictor_partials: partials for constructing predictors. Keys are the names of the corruptions.
            corrector_partials: partials for constructing correctors. Keys are the names of the corruptions.
            device: device to run on
            n_steps_corrector: number of corrector steps
            N: number of noise levels
            eps_t: diffusion time to stop denoising at
            max_t: diffusion time to start denoising at. If None, defaults to the maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
        """
        self._diffusion_module = diffusion_module
        self.N = N
        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"
        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            assert set(
                c.N for c in self._multi_corruption.discrete_corruptions.values()
            ) == {N}
        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }
        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: paddle.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    @classmethod
    def from_pl_module(
        cls, diffusion_module, **kwargs
    ) -> PredictorCorrector:
        device = diffusion_module.parameters()[0].place
        return cls(
            diffusion_module=diffusion_module,
            device=device,
            **kwargs
        )

    @paddle.no_grad()
    def sample(
        self,
        conditioning_data: BatchedData,
        mask: (Mapping[str, paddle.Tensor] | None) = None,
    ) -> SampleAndMean:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @paddle.no_grad()
    def sample_with_record(
        self,
        conditioning_data: BatchedData,
        mask: (Mapping[str, paddle.Tensor] | None) = None,
    ) -> SampleAndMeanAndRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @paddle.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: (Mapping[str, paddle.Tensor] | None) = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch, recorded_samples, recorded_predictions).
           The difference between the former two is that `mean_batch` has no noise added at the final denoising step.
           The latter two are only returned if `record` is True, and contain the samples and predictions from each step of the diffusion process.

        """
        if isinstance(self._diffusion_module, paddle.nn.Layer):
            self._diffusion_module.eval()
        mask = mask or {}
        # conditioning_data = conditioning_data.to(self._device)
        # mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise(batch=batch, mask=mask, record=record)

    @paddle.no_grad()
    def _denoise(
        self, batch: Diffusable, mask: dict[str, paddle.Tensor], record: bool = False
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        timesteps = paddle.linspace(start=self._max_t, stop=self._eps_t, num=self.N)
        dt = -paddle.to_tensor(data=(self._max_t - self._eps_t) / (self.N - 1)).to(
            self._device
        )
        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            t = paddle.full(shape=(batch.get_batch_size(),), fill_value=timesteps[i])
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score
                        for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[
                        str, Tuple[paddle.Tensor, paddle.Tensor]
                    ] = apply(
                        fns=fns,
                        broadcast={"t": t},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().cpu())
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=mask,
                    )
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score
                for k, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().cpu())
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=mask,
            )
        return batch, mean_batch, recorded_samples


def _mask_replace(
    samples_means: dict[str, Tuple[paddle.Tensor, paddle.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, paddle.Tensor | None],
) -> SampleAndMean:
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *,
    sample_and_mean: Tuple[paddle.Tensor, paddle.Tensor],
    old_x: paddle.Tensor,
    mask: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)


def _mask(
    *, old_x: paddle.Tensor, new_x: paddle.Tensor, mask: (paddle.Tensor | None)
) -> paddle.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(y=old_x, weight=mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: (Mapping[str, paddle.Tensor] | None),
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=tuple(conditioning_data[k].shape),
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].place)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(y=conditioning_data[k], weight=msk)
    return conditioning_data.replace(**samples)
