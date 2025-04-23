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

# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

import math
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from ppmat.utils.paddle_utils import randn_tensor


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        exp1 = paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
        p_ += (x + T * i) / sigma**2 * exp1
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].tile([sn, 1])
    nprandom = paddle.randn(shape=sigmas.shape, dtype=sigmas.dtype)
    x_sample = sigma * nprandom
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(axis=0)


@dataclass
class SdeVeOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`paddle.Tensor` of shape `(batch_size, num_channels,
            height, width)` for images): Computed sample `(x_{t-1})` of previous
            timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_sample_mean (`paddle.Tensor` of shape `(batch_size, num_channels,
            height, width)` for images): Mean averaged `prev_sample` over previous
            timesteps.
    """

    prev_sample: paddle.Tensor
    prev_sample_mean: paddle.Tensor


@dataclass
class SchedulerOutput:
    """
    Base class for the output of a scheduler's `step` function.

    Args:
        prev_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height,
            width)` for images): Computed sample `(x_{t-1})` of previous timestep.
            `prev_sample` should be used as next model input in the denoising loop.
    """

    prev_sample: paddle.Tensor


class ScoreSdeVeScheduler:
    """
    `ScoreSdeVeScheduler` is a variance exploding stochastic differential equation
    (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the
    superclass documentation for the generic  methods the library implements for all
    schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        snr (`float`, defaults to 0.15):
            A coefficient weighting the step from the `model_output` sample (from the
            network) to the random noise.
        sigma_min (`float`, defaults to 0.01):
            The initial noise scale for the sigma sequence in the sampling procedure.
            The minimum sigma should mirror the distribution of the data.
        sigma_max (`float`, defaults to 1348.0):
            The maximum value used for the range of continuous timesteps passed into
            the model.
        sampling_eps (`float`, defaults to 1e-5):
            The end value of sampling where timesteps decrease progressively from 1 to
            epsilon.
        correct_steps (`int`, defaults to 1):
            The number of correction steps performed on a produced sample.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.snr = snr
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sampling_eps = sampling_eps
        self.correct_steps = correct_steps

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # setable values
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

    def scale_model_input(
        self, sample: paddle.Tensor, timestep: Optional[int] = None
    ) -> paddle.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising
        model input depending on the current timestep.

        Args:
            sample (`paddle.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `paddle.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(self, num_inference_steps: int, sampling_eps: float = None):
        """
        Sets the continuous timesteps used for the diffusion chain (to be run before
        inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a
                pre-trained model.
            sampling_eps (`float`, *optional*):
                The final timestep value (overrides value given during scheduler
                instantiation).
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the
                timesteps are not moved.

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.sampling_eps

        self.timesteps = paddle.linspace(1, sampling_eps, num_inference_steps)

    def set_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float = None,
        sigma_max: float = None,
        sampling_eps: float = None,
    ):
        """
        Sets the noise scales used for the diffusion chain (to be run before
        inference). The sigmas control the weight of the `drift` and `diffusion`
        components of the sample update.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a
                pre-trained model.
            sigma_min (`float`, optional):
                The initial noise scale value (overrides value given during scheduler
                instantiation).
            sigma_max (`float`, optional):
                The final noise scale value (overrides value given during scheduler
                instantiation).
            sampling_eps (`float`, optional):
                The final timestep value (overrides value given during scheduler
                instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        self.sigmas = sigma_min * (sigma_max / sigma_min) ** (
            self.timesteps / sampling_eps
        )
        self.discrete_sigmas = paddle.exp(
            paddle.linspace(
                math.log(sigma_min), math.log(sigma_max), num_inference_steps
            )
        )
        self.sigmas = paddle.to_tensor(
            [sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps]
        )

    def get_adjacent_sigma(self, timesteps, t):
        # NOTE (TODO, junnyu) BUG in PaddlePaddle, here is the issue https://github.com/PaddlePaddle/Paddle/issues/56335
        index = timesteps - 1
        if (index < 0).all():
            index += self.discrete_sigmas.shape[0]
        return paddle.where(
            timesteps == 0,
            paddle.zeros_like(t),
            self.discrete_sigmas[index],
        )

    def step_pred(
        self,
        model_output: paddle.Tensor,
        timestep: int,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SdeVeOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This
        function propagates the diffusion process from the learned model outputs
        (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`paddle.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`]
                is returned, otherwise a tuple is returned where the first element is
                the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after "
                "creating the scheduler"
            )

        timestep = timestep * paddle.ones(
            [
                sample.shape[0],
            ]
        )  # paddle.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).cast("int64")

        # NOTE(laixinlu) convert sigmas to the dtype of the model output
        if self.discrete_sigmas.dtype != model_output.dtype:
            self.discrete_sigmas = self.discrete_sigmas.cast(model_output.dtype)

        sigma = self.discrete_sigmas[timesteps]
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep)
        drift = paddle.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is
        # grad_x log pt(x) also equation 47 shows the analog from SDE models to
        # ancestral sampling methods
        diffusion = diffusion.flatten()
        while len(diffusion.shape) < len(sample.shape):
            diffusion = diffusion.unsqueeze(-1)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = randn_tensor(sample.shape, generator=generator, dtype=sample.dtype)
        prev_sample_mean = (
            sample - drift
        )  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = (
            prev_sample_mean + diffusion * noise
        )  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)

    def step_correct(
        self,
        model_output: paddle.Tensor,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Correct the predicted sample based on the `model_output` of the network. This
        is often run repeatedly after making the prediction for the previous timestep.

        Args:
            model_output (`paddle.Tensor`):
                The direct output from learned diffusion model.
            sample (`paddle.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`paddle.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`]
                is returned, otherwise a tuple is returned where the first element is
                the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after "
                "creating the scheduler"
            )

        # NOTE(laixinlu) convert sigmas to the dtype of the model output
        if self.sigmas.dtype != model_output.dtype:
            self.sigmas = self.sigmas.cast(model_output.dtype)

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d),
        # where d is the dim. of z" sample noise for correction
        noise = randn_tensor(sample.shape, generator=generator)

        # compute step size from the model_output, the noise, and the snr
        grad_norm = paddle.norm(
            model_output.reshape([model_output.shape[0], -1]), axis=-1
        ).mean()
        noise_norm = paddle.norm(noise.reshape([noise.shape[0], -1]), axis=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * paddle.ones((sample.shape[0],))
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Fix 0D tensor
        if paddle.is_tensor(timesteps) and timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        # Make sure sigmas and timesteps have the same dtype as original_samples
        sigmas = self.discrete_sigmas[timesteps]
        sigmas = sigmas.flatten()
        while len(sigmas.shape) < len(noise.shape):
            sigmas = sigmas.unsqueeze(-1)
        noise = (
            noise * sigmas
            if noise is not None
            else paddle.randn(original_samples.shape, dtype=original_samples.dtype)
            * sigmas
        )
        noisy_samples = noise + original_samples
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps


class ScoreSdeVeSchedulerWrapped(ScoreSdeVeScheduler):
    """
    `ScoreSdeVeScheduler` is a variance exploding stochastic differential
      equation (SDE) scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the
    superclass documentation for the generic methods the library implements for all
    schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        snr (`float`, defaults to 0.15):
            A coefficient weighting the step from the `model_output` sample (from the
            network) to the random noise.
        sigma_min (`float`, defaults to 0.01):
            The initial noise scale for the sigma sequence in the sampling procedure.
            The minimum sigma should mirror the distribution of the data.
        sigma_max (`float`, defaults to 1348.0):
            The maximum value used for the range of continuous timesteps passed into
            the model.
        sampling_eps (`float`, defaults to 1e-5):
            The end value of sampling where timesteps decrease progressively from 1 to
            epsilon.
        correct_steps (`int`, defaults to 1):
            The number of correction steps performed on a produced sample.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            snr=snr,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampling_eps=sampling_eps,
            correct_steps=correct_steps,
        )
        self.discrete_sigmas_norm = sigma_norm(self.discrete_sigmas)

    def step_correct(
        self,
        model_output: paddle.Tensor,
        timestep: float,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        timestep_discrete = (timestep * (len(self.timesteps) - 1)).cast("int64")
        sigma_norm = self.discrete_sigmas_norm[timestep_discrete]
        model_output = model_output * sigma_norm**0.5

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after "
                "creating the scheduler"
            )

        # NOTE(laixinlu) convert sigmas to the dtype of the model output
        if self.sigmas.dtype != model_output.dtype:
            self.sigmas = self.sigmas.cast(model_output.dtype)

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d),
        # where d is the dim. of z" sample noise for correction
        noise = randn_tensor(sample.shape, generator=generator)

        step_size = (
            self.snr
            * (self.discrete_sigmas[timestep_discrete] / self.discrete_sigmas[0]) ** 2
        )
        # compute corrected sample: model_output term and noise term
        # step_size = step_size.flatten()
        # while len(step_size.shape) < len(sample.shape):
        #     step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample - step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def step_pred(
        self,
        model_output: paddle.Tensor,
        timestep: float,
        sample: paddle.Tensor,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        timestep_discrete = (timestep * (len(self.timesteps) - 1)).cast("int64")
        sigma_norm = self.discrete_sigmas_norm[timestep_discrete]
        model_output = model_output * sigma_norm**0.5

        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after "
                "creating the scheduler"
            )

        # timestep = timestep * paddle.ones(
        #     [
        #         sample.shape[0],
        #     ]
        # )  # paddle.repeat_interleave(timestep, sample.shape[0])

        # NOTE(laixinlu) convert sigmas to the dtype of the model output
        if self.discrete_sigmas.dtype != model_output.dtype:
            self.discrete_sigmas = self.discrete_sigmas.cast(model_output.dtype)

        sigma = self.discrete_sigmas[timestep_discrete]
        adjacent_sigma = self.get_adjacent_sigma(timestep_discrete, timestep)
        drift = paddle.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is
        # grad_x log pt(x) also equation 47 shows the analog from SDE models to
        # ancestral sampling methods
        # diffusion = diffusion.flatten()
        # while len(diffusion.shape) < len(sample.shape):
        #     diffusion = diffusion.unsqueeze(-1)
        drift = drift + diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = randn_tensor(sample.shape, generator=generator, dtype=sample.dtype)
        prev_sample_mean = (
            sample - drift
        )  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        diffusion2 = adjacent_sigma / sigma
        # diffusion2 = diffusion2.flatten()
        # while len(diffusion2.shape) < len(sample.shape):
        #     diffusion2 = diffusion2.unsqueeze(-1)

        prev_sample = (
            prev_sample_mean + diffusion * diffusion2 * noise
        )  # add impact of diffusion field g

        if not return_dict:
            return (prev_sample, prev_sample_mean)

        return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean)
