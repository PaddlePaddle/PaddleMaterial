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

import math

import numpy
import paddle

from ppmat.models.diffnmr.utils.diffprior_utils import default
from ppmat.models.diffnmr.utils.diffprior_utils import first
from ppmat.models.diffnmr.utils.diffprior_utils import log


class NoiseScheduler(paddle.nn.Layer):
    def __init__(
        self,
        *,
        beta_schedule,
        timesteps,
        loss_type,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / paddle.linspace(start=timesteps, stop=1, num=timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()
        alphas = 1.0 - betas
        alphas_cumprod = paddle.cumprod(alphas, dim=0)
        alphas_cumprod_prev = paddle.nn.functional.pad(
            x=alphas_cumprod[:-1], pad=(1, 0), value=1.0, pad_from_left_axis=False
        )
        (timesteps,) = tuple(betas.shape)
        self.num_timesteps = int(timesteps)
        if loss_type == "l1":
            loss_fn = paddle.nn.functional.l1_loss
        elif loss_type == "l2":
            loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "huber":
            loss_fn = paddle.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError()
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        register_buffer = lambda name, val: self.register_buffer(  # noqa
            name=name, tensor=val.to("float32")
        )
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", paddle.sqrt(x=alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", paddle.sqrt(x=1.0 - alphas_cumprod)
        )
        register_buffer(
            "log_one_minus_alphas_cumprod", paddle.log(x=1.0 - alphas_cumprod)
        )
        register_buffer(
            "sqrt_recip_alphas_cumprod", paddle.sqrt(x=1.0 / alphas_cumprod)
        )
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", paddle.sqrt(x=1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            paddle.log(x=posterior_variance.clip(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * paddle.sqrt(x=alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * paddle.sqrt(x=alphas)
            / (1.0 - alphas_cumprod),
        )
        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def sample_random_times(self, batch):
        return paddle.randint(
            low=0, high=self.num_timesteps, shape=(batch,), dtype="int64"
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, tuple(x_t.shape)) * x_start
            + extract(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, tuple(x_t.shape))
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, tuple(x_t.shape)
        )
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def q_sample(self, x_start, t, noise=None):
        noise = default(
            noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype)
        )
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape))
            * noise
        )

    def calculate_v(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape))
            * x_start
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = tuple(x_from.shape)
        noise = default(
            noise, lambda: paddle.randn(shape=x_from.shape, dtype=x_from.dtype)
        )
        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)
        return (
            x_from * (alpha_next / alpha)
            + noise * (sigma_next * alpha - sigma * alpha_next) / alpha
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_t.shape)) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape))

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, tuple(loss.shape))


def extract(a, t, x_shape):
    b, *_ = tuple(t.shape)
    out = a.take_along_axis(axis=-1, indices=t, broadcast=False)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def meanflat(x):
    return x.mean(axis=tuple(range(1, len(tuple(x.shape)))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(x=logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * paddle.exp(x=-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0
        + paddle.nn.functional.tanh(x=(2.0 / math.pi) ** 0.5 * (x + 0.044715 * x**3))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert tuple(x.shape) == tuple(means.shape) == tuple(log_scales.shape)
    eps = 1e-12 if x.dtype == "float32" else 0.001
    centered_x = x - means
    inv_stdv = paddle.exp(x=-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps=eps)
    log_one_minus_cdf_min = log(1.0 - cdf_min, eps=eps)
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        condition=x < -thres,
        x=log_cdf_plus,
        y=paddle.where(
            condition=x > thres, x=log_one_minus_cdf_min, y=log(cdf_delta, eps=eps)
        ),
    )
    return log_probs


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = paddle.linspace(start=0, stop=timesteps, num=steps, dtype="float64")
    alphas_cumprod = paddle.cos(x=(x / timesteps + s) / (1 + s) * numpy.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return paddle.clip(x=betas, min=0, max=0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return paddle.linspace(
        start=beta_start, stop=beta_end, num=timesteps, dtype="float64"
    )


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        paddle.linspace(
            start=beta_start**0.5,
            stop=beta_end**0.5,
            num=timesteps,
            dtype="float64",
        )
        ** 2
    )


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = paddle.linspace(start=-6, stop=6, num=timesteps, dtype="float64")
    return paddle.nn.functional.sigmoid(x=betas) * (beta_end - beta_start) + beta_start
