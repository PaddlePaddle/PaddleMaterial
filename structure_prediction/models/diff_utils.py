import math
import sys

import numpy as np
import paddle
from utils import paddle_aux


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = paddle.linspace(start=0, stop=timesteps, num=steps)
    alphas_cumprod = paddle.cos(x=(x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return paddle.clip(x=betas, min=0.0001, max=0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return paddle.linspace(start=beta_start, stop=beta_end, num=timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return (
        paddle.linspace(start=beta_start**0.5, stop=beta_end**0.5, num=timesteps)
        ** 2
    )


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = paddle.linspace(start=-6, stop=6, num=timesteps)
    return paddle.nn.functional.sigmoid(x=betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (
            (x + T * i)
            / sigma**2
            * paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
        )
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].tile([sn, 1])
    x_sample = sigma * paddle.randn(shape=sigmas.shape, dtype=sigmas.dtype)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(axis=0)


class BetaScheduler(paddle.nn.Layer):
    def __init__(self, timesteps, scheduler_mode, beta_start=0.0001, beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        betas = paddle.concat(x=[paddle.zeros(shape=[1]), betas], axis=0)
        alphas = 1.0 - betas
        alphas_cumprod = paddle.cumprod(alphas, dim=0)
        sigmas = paddle.zeros_like(x=betas)
        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )
        sigmas = paddle.sqrt(x=sigmas)
        self.register_buffer(name="betas", tensor=betas)
        self.register_buffer(name="alphas", tensor=alphas)
        self.register_buffer(name="alphas_cumprod", tensor=alphas_cumprod)
        self.register_buffer(name="sigmas", tensor=sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return paddle.to_tensor(data=ts, dtype="float32").to(device)


class SigmaScheduler(paddle.nn.Layer):
    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = paddle.to_tensor(
            data=np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)),
            dtype="float32",
        )
        sigmas_norm_ = sigma_norm(sigmas)
        self.register_buffer(
            name="sigmas",
            tensor=paddle.concat(x=[paddle.zeros(shape=[1]), sigmas], axis=0),
        )
        self.register_buffer(
            name="sigmas_norm",
            tensor=paddle.concat(x=[paddle.ones(shape=[1]), sigmas_norm_], axis=0),
        )

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return paddle.to_tensor(data=ts).to(device)
