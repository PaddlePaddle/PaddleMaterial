import math

import numpy as np
import paddle
from utils import paddle_aux  # noqa


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


class DiscreteScheduler(paddle.nn.Layer):
    def __init__(self, timesteps, num_classes, forward_type="uniform", eps=1e-06):
        super().__init__()
        self.timesteps = timesteps
        self.eps = eps
        self.num_classes = num_classes
        self.forward_type = forward_type
        q_onestep_mats = []
        q_mats = []
        if forward_type == "uniform":
            steps = paddle.arange(dtype="float64", end=timesteps + 1) / timesteps
            alpha_bar = paddle.cos(x=(steps + 0.008) / 1.008 * 3.1415926 / 2)
            self.beta_t = paddle.minimum(
                x=1 - alpha_bar[1:] / alpha_bar[:-1],
                y=paddle.ones_like(x=alpha_bar[1:]) * 0.999,
            )
            for beta in self.beta_t:
                mat = paddle.ones(shape=[num_classes, num_classes]) * beta / num_classes
                mat.diagonal().fill_(
                    value=1 - (num_classes - 1) * beta.item() / num_classes
                )
                q_onestep_mats.append(mat)
        elif forward_type == "absorbing":
            self.beta_t = 1.0 / paddle.linspace(timesteps, 1.0, timesteps)
            self.mask_id = self.num_classes - 1
            for beta in self.beta_t:
                diag = paddle.full(shape=(self.num_classes,), fill_value=1.0 - beta)
                mat = paddle.diag(diag, offset=0)
                mat[:, self.num_classes - 1] += beta
                q_onestep_mats.append(mat)
        else:
            raise NotImplementedError(
                f'{forward_type} not implemented, use one of ["uniform","absorbing"]'
            )

        q_one_step_mats = paddle.stack(x=q_onestep_mats, axis=0)
        x = q_one_step_mats
        q_one_step_transposed = x.transpose([0, 2, 1])
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.timesteps):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = paddle.stack(x=q_mats, axis=0)
        self.logit_type = "logit"
        self.register_buffer(name="q_one_step_transposed", tensor=q_one_step_transposed)
        self.register_buffer(name="q_mats", tensor=q_mats)
