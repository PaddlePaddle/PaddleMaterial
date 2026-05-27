# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-computed coefficient schedulers for MiAD-style diffusion models.

These functions return raw coefficient tensors that diffusion modules index into
during forward/reverse steps. This is a lighter-weight alternative to the class-based
scheduler interface (DDPMScheduler, ScoreSdeVeScheduler, etc.) used by DiffCSP.
"""

import math

import numpy as np
import paddle

from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal
from ppmat.schedulers.scheduling_sde_ve import p_wrapped_normal  # noqa: F401


def scheduler(scheduler_name, num_steps):
    """Factory for pre-computed diffusion coefficient tensors.

    Args:
        scheduler_name: One of 'diffcsp_cosine', 'cosine', 'default_d3pm',
            'default_wrapped_normal'.
        num_steps: Number of diffusion steps.

    Returns:
        Tuple of coefficient tensors. The format depends on the scheduler type:
        - diffcsp_cosine: (cumprod_alphas_t, None)
        - cosine: (cumprod_alphas_t, a_t_fn)
        - default_d3pm: (Q_t, cumprod_Q_t)
        - default_wrapped_normal: (sigmas, sigmas_norm, sigma_begin,
            sigma_end, d_log_p_fn)
    """
    if scheduler_name == "diffcsp_cosine":
        return _scheduler_diffcsp_cosine(num_steps)
    elif scheduler_name == "cosine":
        return _scheduler_cosine(num_steps)
    elif "default_d3pm" in scheduler_name:
        return _scheduler_default_d3pm(num_steps)
    elif scheduler_name == "default_wrapped_normal":
        return _scheduler_default_wrapped_normal(num_steps)

    raise NotImplementedError(f"Scheduler '{scheduler_name}' not implemented")


def _scheduler_diffcsp_cosine(num_steps):
    s = 0.008
    discretization = paddle.linspace(0, num_steps, num_steps + 1)
    alphas_cumprod = (
        paddle.cos(((discretization / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = paddle.clip(betas, 0.0001, 0.9999)
    alphas = 1.0 - betas
    cumprod_alphas_t = paddle.cumprod(alphas, dim=0).cast("float32")
    cumprod_alphas_t = paddle.concat([paddle.to_tensor([1.0]), cumprod_alphas_t])
    return cumprod_alphas_t, None


def _scheduler_cosine(num_steps):
    s = 0.008

    def f_t(t):
        return paddle.cos((t / (num_steps + 1) + s) / (1 + s) * math.pi / 2) ** 2

    def a_t(t):
        return f_t(t) / f_t(paddle.to_tensor(0.0, dtype="float64"))

    discretization = paddle.linspace(0, num_steps, num_steps + 1)
    cumprod_alphas_t = a_t(discretization).cast("float32")
    return cumprod_alphas_t, a_t


def _scheduler_default_d3pm(num_steps, vocab_size=100):
    def get_uniform_transition_mat(vocab_size, beta_t):
        mat = paddle.full((vocab_size, vocab_size), beta_t / float(vocab_size))
        diag_val = 1 - beta_t * (vocab_size - 1) / vocab_size
        for i in range(vocab_size):
            mat[i, i] = diag_val.cast("float32")
        return mat

    s = 0.008

    def f_t(t):
        return paddle.cos((t / (num_steps + 1) + s) / (1 + s) * math.pi / 2)

    def a_t(t):
        return f_t(t) / f_t(paddle.to_tensor(0.0, dtype="float64"))

    discretization = paddle.arange(1, num_steps + 1, dtype="float64")
    cumprod_alphas_t = a_t(discretization)
    cumprod_alphas_t_1 = paddle.concat(
        [paddle.to_tensor([1.0], dtype="float64"), cumprod_alphas_t[:-1]]
    )
    betas_t = 1 - cumprod_alphas_t / cumprod_alphas_t_1

    Q_t_list = []
    for t_idx in range(num_steps):
        Q_t_list.append(get_uniform_transition_mat(vocab_size, betas_t[t_idx]))
    Q_t = paddle.stack(Q_t_list, axis=0)

    cumprod_Q_t_list = [Q_t[0]]
    for t_idx in range(1, num_steps):
        cumprod_Q_t_list.append(paddle.matmul(cumprod_Q_t_list[-1], Q_t[t_idx]))
    cumprod_Q_t = paddle.stack(cumprod_Q_t_list, axis=0)

    return Q_t, cumprod_Q_t


def _scheduler_default_wrapped_normal(num_steps):
    """Wrapped normal scheduler for fractional coordinate diffusion.

    Uses sn=100000 for sigma_norm calculation (differs from scheduling_sde_ve.py
    which uses sn=10000). This larger sample size was used in the original MiAD
    implementation for more precise expected score norm estimation.
    """
    sigma_begin, sigma_end = 0.005, 0.5
    sigmas = paddle.to_tensor(
        np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_steps)),
        dtype="float32",
    )

    def sigma_norm(sigma, T=1.0, sn=100000):
        sigmas_expanded = paddle.tile(sigma[None, :], [sn, 1])
        x_sample = sigma * paddle.randn(sigmas_expanded.shape)
        x_sample = x_sample % T
        normal_ = d_log_p_wrapped_normal(x_sample, sigmas_expanded, T=T)
        return (normal_**2).mean(axis=0)

    sigmas_norm_ = sigma_norm(sigmas)
    return sigmas, sigmas_norm_, sigma_begin, sigma_end, d_log_p_wrapped_normal
