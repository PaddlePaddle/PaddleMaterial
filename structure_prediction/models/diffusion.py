import copy
import math
from typing import Any
from typing import Dict

import numpy as np
import paddle
import paddle.nn as nn
from models import initializer
from models.cspnet import CSPNet
from models.diff_utils import BetaScheduler
from models.diff_utils import SigmaScheduler
from models.diff_utils import d_log_p_wrapped_normal
from tqdm import tqdm

MAX_ATOMIC_NUM = 100


def lattice_params_to_matrix_paddle(lengths, angles):
    """Batched paddle version to compute lattice matrix from params.

    lengths: paddle.Tensor of shape (N, 3), unit A
    angles: paddle.Tensor of shape (N, 3), unit degree
    """
    angles_r = paddle.deg2rad(x=angles)
    coses = paddle.cos(x=angles_r)
    sins = paddle.sin(x=angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(x=val, min=-1.0, max=1.0)
    gamma_star = paddle.acos(x=val)
    vector_a = paddle.stack(
        x=[
            lengths[:, 0] * sins[:, 1],
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 0] * coses[:, 1],
        ],
        axis=1,
    )
    vector_b = paddle.stack(
        x=[
            -lengths[:, 1] * sins[:, 0] * paddle.cos(x=gamma_star),
            lengths[:, 1] * sins[:, 0] * paddle.sin(x=gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        axis=1,
    )
    vector_c = paddle.stack(
        x=[
            paddle.zeros(shape=lengths.shape[0]),
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 2],
        ],
        axis=1,
    )
    return paddle.stack(x=[vector_a, vector_b, vector_c], axis=1)


class SinusoidalTimeEmbeddings(paddle.nn.Layer):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.place
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(x=paddle.arange(end=half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = paddle.concat(x=(embeddings.sin(), embeddings.cos()), axis=-1)
        return embeddings


class CSPDiffusion(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        pretrained=None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decoder = CSPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)

        self.time_dim = time_dim
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord

        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.cost_lattice < 1e-05
        self.keep_coords = self.cost_coord < 1e-05
        self.device = paddle.device.get_device()
        self.pretrained = pretrained
        self.apply(self._init_weights)
        if self.pretrained is not None:
            self.set_dict(paddle.load(self.pretrained))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch):
        batch_size = batch["num_graphs"]
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[
            paddle.cast(times, dtype="int32")
        ]
        beta = self.beta_scheduler.betas[paddle.cast(times, dtype="int32")]
        c0 = paddle.sqrt(x=alphas_cumprod)
        c1 = paddle.sqrt(x=1.0 - alphas_cumprod)
        sigmas = self.sigma_scheduler.sigmas[paddle.cast(times, dtype="int32")]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[
            paddle.cast(times, dtype="int32")
        ]

        lattices = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        frac_coords = batch["frac_coords"]
        rand_l, rand_x = paddle.randn(
            shape=lattices.shape, dtype=lattices.dtype
        ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(repeats=batch["num_atoms"])[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            repeats=batch["num_atoms"]
        )[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x = self.decoder(
            time_emb,
            batch["atom_types"],
            input_frac_coords,
            input_lattice,
            batch["num_atoms"],
            batch["batch"],
        )

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)
        loss_lattice = paddle.nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = paddle.nn.functional.mse_loss(input=pred_x, label=tar_x)
        loss = self.cost_lattice * loss_lattice + self.cost_coord * loss_coord
        return {"loss": loss, "loss_lattice": loss_lattice, "loss_coord": loss_coord}

    @paddle.no_grad()
    def sample(self, batch, step_lr=1e-05):
        batch_size = batch["num_graphs"]
        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]).to(self.device), paddle.rand(
            shape=[batch["num_nodes"], 3]
        ).to(self.device)
        if self.keep_coords:
            x_T = batch["frac_coords"]
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        time_start = self.beta_scheduler.timesteps
        traj = {
            time_start: {
                "num_atoms": batch["num_atoms"],
                "atom_types": batch["atom_types"],
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
            }
        }
        for t in tqdm(range(time_start, 0, -1)):
            times = paddle.full(shape=(batch_size,), fill_value=t)
            time_emb = self.time_embedding(times)
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / paddle.sqrt(x=alphas)
            c1 = (1 - alphas) / paddle.sqrt(x=1 - alphas_cumprod)
            x_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = paddle.sqrt(x=2 * step_size)
            pred_l, pred_x = self.decoder(
                time_emb,
                batch["atom_types"],
                x_t,
                l_t,
                batch["num_atoms"],
                batch["batch"],
            )
            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_05 = (
                x_t - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_05 = l_t if not self.keep_lattice else l_t
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = paddle.sqrt(
                x=adjacent_sigma_x**2
                * (sigma_x**2 - adjacent_sigma_x**2)
                / sigma_x**2
            )
            pred_l, pred_x = self.decoder(
                time_emb,
                batch["atom_types"],
                x_t_minus_05,
                l_t_minus_05,
                batch["num_atoms"],
                batch["batch"],
            )
            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_1 = (
                x_t_minus_05 - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_1 = (
                c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                if not self.keep_lattice
                else l_t
            )
            traj[t - 1] = {
                "num_atoms": batch["num_atoms"],
                "atom_types": batch["atom_types"],
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
            }
        traj_stack = {
            "num_atoms": batch["num_atoms"],
            "atom_types": batch["atom_types"],
            "all_frac_coords": paddle.stack(
                x=[traj[i]["frac_coords"] for i in range(time_start, -1, -1)]
            ),
            "all_lattices": paddle.stack(
                x=[traj[i]["lattices"] for i in range(time_start, -1, -1)]
            ),
        }
        return traj[0], traj_stack


class CSPDiffusionWithType(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        cost_type,
        pretrained=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.decoder = CSPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)

        self.time_dim = time_dim
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.cost_type = cost_type

        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.cost_lattice < 1e-05
        self.keep_coords = self.cost_coord < 1e-05
        self.device = paddle.device.get_device()
        self.pretrained = pretrained
        self.apply(self._init_weights)
        if self.pretrained is not None:
            self.set_dict(paddle.load(self.pretrained))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch):
        batch_size = batch["num_graphs"]
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[
            paddle.cast(times, dtype="int32")
        ]
        beta = self.beta_scheduler.betas[paddle.cast(times, dtype="int32")]
        c0 = paddle.sqrt(x=alphas_cumprod)
        c1 = paddle.sqrt(x=1.0 - alphas_cumprod)
        sigmas = self.sigma_scheduler.sigmas[paddle.cast(times, dtype="int32")]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[
            paddle.cast(times, dtype="int32")
        ]
        lattices = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        frac_coords = batch["frac_coords"]
        rand_l, rand_x = paddle.randn(
            shape=lattices.shape, dtype=lattices.dtype
        ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(repeats=batch["num_atoms"])[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            repeats=batch["num_atoms"]
        )[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0
        gt_atom_types_onehot = (
            paddle.nn.functional.one_hot(
                num_classes=MAX_ATOMIC_NUM, x=batch["atom_types"] - 1
            )
            .astype("int64")
            .astype(dtype="float32")
        )
        rand_t = paddle.randn(
            shape=gt_atom_types_onehot.shape, dtype=gt_atom_types_onehot.dtype
        )
        atom_type_probs = (
            c0.repeat_interleave(repeats=batch["num_atoms"])[:, None]
            * gt_atom_types_onehot
            + c1.repeat_interleave(repeats=batch["num_atoms"])[:, None] * rand_t
        )
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x, pred_t = self.decoder(
            time_emb,
            atom_type_probs,
            input_frac_coords,
            input_lattice,
            batch["num_atoms"],
            batch["batch"],
        )
        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)
        loss_lattice = paddle.nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = paddle.nn.functional.mse_loss(input=pred_x, label=tar_x)
        loss_type = paddle.nn.functional.mse_loss(input=pred_t, label=rand_t)
        loss = (
            self.cost_lattice * loss_lattice
            + self.cost_coord * loss_coord
            + self.cost_type * loss_type
        )
        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_type": loss_type,
        }

    @paddle.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-05):
        batch_size = batch["num_graphs"]
        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]).to(self.device), paddle.rand(
            shape=[batch["num_nodes"], 3]
        ).to(self.device)
        t_T = paddle.randn(shape=[batch["num_nodes"], MAX_ATOMIC_NUM]).to(self.device)
        if self.keep_coords:
            x_T = batch["frac_coords"]
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        traj = {
            self.beta_scheduler.timesteps: {
                "num_atoms": batch["num_atoms"],
                "atom_types": t_T,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
            }
        }
        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times = paddle.full(shape=(batch_size,), fill_value=t)
            time_emb = self.time_embedding(times)
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / paddle.sqrt(x=alphas)
            c1 = (1 - alphas) / paddle.sqrt(x=1 - alphas_cumprod)
            x_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            t_t = traj[t]["atom_types"]
            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
            rand_t = (
                paddle.randn(shape=t_T.shape, dtype=t_T.dtype)
                if t > 1
                else paddle.zeros_like(x=t_T)
            )
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = paddle.sqrt(x=2 * step_size)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb, t_t, x_t, l_t, batch["num_atoms"], batch["batch"]
            )
            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_05 = (
                x_t - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
            rand_t = (
                paddle.randn(shape=t_T.shape, dtype=t_T.dtype)
                if t > 1
                else paddle.zeros_like(x=t_T)
            )
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = paddle.sqrt(
                x=adjacent_sigma_x**2
                * (sigma_x**2 - adjacent_sigma_x**2)
                / sigma_x**2
            )
            pred_l, pred_x, pred_t = self.decoder(
                time_emb,
                t_t_minus_05,
                x_t_minus_05,
                l_t_minus_05,
                batch["num_atoms"],
                batch["batch"],
            )
            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_1 = (
                x_t_minus_05 - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_1 = (
                c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                if not self.keep_lattice
                else l_t
            )
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t
            traj[t - 1] = {
                "num_atoms": batch["num_atoms"],
                "atom_types": t_t_minus_1,
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
            }
        traj_stack = {
            "num_atoms": batch["num_atoms"],
            "atom_types": paddle.stack(
                x=[
                    traj[i]["atom_types"]
                    for i in range(self.beta_scheduler.timesteps, -1, -1)
                ]
            ).argmax(axis=-1)
            + 1,
            "all_frac_coords": paddle.stack(
                x=[
                    traj[i]["frac_coords"]
                    for i in range(self.beta_scheduler.timesteps, -1, -1)
                ]
            ),
            "all_lattices": paddle.stack(
                x=[
                    traj[i]["lattices"]
                    for i in range(self.beta_scheduler.timesteps, -1, -1)
                ]
            ),
        }
        return traj[0], traj_stack
