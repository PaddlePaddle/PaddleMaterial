import copy
import math
from typing import Any
from typing import Dict

import numpy as np
import paddle
import paddle.nn as nn
from models import initializer
from models.cspnet import CSPNet
from models.noise_schedule import BetaScheduler
from models.noise_schedule import SigmaScheduler
from models.noise_schedule import d_log_p_wrapped_normal
from models.time_embedding import SinusoidalTimeEmbeddings
from tqdm import tqdm
from utils.crystal import lattice_params_to_matrix_paddle

MAX_ATOMIC_NUM = 100


class CSPDiffusionWithGuidance(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        cost_type,
        property_input_dim=2,
        property_dim=512,
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

        self.property_embedding = paddle.nn.Linear(property_input_dim, property_dim)
        self.drop_prob = 0.1

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
        # import pdb;pdb.set_trace()
        batch_size = batch["num_graphs"]
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        property_emb = self.property_embedding(batch["prop"])
        property_mask = paddle.bernoulli(paddle.zeros([batch_size, 1]) + self.drop_prob)
        property_mask = 1 - property_mask

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
            property_emb=property_emb,
            property_mask=property_mask,
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
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-05, guide_w=0.5):
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

        property_emb = self.property_embedding(batch["prop"])
        property_mask = paddle.zeros([batch_size * 2, 1])
        property_mask[:batch_size] = 1

        num_atoms_double = paddle.concat([batch["num_atoms"], batch["num_atoms"]])
        batch_double = paddle.concat([batch["batch"], batch["batch"] + batch_size])
        property_emb_double = paddle.concat([property_emb, property_emb], axis=0)

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

            # double concat it
            time_emb_double = paddle.concat([time_emb, time_emb], axis=0)
            t_t_double = paddle.concat([t_t, t_t], axis=0)
            x_t_double = paddle.concat([x_t, x_t], axis=0)
            l_t_double = paddle.concat([l_t, l_t], axis=0)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                t_t_double,
                x_t_double,
                l_t_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[: batch["num_nodes"]] - guide_w * pred_x[
                batch["num_nodes"] :
            ]
            pred_t = (1 + guide_w) * pred_t[: batch["num_nodes"]] - guide_w * pred_t[
                batch["num_nodes"] :
            ]

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

            # double concat it
            t_t_minus_05_double = paddle.concat([t_t_minus_05, t_t_minus_05], axis=0)
            x_t_minus_05_double = paddle.concat([x_t_minus_05, x_t_minus_05], axis=0)
            l_t_minus_05_double = paddle.concat([l_t_minus_05, l_t_minus_05], axis=0)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                t_t_minus_05_double,
                x_t_minus_05_double,
                l_t_minus_05_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[: batch["num_nodes"]] - guide_w * pred_x[
                batch["num_nodes"] :
            ]
            pred_t = (1 + guide_w) * pred_t[: batch["num_nodes"]] - guide_w * pred_t[
                batch["num_nodes"] :
            ]

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
