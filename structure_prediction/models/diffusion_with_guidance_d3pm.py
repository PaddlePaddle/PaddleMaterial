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
from models.diff_utils import DiscreteScheduler
from models.diff_utils import SigmaScheduler
from models.diff_utils import d_log_p_wrapped_normal
from tqdm import tqdm
from utils.crystal import lattice_params_to_matrix_paddle

MAX_ATOMIC_NUM = 100


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


class CSPDiffusionWithGuidanceD3PM(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        discrete_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        cost_type,
        property_input_dim=2,
        property_dim=512,
        pretrained=None,
        num_classes=100,
        **kwargs,
    ) -> None:
        super().__init__()
        self.decoder = CSPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)
        self.discrete_scheduler = DiscreteScheduler(**discrete_scheduler_cfg)

        self.time_dim = time_dim
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.cost_type = cost_type
        self.num_classes = num_classes

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

    def _at(self, a, t, x):
        t = t.reshape((t.shape[0], *([1] * (x.dim() - 1))))
        return a[t - 1, x, :]

    def q_sample(self, x_0, t, noise):
        logits = paddle.log(
            x=self._at(self.discrete_scheduler.q_mats, t, x_0)
            + self.discrete_scheduler.eps
        )
        noise = paddle.clip(x=noise, min=self.discrete_scheduler.eps, max=1.0)
        gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
        return paddle.argmax(x=logits + gumbel_noise, axis=-1)

    def q_posterior_logits(self, x_0, x_t, t):

        if x_0.dtype == paddle.int64 or x_0.dtype == paddle.int32:
            x_0_logits = paddle.log(
                x=nn.functional.one_hot(num_classes=self.num_classes, x=x_0).astype(
                    "int64"
                )
                + self.discrete_scheduler.eps
            )
        else:
            x_0_logits = x_0.clone()
        assert tuple(x_0_logits.shape) == tuple(x_t.shape) + (self.num_classes,)

        fact1 = self._at(self.discrete_scheduler.q_one_step_transposed, t, x_t)
        softmaxed = nn.functional.softmax(x=x_0_logits, axis=-1)
        index = t - 2
        index = paddle.where(condition=index < 0, x=index + self.num_classes, y=index)
        qmats2 = self.discrete_scheduler.q_mats[index].cast(softmaxed.dtype)

        fact2 = paddle.einsum("bc,bcd->bd", softmaxed, qmats2)
        out = paddle.log(x=fact1 + self.discrete_scheduler.eps) + paddle.log(
            x=fact2 + self.discrete_scheduler.eps
        )
        t_broadcast = t.reshape((t.shape[0], *([1] * x_t.dim())))
        bc = paddle.where(condition=t_broadcast == 1, x=x_0_logits, y=out)
        return bc

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_axis=0, stop_axis=-2)
        dist2 = dist2.flatten(start_axis=0, stop_axis=-2)
        out = nn.functional.softmax(x=dist1 + self.discrete_scheduler.eps, axis=-1) * (
            nn.functional.log_softmax(dist1 + self.discrete_scheduler.eps, axis=-1)
            - nn.functional.log_softmax(dist2 + self.discrete_scheduler.eps, axis=-1)
        )
        return out.sum(axis=-1).mean()

    def forward(self, batch):
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

        atom_types = batch["atom_types"] - 1
        atom_types_times = times.repeat_interleave(repeats=batch["num_atoms"])
        input_atom_types = self.q_sample(
            atom_types,
            paddle.cast(atom_types_times, dtype="int64"),
            paddle.rand(shape=(*atom_types.shape, self.num_classes)),
        )
        input_atom_types += 1
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x, pred_t = self.decoder(
            time_emb,
            input_atom_types,
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

        true_q_posterior_logits = self.q_posterior_logits(
            atom_types,
            input_atom_types - 1,
            paddle.cast(atom_types_times, dtype="int64"),
        )
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_t, input_atom_types - 1, paddle.cast(atom_types_times, dtype="int64")
        )
        loss_type_vb = self.vb(true_q_posterior_logits, pred_q_posterior_logits)
        loss_type_ce = nn.functional.cross_entropy(pred_t, atom_types)
        loss_type = loss_type_vb + 0.01 * loss_type_ce

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
            "loss_type_vb": loss_type_vb,
            "loss_type_ce": loss_type_ce,
        }

    def p_sample(self, x, t, cond, noise):
        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)
        noise = paddle.clip(x=noise, min=self.eps, max=1.0)
        not_first_step = (
            (t != 1).astype(dtype="float32").reshape((x.shape[0], *([1] * x.dim())))
        )
        gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
        sample = paddle.argmax(
            x=pred_q_posterior_logits + gumbel_noise * not_first_step, axis=-1
        )
        return sample

    @paddle.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-05, guide_w=0.5):
        batch_size = batch["num_graphs"]
        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]).to(self.device), paddle.rand(
            shape=[batch["num_nodes"], 3]
        ).to(self.device)
        t_T = paddle.randint(low=0, high=self.num_classes, shape=[batch["num_nodes"]])

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

            t_t += 1

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
            noise = paddle.rand(shape=(*t_t_minus_05.shape, self.num_classes))
            atom_types_times = times.repeat_interleave(repeats=batch["num_atoms"])
            pred_q_posterior_logits = self.q_posterior_logits(
                pred_t, t_t_minus_05 - 1, atom_types_times.cast("int64")
            )
            noise = paddle.clip(x=noise, min=self.discrete_scheduler.eps, max=1.0)
            not_first_step = (
                (atom_types_times != 1)
                .astype(dtype="float32")
                .reshape((t_t_minus_05.shape[0], *([1] * t_t_minus_05.dim())))
            )
            gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
            sample = paddle.argmax(
                x=pred_q_posterior_logits + gumbel_noise * not_first_step, axis=-1
            )
            t_t_minus_1 = sample

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
