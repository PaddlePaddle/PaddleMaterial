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

import numpy as np
import paddle
from paddle_scatter import scatter_mean

from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.miad.type_diffusion import D3PM
from ppmat.models.miad.type_diffusion import DDPMOnehot
from ppmat.schedulers import build_scheduler
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal
from ppmat.utils.crystal import lattice_params_to_matrix_paddle

_DEFAULT_GAMMA = {
    "csp_perov5": 5e-7, "gen_perov5": 5e-7,
    "csp_mp20": 1e-5, "gen_mp20": 1e-5,
    "csp_alex_mp20": 1e-5, "gen_alex_mp20": 1e-5,
    "csp_mpts52": 1e-5, "gen_mpts52": 1e-5,
    "csp_carbon24": 5e-7, "gen_carbon24": 1e-5,
}


def parse_num_atoms_to_per_crystal(num_atoms_data):
    if num_atoms_data is None:
        return None
    if hasattr(num_atoms_data, "numpy"):
        num_atoms_np = num_atoms_data.numpy().flatten()
    elif hasattr(num_atoms_data, "reshape"):
        num_atoms_np = num_atoms_data.reshape(-1)
    else:
        num_atoms_np = np.array(num_atoms_data).flatten()
    return paddle.to_tensor(num_atoms_np.astype("int64")), num_atoms_np


def mean_interleave(t, num_repeats):
    batch_idx = paddle.repeat_interleave(
        paddle.arange(num_repeats.shape[0]), num_repeats
    )
    return scatter_mean(t, batch_idx, dim=0)


class CrystalGen:
    """Crystal generation orchestrator.

    Directly uses ppmat schedulers (DDPMScheduler, ScoreSdeVeSchedulerWrapped)
    for coefficient pre-computation, with custom step logic for each diffusion
    type.
    """

    def __init__(self, diffusion_config, logger):
        self.config = diffusion_config
        self.logger = logger
        self.cont_time = self.config["cont_time"]
        self.num_steps = self.config["num_steps"]
        self.eps = 1e-3
        self.time_embedding = SinusoidalTimeEmbeddings(
            self.config.get("time_embed_dim", 256)
        )
        self.gen_type = "gen" in self.config["task"]

        # Lattice diffusion
        lat_cfg = self.config["lat_diffusion"]
        self.lat_method = lat_cfg["method"]
        self.lat_scheduler = build_scheduler(lat_cfg.get("scheduler_cfg", {
            "__class_name__": "DDPMScheduler",
            "__init_params__": {
                "num_train_timesteps": self.num_steps,
                "beta_schedule": "squaredcos_cap_v2",
            },
        }))
        if self.lat_method == "fm_lenang":
            self._lenang2lat = None
            self.gamma_alpha, self.gamma_theta = 1.3, 0.25
            self.angle_difference_bound = 20

        # Frac diffusion
        frac_cfg = self.config["frac_diffusion"]
        self.frac_method = frac_cfg["method"]
        self.frac_scheduler = build_scheduler(frac_cfg.get("scheduler_cfg", {
            "__class_name__": "ScoreSdeVeSchedulerWrapped",
            "__init_params__": {
                "num_train_timesteps": self.num_steps,
                "sigma_min": 0.005,
                "sigma_max": 0.5,
                "sampling_eps": 1e-3,
            },
        }))
        self.step_lr = frac_cfg.get("step_lr", None)
        if self.step_lr is None:
            self.step_lr = _DEFAULT_GAMMA.get(self.config["task"], 1e-5)
        self.sigmas_t = self.frac_scheduler.discrete_sigmas[:, None]
        self.sigmas_norm_t = self.frac_scheduler.discrete_sigmas_norm[:, None]
        self.sb = 0.005

        # Type diffusion
        if self.gen_type:
            type_cfg = self.config["type_diffusion"]
            switch_type = {
                "ddpm_onehot": DDPMOnehot,
                "d3pm": D3PM,
            }
            self.type_diffusion = switch_type[type_cfg["method"]](self.config)
        else:
            self.type_diffusion = None

    def _time_sample(self, batch):
        t = paddle.rand([batch["batch_size"]])
        t = self.eps + (self.num_steps - 1 - self.eps) * t
        if not self.cont_time:
            t = t.round().cast("int64")
        t_per_atom = t.repeat_interleave(batch["num_atoms"])
        return [t, t_per_atom]

    def _time_iterator(self, batch, start_from=-1):
        if start_from == -1:
            start_from = self.num_steps - 1
        for t_val in range(start_from, -1, -1):
            t = paddle.full([batch["batch_size"]], t_val, dtype="float32")
            yield [t, t.repeat_interleave(batch["num_atoms"])]

    def _lenang2lat_fn(self, la):
        if self._lenang2lat is None:
            self._lenang2lat = lambda la: lattice_params_to_matrix_paddle(
                la[:, :3], la[:, 3:]
            )
        return self._lenang2lat(la)

    def forward_step_sample(self, x0, t, batch):
        l0, f0, a0 = x0
        t0_idx = t[0].cast("int64")
        t1_idx = t[1].cast("int64")

        # Lattice forward
        if self.lat_method == "ddpm":
            noise = paddle.randn(l0.shape)
            self.lat_randn = noise
            lt = self.lat_scheduler.add_noise(l0, noise, t0_idx)
        elif self.lat_method == "fm":
            noise = paddle.randn(l0.shape)
            step = (1 + t[0][:, None, None]) / self.num_steps
            lt = (1 - step) * l0 + step * noise
            self.lat_ut = -noise
        elif self.lat_method == "fm_lenang":
            xT = self._prior_lat(batch)
            step = (1 + t[0][:, None, None]) / self.num_steps
            lt = (1 - step) * l0 + step * xT
            self.lat_ut = l0 - xT
        else:
            lt = l0

        # Frac forward
        if self.frac_method == "wrapped_normal":
            noise = paddle.randn(f0.shape)
            self.frac_randn = noise
            ft = (f0 + self.sigmas_t[t1_idx] * noise) % 1.0
        elif self.frac_method == "pfm":
            xT_minus_x0 = paddle.rand(f0.shape) - 0.5
            step = (1 + t[1][:, None]) / self.num_steps
            ft = (f0 + step * xT_minus_x0) % 1.0
            self.frac_ut = -xT_minus_x0
        else:
            ft = f0

        # Type forward
        if self.gen_type:
            at = self.type_diffusion.forward_step_sample(a0, t[1], batch)
        else:
            at = a0

        return [lt, ft, at]

    def reverse_step_sample(self, xt, t, model, batch):
        lt, ft, at = xt

        if self.config["method"] == "DiffCSP":
            _, f_pred, _ = self.model_prediction(xt, t, model, batch)
            ft_05 = self.frac_reverse_part1(f_pred, ft, t[1])
            xt_05 = [lt, ft_05, at]
            l_pred, f_pred, a_pred = self.model_prediction(xt_05, t, model, batch)
            lt_1 = self.lat_reverse(l_pred, lt, t[0])
            ft_1 = self.frac_reverse_part2(f_pred, ft_05, t[1])
        else:
            batch["prediction"] = self.model_prediction(xt, t, model, batch)
            l_pred, f_pred, a_pred = batch["prediction"]
            lt_1 = self.lat_reverse(l_pred, lt, t[0])
            ft_1 = self.frac_reverse(f_pred, ft, t[1])

        at_1 = (
            self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch)
            if self.gen_type else at
        )
        return [lt_1, ft_1, at_1]

    def lat_reverse(self, pred, xt, t):
        t_idx = t.cast("int64")
        if self.lat_method == "ddpm":
            return self.lat_scheduler.step(pred, t_idx[0], xt).prev_sample
        elif self.lat_method == "fm":
            pred = -pred
            step = (1 + t[:, None, None]) / self.num_steps
            x0_pred = (xt - step * pred) / (1 - step)
            vt = x0_pred - pred
            return xt + step / self.num_steps * vt
        elif self.lat_method == "fm_lenang":
            vt = pred
            return xt + vt / self.num_steps
        return xt

    def frac_reverse(self, pred, xt, t):
        t_idx = t.cast("int64")
        if self.frac_method == "wrapped_normal":
            return self.frac_reverse_part2(pred, xt, t)
        elif self.frac_method == "pfm":
            return (xt + pred / self.num_steps) % 1.0
        return xt

    def frac_reverse_part1(self, pred, xt, t):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        step_size = self.step_lr * (st / self.sb) ** 2
        drift = -step_size * pred * paddle.sqrt(snt)
        diffusion = paddle.sqrt(2 * step_size) * paddle.randn(xt.shape)
        return xt + drift + diffusion

    def frac_reverse_part2(self, pred, xt, t):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        st_1 = self.sigmas_t[paddle.maximum(t_idx - 1, paddle.to_tensor(0))]
        snt = self.sigmas_norm_t[t_idx]
        step_size = st ** 2 - st_1 ** 2
        drift = -step_size * pred * paddle.sqrt(snt)
        diffusion = paddle.sqrt(st_1 ** 2 * (st ** 2 - st_1 ** 2) / (st ** 2)) * paddle.randn(xt.shape)
        return (xt + drift + diffusion) % 1.0

    def _prior_lat(self, batch):
        if self.lat_method == "ddpm":
            return paddle.randn([batch["batch_size"], 3, 3], dtype="float32")
        elif self.lat_method == "fm":
            return paddle.randn([batch["batch_size"], 3, 3], dtype="float32")
        elif self.lat_method == "fm_lenang":
            bs = batch["batch_size"]
            la = paddle.zeros([bs, 6], dtype="float32")
            gamma = paddle.distribution.Gamma(
                paddle.to_tensor([self.gamma_alpha], dtype="float32"),
                paddle.to_tensor([self.gamma_theta], dtype="float32"),
            )
            la[:, :3] = 2 + gamma.sample([bs, 3])
            collected = []
            while len(collected) < bs:
                ang = 60 + 60 * paddle.rand([bs, 3])
                check = (
                    (ang[:, 0] + ang[:, 1] - ang[:, 2] > self.angle_difference_bound)
                    * (ang[:, 2] + ang[:, 0] - ang[:, 1] > self.angle_difference_bound)
                    * (ang[:, 1] + ang[:, 2] - ang[:, 0] > self.angle_difference_bound)
                )
                collected.append(ang[check])
            la[:, 3:] = paddle.concat(collected)[:bs]
            return self._lenang2lat_fn(la)
        raise ValueError(f"Unknown lat_method: {self.lat_method}")

    def _prior_frac(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum()) if na.ndim > 0 else int(na)
        if self.frac_method == "wrapped_normal":
            return paddle.rand([total, 3], dtype="float32")
        elif self.frac_method == "pfm":
            return paddle.rand([total, 3])
        raise ValueError(f"Unknown frac_method: {self.frac_method}")

    def prior_sample(self, batch):
        prior_at = (
            self.type_diffusion.prior_sample(batch)
            if self.gen_type else batch["atom_types"]
        )
        return [self._prior_lat(batch), self._prior_frac(batch), prior_at]

    def model_prediction(self, xt, t, model, batch):
        lt, ft, at = xt
        time_emb = self.time_embedding(1000 * (t[0] / self.num_steps) + 1)
        nn_pred = model(time_emb, at, ft, lt, batch["num_atoms"], batch["batch_idx"])
        return [
            nn_pred[0],
            nn_pred[1],
            nn_pred[2] if self.gen_type else None,
        ]

    def train_step(self, batch, model, mode):
        batch["t"] = self._time_sample(batch)
        batch["xt"] = self.forward_step_sample(batch["x0"], batch["t"], batch)
        batch["prediction"] = self.model_prediction(batch["xt"], batch["t"], model, batch)

        # Lattice loss
        if self.lat_method == "ddpm":
            loss_lat = ((batch["prediction"][0] - self.lat_randn) ** 2).reshape([-1, 9]).mean(axis=1).mean()
        elif self.lat_method in ("fm", "fm_lenang"):
            loss_lat = ((batch["prediction"][0] - self.lat_ut) ** 2).reshape([-1, 9]).mean(axis=1).mean()
        else:
            loss_lat = paddle.to_tensor(0.0)

        # Frac loss
        if self.frac_method == "wrapped_normal":
            t_idx = batch["t"][1].cast("int64")
            st = self.sigmas_t[t_idx]
            snt = self.sigmas_norm_t[t_idx]
            normed_score = d_log_p_wrapped_normal(st * self.frac_randn, st) / paddle.sqrt(snt)
            loss_frac = ((batch["prediction"][1] - normed_score) ** 2).reshape([-1, 3]).mean(axis=1).mean()
        elif self.frac_method == "pfm":
            loss_frac = ((batch["prediction"][1] - self.frac_ut) ** 2).reshape([-1, 3]).mean(axis=1).mean() * 10
        else:
            loss_frac = paddle.to_tensor(0.0)

        batch["loss"] = loss_lat + loss_frac

        if self.gen_type:
            loss_type = self.type_diffusion.loss(batch).mean()
            batch["loss"] = batch["loss"] + loss_type

        if self.logger is not None:
            t_log = paddle.clip(batch["t"][0].clone().cast("int64"), 0, 999)
            self.logger.add(f"loss:lattice:{mode}", loss_lat.item(), stack_after_epoch=True)
            self.logger.add(f"loss:coord:{mode}", loss_frac.item(), stack_after_epoch=True)
            loss_lat_t = paddle.zeros([self.num_steps], dtype=loss_lat.dtype)
            loss_lat_t[t_log] = batch["prediction"][0].reshape([-1, 9]).mean(axis=1)
            self.logger.add(f"loss:lattice4time:{mode}", loss_lat_t, stack_after_epoch=True)
            loss_frac_t = paddle.zeros([self.num_steps], dtype=loss_frac.dtype)
            loss_frac_t[t_log] = mean_interleave(
                batch["prediction"][1].reshape([-1, 3]).mean(axis=1), batch["num_atoms"],
            )
            self.logger.add(f"loss:coord4time:{mode}", loss_frac_t, stack_after_epoch=True)
            if self.gen_type:
                self.logger.add(f"loss:type:{mode}", loss_type.item(), stack_after_epoch=True)

        return batch

    def sampling_procedure(self, model, batch, progress_printer):
        batch["xt"] = self.prior_sample(batch)
        for t_vec in self._time_iterator(batch, start_from=self.num_steps - 1):
            batch["t"] = t_vec
            progress_printer(batch["t"][0][0].item())
            batch["xt"] = self.reverse_step_sample(batch["xt"], batch["t"], model, batch)
        batch["xt"] = self.output_transform(batch["xt"], batch)
        batch["x0_prediction"] = batch["xt"]
        return batch

    def output_transform(self, x0, batch):
        _out = lambda diff, x: diff.output_transform(x, batch) if hasattr(diff, 'output_transform') else x
        return [
            x0[0],
            x0[1],
            _out(self.type_diffusion, x0[2]) if self.gen_type else x0[2],
        ]
