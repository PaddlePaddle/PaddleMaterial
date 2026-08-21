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

from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.miad.type_diffusion import build_type_diffusion
from ppmat.schedulers import build_scheduler
from ppmat.schedulers.scheduling_sde_ve import d_log_p_wrapped_normal

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


class CrystalGen:
    """Crystal generation orchestrator: coefficient pre-computation via ppmat
    schedulers, custom step logic per diffusion type.
    """

    def __init__(self, diffusion_config):
        self.config = diffusion_config
        self.cont_time = self.config["cont_time"]
        self.num_steps = self.config["num_steps"]
        self.eps = 1e-3
        self.time_embedding = SinusoidalTimeEmbeddings(
            self.config.get("time_embed_dim", 256)
        )
        # gen_* tasks also diffuse atom types; csp_* tasks do not
        self.gen_type = self.config["task"].startswith("gen")

        # Lattice diffusion
        lat_cfg = self.config["lat_diffusion"]
        self.lat_scheduler = build_scheduler(lat_cfg["scheduler_cfg"])

        # Frac diffusion
        frac_cfg = self.config["frac_diffusion"]
        self.frac_scheduler = build_scheduler(frac_cfg["scheduler_cfg"])
        self.step_lr = frac_cfg.get("step_lr")
        if self.step_lr is None:
            raise ValueError(
                "frac_diffusion.step_lr must be provided in the diffusion config "
                "(the Langevin step-size coefficient, e.g. 1e-5 for gen_mp20)"
            )
        self.sigmas_t = self.frac_scheduler.discrete_sigmas[:, None]
        self.sigmas_norm_t = self.frac_scheduler.discrete_sigmas_norm[:, None]
        self.sb = self.frac_scheduler.sigma_min

        # Type diffusion
        if self.gen_type:
            self.type_diffusion = build_type_diffusion(
                self.config.get("type_diffusion")
            )
            assert self.type_diffusion is not None, (
                "type_diffusion config must be provided for generation task "
                f"{self.config['task']}"
            )
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

    def forward_step_sample(self, x0, t, batch):
        l0, f0, a0 = x0
        t0_idx = t[0].cast("int64")
        t1_idx = t[1].cast("int64")

        noise = paddle.randn(l0.shape)
        self.lat_randn = noise
        lt = self.lat_scheduler.add_noise(l0, noise, t0_idx)

        noise = paddle.randn(f0.shape)
        self.frac_randn = noise
        ft = (f0 + self.sigmas_t[t1_idx] * noise) % 1.0

        if self.gen_type:
            at = self.type_diffusion.forward_step_sample(a0, t[1], batch)
        else:
            at = a0

        return [lt, ft, at]

    def reverse_step_sample(self, xt, t, model, batch):
        lt, ft, at = xt
        _, f_pred, _ = self.model_prediction(xt, t, model, batch)
        ft_05 = self.frac_reverse_part1(f_pred, ft, t[1])
        xt_05 = [lt, ft_05, at]
        l_pred, f_pred, a_pred = self.model_prediction(xt_05, t, model, batch)
        lt_1 = self.lat_reverse(l_pred, lt, t[0])
        ft_1 = self.frac_reverse_part2(f_pred, ft_05, t[1])

        at_1 = (
            self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch)
            if self.gen_type else at
        )
        return [lt_1, ft_1, at_1]

    def lat_reverse(self, pred, xt, t):
        t_idx = t.cast("int64")
        return self.lat_scheduler.step(pred, t_idx[0], xt).prev_sample

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
        return paddle.randn([batch["batch_size"], 3, 3], dtype="float32")

    def _prior_frac(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum()) if na.ndim > 0 else int(na)
        return paddle.rand([total, 3], dtype="float32")

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

    def train_step(self, batch, model):
        batch["t"] = self._time_sample(batch)
        batch["xt"] = self.forward_step_sample(batch["x0"], batch["t"], batch)
        batch["prediction"] = self.model_prediction(batch["xt"], batch["t"], model, batch)

        loss_lat = ((batch["prediction"][0] - self.lat_randn) ** 2).reshape([-1, 9]).mean(axis=1).mean()

        t_idx = batch["t"][1].cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        normed_score = d_log_p_wrapped_normal(st * self.frac_randn, st) / paddle.sqrt(snt)
        loss_frac = ((batch["prediction"][1] - normed_score) ** 2).reshape([-1, 3]).mean(axis=1)
        # Mask out type-0 mirage atoms from the coordinate loss, rescaling to
        # keep the loss scale over the reduced atom count.
        if self.gen_type and batch["x0"][2] is not None:
            mirage_type = 0
            mask = (batch["x0"][2] != mirage_type).cast(loss_frac.dtype)
            coef = mask.shape[0] / mask.sum().clip(min=1)
            loss_frac = loss_frac * mask * coef
        loss_frac = loss_frac.mean()

        batch["loss"] = loss_lat + loss_frac

        if self.gen_type:
            loss_type = self.type_diffusion.loss(
                batch["prediction"][2],
                batch["xt"][2],
                self.type_diffusion.to_domain(batch["x0"][2]),
                batch["t"][1],
            ).mean()
            batch["loss"] = batch["loss"] + loss_type

        return batch

    def sampling_procedure(self, model, batch):
        batch["xt"] = self.prior_sample(batch)
        for t_vec in self._time_iterator(batch, start_from=self.num_steps - 1):
            batch["t"] = t_vec
            batch["xt"] = self.reverse_step_sample(batch["xt"], batch["t"], model, batch)
        batch["xt"] = self.output_transform(batch["xt"], batch)
        batch["x0_prediction"] = batch["xt"]
        return batch

    def output_transform(self, x0, batch):
        def _out(diff, x):
            return diff.output_transform(x, batch) if hasattr(diff, "output_transform") else x

        return [
            x0[0],
            x0[1],
            _out(self.type_diffusion, x0[2]) if self.gen_type else x0[2],
        ]
