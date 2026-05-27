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
Fractional coordinate diffusion models for MiAD.
"""

import os

import paddle

from ppmat.schedulers.miad_schedulers import scheduler as get_scheduler


class WrappedNormal:
    """Wrapped Normal diffusion for fractional coordinates."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config.frac_diffusion
        self.num_steps = diffusion_config.num_steps

        sigmas_t, sigmas_norm_t, sb, se, d_log_p = get_scheduler(
            self.config.scheduler, self.num_steps
        )

        self.sigmas_t = sigmas_t[:, None]
        self.sigmas_norm_t = sigmas_norm_t[:, None]
        self.sb = sb
        self.d_log_p = d_log_p

        _DEFAULT_GAMMA = {
            "csp_perov5": 5e-7,
            "gen_perov5": 5e-7,
            "csp_mp20": 1e-5,
            "gen_mp20": 1e-5,
            "csp_alex_mp20": 1e-5,
            "gen_alex_mp20": 1e-5,
            "csp_mpts52": 1e-5,
            "gen_mpts52": 1e-5,
            "csp_carbon24": 5e-7,
            "gen_carbon24": 1e-5,
        }
        self.step_lr = getattr(self.config, "step_lr", None)
        if self.step_lr is None:
            self.step_lr = _DEFAULT_GAMMA.get(diffusion_config.task, 1e-5)

        self.drift_step_coef = paddle.ones([self.num_steps], dtype="float32")[:, None]
        self.diff_step_coef = paddle.ones([self.num_steps], dtype="float32")[:, None]
        self.to_domain = lambda x: x

    def output_transform(self, x0, batch):
        return x0

    def forward_step_sample(self, x0, t, batch):
        st = self.sigmas_t[t.cast("int64")]
        self.randn_x = paddle.randn(x0.shape)
        xt = (x0 + st * self.randn_x) % 1.0
        return xt

    def reverse_step_sample_part_1(self, normed_score_pred, xt, t, batch):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        step_size = self.step_lr * (st / self.sb) ** 2
        std_x = paddle.sqrt(2 * step_size)
        drift = (
            -step_size
            * normed_score_pred
            * paddle.sqrt(snt)
            * self.drift_step_coef[t_idx]
        )
        diffusion = std_x * paddle.randn(xt.shape) * self.diff_step_coef[t_idx]
        xt_05 = xt + drift + diffusion
        return xt_05

    def reverse_step_sample_part_2(self, normed_score_pred, xt_05, t, batch):
        t_idx = t.cast("int64")
        st = self.sigmas_t[t_idx]
        st_1 = self.sigmas_t[paddle.maximum(t_idx - 1, paddle.to_tensor([0]))]
        snt = self.sigmas_norm_t[t_idx]
        step_size = st**2 - st_1**2
        std_x = paddle.sqrt((st_1**2 * (st**2 - st_1**2)) / (st**2))
        drift = (
            -step_size
            * normed_score_pred
            * paddle.sqrt(snt)
            * self.drift_step_coef[t_idx]
        )
        diffusion = std_x * paddle.randn(xt_05.shape) * self.diff_step_coef[t_idx]
        xt_1 = (xt_05 + drift + diffusion) % 1.0
        return xt_1

    def reverse_step_sample(self, normed_score_pred, xt, t, batch):
        xt_05 = self.reverse_step_sample_part_1(normed_score_pred, xt, t, batch)
        xt_1 = self.reverse_step_sample_part_2(normed_score_pred, xt_05, t, batch)
        return xt_1

    def prior_sample(self, batch):
        num_atoms = batch["num_atoms"]
        total_atoms = int(num_atoms.sum()) if num_atoms.ndim > 0 else int(num_atoms)
        return paddle.rand([total_atoms, 3], dtype="float32")

    def loss(self, batch):
        t_idx = batch["t"][1].cast("int64")
        st = self.sigmas_t[t_idx]
        snt = self.sigmas_norm_t[t_idx]
        normed_score_pred = batch["prediction"][1]
        normed_score = self.d_log_p(st * self.randn_x, st) / paddle.sqrt(snt)
        l2 = (
            ((normed_score_pred - normed_score) ** 2)
            .reshape([normed_score_pred.shape[0], -1])
            .mean(axis=1)
        )

        # mirage infusion code
        modifications = os.environ.get("MODIFICATIONS_FIELD", "")
        if "miad:add_mirage_atoms_upto" in modifications:
            mirage_type = 0
            atom_types = batch["x0"][2]
            mask = (atom_types != mirage_type).cast(l2.dtype)
            coef = mask.shape[0] / mask.sum()
            l2 = l2 * mask * coef

        return l2

    def get_x0_prediction(self, normed_score_pred, xt, t, batch):
        t_idx = t.cast("int64")
        x0_pred = (xt - self.sigmas_t[t_idx] * normed_score_pred) % 1.0
        return x0_pred


class PFM:
    """Periodic Flow Matching for fractional coordinates."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config.frac_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.final_t = diffusion_config.num_steps
        self.step_coef = paddle.ones([self.final_t], dtype="float32")[:, None]
        self.step = paddle.to_tensor(1.0 / self.final_t)
        self.to_domain = lambda x: x
        self.default_loss_scale = 10

    def output_transform(self, x0, batch):
        return x0

    def forward_step_sample(self, x0, t, batch):
        xT_minus_x0 = paddle.rand(x0.shape) - 0.5
        step = (1 + t[:, None]) * self.step
        xt = (x0 + step * xT_minus_x0) % 1.0
        self.ut = (-1) * xT_minus_x0
        return xt

    def reverse_step_sample(self, vt, xt, t, batch):
        t_idx = t.cast("int64")
        xt_1 = (xt + self.step_coef[t_idx] * self.step * vt) % 1.0
        return xt_1

    def prior_sample(self, batch):
        num_atoms = batch["num_atoms"]
        total_atoms = int(num_atoms.sum()) if num_atoms.ndim > 0 else int(num_atoms)
        return paddle.rand([total_atoms, 3])

    def loss(self, batch):
        vt = batch["prediction"][1]
        l2 = ((vt - self.ut) ** 2).reshape([-1, 3]).mean(axis=1)
        return l2 * self.default_loss_scale

    def get_x0_prediction(self, vt, xt, t, batch):
        step = (1 + t[:, None]) * self.step
        return (xt + step * vt) % 1.0
