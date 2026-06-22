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
Lattice diffusion models for MiAD.
"""

import paddle

from ppmat.schedulers.miad_schedulers import scheduler as get_scheduler


class DDPM:
    """DDPM for lattice diffusion.

    Note: This is a lightweight DDPM that pre-computes and stores coefficient
    tensors for direct indexing during forward/reverse steps. It differs from
    ppmat.schedulers.scheduling_ddpm.DDPMScheduler which is a full-featured
    class-based scheduler. The direct-index design is required by MiAD's
    CrystalGen which indexes coeff tensors by timestep in tight loops, where
    the class-based interface overhead would be prohibitive.
    """

    def __init__(self, diffusion_config, scheduler_cfg=None):
        self.config = (
            scheduler_cfg
            if scheduler_cfg is not None
            else diffusion_config.lat_diffusion
        )
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time

        cumprod_alphas_t, ca_t = get_scheduler(self.config.scheduler, self.num_steps)

        cumprod_alphas_t_1 = cumprod_alphas_t[:-1]
        cumprod_alphas_t = cumprod_alphas_t[1:]

        if ca_t is not None:
            self.f_cumprod_alphas_t = lambda t: ca_t(1 + t).reshape([-1, 1, 1])
        else:
            self.f_cumprod_alphas_t = None

        alphas_t = cumprod_alphas_t / cumprod_alphas_t_1
        betas_t = 1 - alphas_t

        # Reshape to (num_steps, 1, 1) for broadcasting with (batch, 3, 3)
        self.betas_t = betas_t.reshape([-1, 1, 1])
        self.alphas_t = alphas_t.reshape([-1, 1, 1])
        self.cumprod_alphas_t = cumprod_alphas_t.reshape([-1, 1, 1])
        self.cumprod_alphas_t_1 = cumprod_alphas_t_1.reshape([-1, 1, 1])

        # Pre-compute reverse sampling coefficients
        self.reverse_c0 = 1 / paddle.sqrt(self.alphas_t)
        self.reverse_c1 = (1 - self.alphas_t) / paddle.sqrt(1 - self.cumprod_alphas_t)
        self.reverse_std_coef = paddle.sqrt(
            self.betas_t * (1 - self.cumprod_alphas_t_1) / (1 - self.cumprod_alphas_t)
        )

        self.to_domain = lambda x: x

    def forward_step_sample(self, x0, t, batch):
        if self.cont_time and self.f_cumprod_alphas_t is not None:
            at = self.f_cumprod_alphas_t(t)
        else:
            at = self.cumprod_alphas_t[t.cast("int64")]
        self.randn_x = paddle.randn(x0.shape)
        xt = paddle.sqrt(at) * x0 + paddle.sqrt(1 - at) * self.randn_x
        return xt

    def reverse_step_sample(self, eps_pred, xt, t, batch):
        t_idx = t.cast("int64")
        mu_xt_1 = self.reverse_c0[t_idx] * (xt - self.reverse_c1[t_idx] * eps_pred)
        if (t_idx == 0).all():
            return mu_xt_1
        return mu_xt_1 + self.reverse_std_coef[t_idx] * paddle.randn(mu_xt_1.shape)

    def prior_sample(self, batch):
        return paddle.randn([batch["batch_size"], 3, 3], dtype="float32")

    def loss(self, batch):
        eps_pred = batch["prediction"][0]
        l2 = (
            ((eps_pred - self.randn_x) ** 2)
            .reshape([eps_pred.shape[0], -1])
            .mean(axis=1)
        )
        return l2


class FM:
    """Flow Matching for lattice diffusion."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config.lat_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.step = 1.0 / self.num_steps
        self.parameterization = getattr(
            diffusion_config.lat_diffusion, "parameterization", "eps"
        )
        self.to_domain = lambda x: x

    def forward_step_sample(self, x0, t, batch):
        eps = paddle.randn(x0.shape)
        step = (1 + t[:, None, None]) * self.step
        xt = (1 - step) * x0 + step * eps
        if self.parameterization == "eps":
            self.ut = (-1) * eps
        elif self.parameterization == "v":
            self.ut = x0 - eps
        return xt

    def reverse_step_sample(self, pred, xt, t, batch):
        t_idx = t.cast("int64")
        if self.parameterization == "eps":
            if t_idx[0] >= 999:
                return paddle.randn(xt.shape)
            eps_pred = (-1) * pred
            step = (1 + t[:, None, None]) * self.step
            x0_pred = (xt - step * eps_pred) / (1 - step)
            vt = x0_pred - eps_pred
        elif self.parameterization == "v":
            vt = pred
        xt_1 = xt + self.step * vt
        return xt_1

    def prior_sample(self, batch):
        return paddle.randn([batch["batch_size"], 3, 3], dtype="float32")

    def loss(self, batch):
        vt = batch["prediction"][0]
        l2 = ((vt - self.ut) ** 2).reshape([-1, 9]).mean(axis=1)
        return l2


class FM_LenAng:
    """Flow Matching for lattice in length-angle parameterization."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config.lat_diffusion
        self.num_steps = diffusion_config.num_steps
        self.cont_time = diffusion_config.cont_time
        self.step = 1.0 / self.num_steps

        alpha, theta = 1.3, 0.25
        self.gamma_alpha = alpha
        self.gamma_theta = theta
        self.angle_difference_bound = 20
        self.default_loss_scale = 0.45

        self._lenang2lat = None
        self.to_domain = lambda x: x

    def _get_converter(self):
        if self._lenang2lat is None:
            from ppmat.utils.crystal import lattice_params_to_matrix_paddle

            self._lenang2lat = lambda la: lattice_params_to_matrix_paddle(
                la[:, :3], la[:, 3:]
            )
        return self._lenang2lat

    def forward_step_sample(self, x0, t, batch):
        xT = self.prior_sample(batch)
        step = (1 + t[:, None, None]) * self.step
        xt = (1 - step) * x0 + step * xT
        self.ut = x0 - xT
        return xt

    def reverse_step_sample(self, vt, xt, t, batch):
        xt_1 = xt + self.step * vt
        return xt_1

    def prior_sample(self, batch):
        bs = batch["batch_size"]
        lenang_xT = paddle.zeros([bs, 6], dtype="float32")

        gamma_dist = paddle.distribution.Gamma(
            paddle.to_tensor([self.gamma_alpha], dtype="float32"),
            paddle.to_tensor([self.gamma_theta], dtype="float32"),
        )
        lenang_xT[:, :3] = 2 + gamma_dist.sample([bs, 3]).reshape([bs, 3])

        # Sample angles from constrained uniform
        ang = 60 + 60 * paddle.rand([4 * bs, 3])
        check = (
            (ang[:, 0] + ang[:, 1] - ang[:, 2] > self.angle_difference_bound)
            * (ang[:, 2] + ang[:, 0] - ang[:, 1] > self.angle_difference_bound)
            * (ang[:, 1] + ang[:, 2] - ang[:, 0] > self.angle_difference_bound)
        )
        valid = ang[check][:bs]
        lenang_xT[:, 3:] = valid

        lenang2lat = self._get_converter()
        xT = lenang2lat(lenang_xT)
        return xT

    def loss(self, batch):
        vt = batch["prediction"][0]
        l2 = ((vt - self.ut) ** 2).reshape([-1, 9]).mean(axis=1)
        return self.default_loss_scale * l2
