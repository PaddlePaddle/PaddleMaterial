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
Atom type diffusion models for MiAD.
"""

import paddle
import paddle.nn.functional as F

from ppmat.models.miad.lattice_diffusion import DDPM
from ppmat.schedulers.miad_schedulers import scheduler as get_scheduler


class DDPM_onehot(DDPM):
    """DDPM for atom types with one-hot encoding."""

    def __init__(self, diffusion_config):
        super().__init__(
            diffusion_config, scheduler_cfg=diffusion_config.type_diffusion
        )

        # Reshape coefficients from (N,1,1) to (N,1) for atom types
        _reshape_attrs = [
            "betas_t",
            "alphas_t",
            "cumprod_alphas_t",
            "cumprod_alphas_t_1",
            "reverse_c0",
            "reverse_c1",
            "reverse_std_coef",
            "eps_to_x0_c0",
            "eps_to_x0_c1",
        ]
        for attr in _reshape_attrs:
            value = getattr(self, attr, None)
            if value is not None and isinstance(value, paddle.Tensor):
                if len(value.shape) == 3:
                    setattr(self, attr, value.reshape([-1, 1]))

        self.num_types = 100
        self.to_domain = lambda types: F.one_hot(
            types - 1, num_classes=self.num_types
        ).cast("float32")
        self.from_domain = lambda onehot: onehot.argmax(axis=-1) + 1

    def output_transform(self, x0, batch):
        return x0

    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        onehot_xt = super().forward_step_sample(onehot_x0, t, batch)
        return onehot_xt

    def reverse_step_sample(self, onehot_eps_pred, onehot_xt, t, batch):
        onehot_xt_1 = super().reverse_step_sample(onehot_eps_pred, onehot_xt, t, batch)
        if (t.cast("int64") == 0).all():
            xt_1 = self.from_domain(onehot_xt_1)
            return xt_1
        return onehot_xt_1

    def prior_sample(self, batch):
        num_atoms = batch["num_atoms"]
        total_atoms = int(num_atoms.sum()) if num_atoms.ndim > 0 else int(num_atoms)
        return paddle.randn([total_atoms, self.num_types], dtype="float32")

    def loss(self, batch):
        eps_pred = batch["prediction"][2]
        l2 = (
            ((eps_pred - self.randn_x) ** 2)
            .reshape([eps_pred.shape[0], -1])
            .mean(axis=1)
        )
        return l2

    def get_x0_prediction(self, onehot_eps_pred, onehot_xt, t, batch, x0_format="disc"):
        onehot_x0_pred = super().get_x0_prediction(onehot_eps_pred, onehot_xt, t, batch)
        if x0_format == "onehot":
            return onehot_x0_pred
        elif x0_format == "disc":
            return self.from_domain(onehot_x0_pred)


class D3PM:
    """Discrete Denoising Diffusion Probabilistic Model for atom types."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config.type_diffusion
        self.num_steps = diffusion_config.num_steps
        self.num_types = 100

        Q_t, cumprod_Q_t = get_scheduler(self.config.scheduler, self.num_steps)

        Q_t_1 = paddle.concat(
            [
                paddle.eye(Q_t.shape[1], dtype="float32").unsqueeze(0),
                Q_t[:-1],
            ],
            axis=0,
        )
        self.Q_t = Q_t.reshape([-1, self.num_types, self.num_types])
        self.Q_t_1 = Q_t_1.reshape([-1, self.num_types, self.num_types])
        cumprod_Q_t_1 = paddle.concat(
            [
                paddle.eye(cumprod_Q_t.shape[1], dtype="float32").unsqueeze(0),
                cumprod_Q_t[:-1],
            ],
            axis=0,
        )
        self.cumprod_Q_t = cumprod_Q_t.reshape([-1, self.num_types, self.num_types])
        self.cumprod_Q_t_1 = cumprod_Q_t_1.reshape([-1, self.num_types, self.num_types])

        self.to_domain = lambda types: F.one_hot(
            types, num_classes=self.num_types
        ).cast("float32")
        self.from_domain = lambda onehot: onehot.argmax(axis=-1)
        self.prediction_to_domain = lambda pred: F.softmax(pred, axis=-1)
        self.default_loss_scale = 1000

    def output_transform(self, x0, batch):
        return self.from_domain(x0)

    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        xt_probs = paddle.matmul(
            onehot_x0[:, None, :], self.cumprod_Q_t[t.cast("int64")]
        )[:, 0, :]
        xt = (
            paddle.distribution.Categorical(logits=paddle.log(xt_probs.clip(1e-12)))
            .sample()
            .cast("int64")
        )
        onehot_xt = self.to_domain(xt)
        return onehot_xt

    def _reverse_step_distribution(self, onehot_x0, onehot_xt, t):
        t_idx = t.cast("int64")
        numerator = (
            paddle.matmul(onehot_xt[:, None, :], self.Q_t[t_idx].transpose([0, 2, 1]))[
                :, 0, :
            ]
            * paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t_1[t_idx])[:, 0, :]
        )
        denominator = (
            paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t[t_idx])[:, 0, :]
            * onehot_xt
        ).sum(axis=-1)[:, None]
        result = numerator / (denominator + 1e-8)
        result = result / result.sum(axis=-1, keepdim=True)
        result = paddle.where(
            paddle.isnan(result),
            paddle.full_like(result, 1.0 / self.num_types),
            result,
        )
        return result

    def reverse_step_sample(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        xt_1_probs = paddle.where(
            paddle.isnan(xt_1_probs),
            paddle.full_like(xt_1_probs, 1.0 / self.num_types),
            xt_1_probs,
        )
        if (t.cast("int64") == 0).all():
            return self.to_domain(self.from_domain(xt_1_probs.cast("float32")))
        xt_1 = (
            paddle.distribution.Categorical(logits=paddle.log(xt_1_probs.clip(1e-12)))
            .sample()
            .cast("int64")
        )
        onehot_xt_1 = self.to_domain(xt_1)
        return onehot_xt_1

    def prior_sample(self, batch):
        num_atoms = batch["num_atoms"]
        total_atoms = int(num_atoms.sum()) if num_atoms.ndim > 0 else int(num_atoms)
        shape = [total_atoms, self.num_types]
        xT_probs = paddle.ones(shape, dtype="float32") / self.num_types
        xT = (
            paddle.distribution.Categorical(logits=paddle.log(xT_probs.clip(1e-12)))
            .sample()
            .cast("int64")
        )
        onehot_xT = self.to_domain(xT)
        return onehot_xT

    def loss(self, batch):
        onehot_xt = batch["xt"][2]
        t = batch["t"][1].cast("int64")
        onehot_x0_pred = self.prediction_to_domain(batch["prediction"][2])
        pred_xt_1_probs = self._reverse_step_distribution(onehot_x0_pred, onehot_xt, t)
        onehot_x0 = self.to_domain(batch["x0"][2])
        orig_xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        eps = 1e-4
        kl_loss = (
            (
                orig_xt_1_probs
                * (
                    paddle.log(orig_xt_1_probs + eps)
                    - paddle.log(pred_xt_1_probs + eps)
                )
            )
            .reshape([onehot_xt.shape[0], -1])
            .sum(axis=-1)
        )
        return self.default_loss_scale * kl_loss

    def get_x0_prediction(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        return self.from_domain(onehot_x0)
