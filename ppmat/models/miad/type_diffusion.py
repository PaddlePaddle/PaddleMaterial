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

import math

import paddle
import paddle.nn.functional as F

from ppmat.schedulers import build_scheduler


class DDPMOnehot:
    """DDPM for atom types with one-hot encoding."""

    def __init__(self, diffusion_config):
        self.num_steps = diffusion_config["num_steps"]
        self.num_types = 100
        self.scheduler = build_scheduler(
            diffusion_config["type_diffusion"].get("scheduler_cfg", {
                "__class_name__": "DDPMScheduler",
                "__init_params__": {
                    "num_train_timesteps": self.num_steps,
                    "beta_schedule": "squaredcos_cap_v2",
                },
            })
        )
        self.to_domain = lambda types: F.one_hot(types - 1, num_classes=self.num_types).cast("float32")
        self.from_domain = lambda onehot: onehot.argmax(axis=-1) + 1

    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        noise = paddle.randn(onehot_x0.shape)
        self.randn_x = noise
        t_idx = t.cast("int64")
        return self.scheduler.add_noise(onehot_x0, noise, t_idx)

    def reverse_step_sample(self, eps_pred, onehot_xt, t, batch):
        t_idx = t.cast("int64")
        out = self.scheduler.step(eps_pred, t_idx[0], onehot_xt)
        onehot_xt_1 = out.prev_sample
        if (t_idx == 0).all():
            return self.from_domain(onehot_xt_1)
        return onehot_xt_1

    def prior_sample(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum()) if na.ndim > 0 else int(na)
        return paddle.randn([total, self.num_types], dtype="float32")

    def loss(self, batch):
        eps_pred = batch["prediction"][2]
        return ((eps_pred - self.randn_x) ** 2).reshape([eps_pred.shape[0], -1]).mean(axis=1)


class D3PMUniformScheduler:
    """D3PM scheduler with uniform transition matrices (model-internal).

    Pre-computes Q_t, cumprod_Q_t, Q_{t-1}, cumprod_Q_{t-1} for all timesteps
    using a cosine schedule. This is the uniform-transition variant used by MiAD,
    as opposed to the absorbing-state D3PMScheduler in ppmat.schedulers.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_types: int = 100,
        s: float = 0.008,
    ):
        self.num_types = num_types

        discretization = paddle.arange(
            1, num_train_timesteps + 1, dtype="float64"
        )
        f_t = paddle.cos(
            (discretization / (num_train_timesteps + 1) + s) / (1 + s) * math.pi / 2
        )
        f_0 = paddle.cos(
            (paddle.to_tensor(0.0, dtype="float64") + s) / (1 + s) * math.pi / 2
        )
        a_t = f_t / f_0
        cumprod_alphas_t = a_t
        cumprod_alphas_t_1 = paddle.concat(
            [paddle.to_tensor([1.0], dtype="float64"), cumprod_alphas_t[:-1]]
        )
        betas_t = 1 - cumprod_alphas_t / cumprod_alphas_t_1

        Q_t_list = []
        for t_idx in range(num_train_timesteps):
            mat = paddle.full(
                (num_types, num_types), betas_t[t_idx] / float(num_types)
            )
            diag_val = 1 - betas_t[t_idx] * (num_types - 1) / num_types
            for i in range(num_types):
                mat[i, i] = diag_val.cast("float32")
            Q_t_list.append(mat)
        Q_t = paddle.stack(Q_t_list, axis=0)

        cumprod_Q_t_list = [Q_t[0]]
        for t_idx in range(1, num_train_timesteps):
            cumprod_Q_t_list.append(
                paddle.matmul(cumprod_Q_t_list[-1], Q_t[t_idx])
            )
        cumprod_Q_t = paddle.stack(cumprod_Q_t_list, axis=0)

        Q_t_1 = paddle.concat(
            [paddle.eye(num_types, dtype="float32").unsqueeze(0), Q_t[:-1]],
            axis=0,
        )

        cumprod_Q_t_1 = paddle.concat(
            [
                paddle.eye(num_types, dtype="float32").unsqueeze(0),
                cumprod_Q_t[:-1],
            ],
            axis=0,
        )

        self.Q_t = Q_t.reshape([-1, num_types, num_types])
        self.Q_t_1 = Q_t_1.reshape([-1, num_types, num_types])
        self.cumprod_Q_t = cumprod_Q_t.reshape([-1, num_types, num_types])
        self.cumprod_Q_t_1 = cumprod_Q_t_1.reshape([-1, num_types, num_types])


class D3PM:
    """Discrete Denoising Diffusion Probabilistic Model for atom types."""

    def __init__(self, diffusion_config):
        self.config = diffusion_config["type_diffusion"]
        self.num_steps = diffusion_config["num_steps"]
        self.scheduler = D3PMUniformScheduler(
            num_train_timesteps=self.num_steps,
            num_types=100,
        )
        self.num_types = self.scheduler.num_types
        self.Q_t = self.scheduler.Q_t
        self.Q_t_1 = self.scheduler.Q_t_1
        self.cumprod_Q_t = self.scheduler.cumprod_Q_t
        self.cumprod_Q_t_1 = self.scheduler.cumprod_Q_t_1

        self.to_domain = lambda types: F.one_hot(types, num_classes=self.num_types).cast("float32")
        self.from_domain = lambda onehot: onehot.argmax(axis=-1)
        self.prediction_to_domain = lambda pred: F.softmax(pred, axis=-1)
        self.default_loss_scale = 1000

    def output_transform(self, x0, batch):
        return self.from_domain(x0)

    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        xt_probs = paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t[t.cast("int64")])[:, 0, :]
        xt = paddle.distribution.Categorical(logits=paddle.log(xt_probs.clip(1e-12))).sample().cast("int64")
        return self.to_domain(xt)

    def _reverse_step_distribution(self, onehot_x0, onehot_xt, t):
        t_idx = t.cast("int64")
        numerator = (
            paddle.matmul(onehot_xt[:, None, :], self.Q_t[t_idx].transpose([0, 2, 1]))[:, 0, :]
            * paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t_1[t_idx])[:, 0, :]
        )
        denominator = (
            paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t[t_idx])[:, 0, :] * onehot_xt
        ).sum(axis=-1)[:, None]
        result = numerator / (denominator + 1e-8)
        result = result / result.sum(axis=-1, keepdim=True)
        return paddle.where(paddle.isnan(result), paddle.full_like(result, 1.0 / self.num_types), result)

    def reverse_step_sample(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        if (t.cast("int64") == 0).all():
            return self.to_domain(self.from_domain(xt_1_probs.cast("float32")))
        xt_1 = paddle.distribution.Categorical(logits=paddle.log(xt_1_probs.clip(1e-12))).sample().cast("int64")
        return self.to_domain(xt_1)

    def prior_sample(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum()) if na.ndim > 0 else int(na)
        shape = [total, self.num_types]
        xT_probs = paddle.ones(shape, dtype="float32") / self.num_types
        xT = paddle.distribution.Categorical(logits=paddle.log(xT_probs.clip(1e-12))).sample().cast("int64")
        return self.to_domain(xT)

    def loss(self, batch):
        onehot_xt = batch["xt"][2]
        t = batch["t"][1].cast("int64")
        onehot_x0_pred = self.prediction_to_domain(batch["prediction"][2])
        pred_xt_1_probs = self._reverse_step_distribution(onehot_x0_pred, onehot_xt, t)
        onehot_x0 = self.to_domain(batch["x0"][2])
        orig_xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        eps = 1e-8
        kl_loss = (
            (orig_xt_1_probs * (paddle.log(orig_xt_1_probs + eps) - paddle.log(pred_xt_1_probs + eps)))
            .reshape([onehot_xt.shape[0], -1]).sum(axis=-1)
        )
        return self.default_loss_scale * kl_loss
