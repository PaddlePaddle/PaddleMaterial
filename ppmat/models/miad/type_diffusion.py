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

import copy

import paddle
import paddle.nn.functional as F

from ppmat.schedulers import build_scheduler


def build_type_diffusion(cfg):
    """Build a type diffusion module from ``__class_name__`` / ``__init_params__``.

    Evaluates in this module's namespace: ``D3PM`` lives here, not in
    ``ppmat.schedulers``, so the generic ``build_scheduler`` cannot resolve it.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__", {})
    return eval(class_name)(**init_params)


class D3PM:
    """Discrete Denoising Diffusion Probabilistic Model for atom types."""

    def __init__(self, scheduler_cfg, loss_scale=1000, kl_eps=1e-4):
        self.scheduler = build_scheduler(scheduler_cfg)
        self.num_types = self.scheduler.num_types
        self.Q_t = self.scheduler.Q_t
        self.Q_t_1 = self.scheduler.Q_t_1
        self.cumprod_Q_t = self.scheduler.cumprod_Q_t
        self.cumprod_Q_t_1 = self.scheduler.cumprod_Q_t_1

        self.to_domain = lambda types: F.one_hot(
            types, num_classes=self.num_types
        ).cast("float32")
        self.from_domain = lambda onehot: onehot.argmax(axis=-1)
        self.prediction_to_domain = lambda pred: F.softmax(pred, axis=-1)
        self.loss_scale = loss_scale
        self.kl_eps = kl_eps

    def output_transform(self, x0, batch):
        return self.from_domain(x0)

    def forward_step_sample(self, x0, t, batch):
        onehot_x0 = self.to_domain(x0)
        xt_probs = paddle.matmul(
            onehot_x0[:, None, :], self.cumprod_Q_t[t.cast("int64")]
        )[:, 0, :]
        xt = paddle.distribution.Categorical(
            logits=paddle.log(xt_probs.clip(1e-12))
        ).sample().cast("int64")
        return self.to_domain(xt)

    def _reverse_step_distribution(self, onehot_x0, onehot_xt, t):
        t_idx = t.cast("int64")
        numerator = (
            paddle.matmul(
                onehot_xt[:, None, :], self.Q_t[t_idx].transpose([0, 2, 1])
            )[:, 0, :]
            * paddle.matmul(
                onehot_x0[:, None, :], self.cumprod_Q_t_1[t_idx]
            )[:, 0, :]
        )
        denominator = (
            paddle.matmul(onehot_x0[:, None, :], self.cumprod_Q_t[t_idx])[:, 0, :]
            * onehot_xt
        ).sum(axis=-1)[:, None]
        result = numerator / (denominator + 1e-8)
        result = result / result.sum(axis=-1, keepdim=True)
        # Fallback for zero-denominator edge cases (uniform distribution)
        return paddle.where(
            paddle.isnan(result),
            paddle.full_like(result, 1.0 / self.num_types),
            result,
        )

    def reverse_step_sample(self, onehot_pred, onehot_xt, t, batch):
        onehot_x0 = self.prediction_to_domain(onehot_pred)
        xt_1_probs = self._reverse_step_distribution(onehot_x0, onehot_xt, t)
        if (t.cast("int64") == 0).all():
            return self.to_domain(self.from_domain(xt_1_probs.cast("float32")))
        xt_1 = paddle.distribution.Categorical(
            logits=paddle.log(xt_1_probs.clip(1e-12))
        ).sample().cast("int64")
        return self.to_domain(xt_1)

    def prior_sample(self, batch):
        na = batch["num_atoms"]
        total = int(na.sum()) if na.ndim > 0 else int(na)
        shape = [total, self.num_types]
        xT_probs = paddle.ones(shape, dtype="float32") / self.num_types
        xT = paddle.distribution.Categorical(
            logits=paddle.log(xT_probs.clip(1e-12))
        ).sample().cast("int64")
        return self.to_domain(xT)

    def loss(self, onehot_pred, onehot_xt, onehot_x0, t):
        """Per-atom D3PM KL loss between predicted and ground-truth reverse
        distributions, scaled by ``loss_scale``."""
        t_idx = t.cast("int64")
        onehot_x0_pred = self.prediction_to_domain(onehot_pred)
        pred_xt_1_probs = self._reverse_step_distribution(
            onehot_x0_pred, onehot_xt, t_idx
        )
        orig_xt_1_probs = self._reverse_step_distribution(
            onehot_x0, onehot_xt, t_idx
        )
        kl_loss = (
            (
                orig_xt_1_probs
                * (
                    paddle.log(orig_xt_1_probs + self.kl_eps)
                    - paddle.log(pred_xt_1_probs + self.kl_eps)
                )
            )
            .reshape([onehot_xt.shape[0], -1])
            .sum(axis=-1)
        )
        return self.loss_scale * kl_loss
