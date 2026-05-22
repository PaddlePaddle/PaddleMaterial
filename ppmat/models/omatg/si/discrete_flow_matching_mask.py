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

"""Discrete flow matching with mask base distribution for atom species (DNG)."""

from typing import Callable, Dict, Iterable, Tuple

import paddle

from .abstracts import StochasticInterpolantSpecies

MAX_ATOM_NUM: int = 100


class DiscreteFlowMatchingMask(StochasticInterpolantSpecies):
    """Discrete flow matching between masked base p_0 and target p_1 for species.

    The base points x_0 are entirely in the masked state (token 0).
    The model prediction returns (sum(n_atoms), MAX_ATOM_NUM) logits.
    Loss is cross_entropy(pred, x_1 - 1).

    :param noise: noise parameter scaling added during integration.
    """

    def __init__(self, noise: float = 0.0) -> None:
        super().__init__()
        if noise < 0.0:
            raise ValueError("Noise parameter must be greater than or equal to 0.")
        self._mask_index = 0
        self._noise = noise

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        assert paddle.all(x_1 != self._mask_index)
        x_t = x_0.clone()
        mask = paddle.rand(x_0.shape) < t
        x_t[mask] = x_1[mask]
        return x_t, paddle.zeros_like(x_t)

    def loss_keys(self) -> Iterable[str]:
        yield "loss"

    def loss(
        self,
        model_function: Callable[[paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]],
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        assert paddle.all(x_1 != self._mask_index)
        pred = model_function(x_t)[0]
        assert pred.shape == (x_0.shape[0], MAX_ATOM_NUM)
        return {"loss": paddle.nn.functional.cross_entropy(input=pred, label=x_1 - 1)}

    def integrate(
        self,
        model_function: Callable[
            [paddle.Tensor, paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]
        ],
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        eps = paddle.finfo(paddle.float64).eps
        x_1_probs = paddle.nn.functional.softmax(
            model_function(time, x_t)[0], axis=-1
        )
        x_1_probs = x_1_probs.reshape((-1, MAX_ATOM_NUM))
        shifted_x_1 = paddle.multinomial(
            x_1_probs, num_samples=1, replacement=True
        ).squeeze(-1)
        shifted_x_t = x_t - 1
        assert shifted_x_1.shape == x_t.shape == shifted_x_t.shape
        shifted_x_1_hot = paddle.nn.functional.one_hot(
            shifted_x_1, num_classes=MAX_ATOM_NUM
        )
        dpt = shifted_x_1_hot - 1.0 / MAX_ATOM_NUM
        dpt_xt = dpt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        pt = time * shifted_x_1_hot + (1.0 - time) * (1.0 / MAX_ATOM_NUM)
        pt_xt = pt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        S = paddle.count_nonzero(x=pt, axis=-1)
        rate = paddle.nn.functional.relu(x=dpt - dpt_xt[:, None]) / (
            S * pt_xt
        )[:, None]
        rate[(pt_xt == 0.0)[:, None].expand([-1, MAX_ATOM_NUM])] = 0.0
        rate[pt == 0.0] = 0.0
        rate_db = paddle.zeros_like(rate)
        if self._noise > 0.0:
            rate_db[shifted_x_t == shifted_x_1] = 1.0
            rate_db[shifted_x_1 != shifted_x_t] = (MAX_ATOM_NUM * time + 1.0 - time) / (
                1.0 - time + eps
            )
            rate_db *= self._noise
        rate = rate + rate_db
        step_probs = (rate * time_step).clip(max=1.0)
        step_probs.scatter_(-1, shifted_x_t[:, None], 0.0)
        step_probs.scatter_(
            -1,
            shifted_x_t[:, None],
            (1.0 - step_probs.sum(axis=-1, keepdim=True)).clip(min=0.0),
        )
        x_t = paddle.multinomial(
            step_probs, num_samples=1, replacement=True
        ).squeeze(-1) + 1
        return x_t

    def uses_masked_species(self) -> bool:
        return True
