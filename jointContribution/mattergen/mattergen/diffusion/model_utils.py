import math
from typing import Any, TypeVar

import paddle
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTarget

T = TypeVar("T", bound=BatchedData)


def convert_model_out_to_score(
    *,
    model_target: ModelTarget,
    sde: SDE,
    model_out: paddle.Tensor,
    batch_idx: paddle.Tensor,
    t: paddle.Tensor,
    batch: Any
) -> paddle.Tensor:
    """
    Convert a model output to a score, according to the specified model_target.

    model_target: says what the model predicts.
        For example, in RFDiffusion the model predicts clean coordinates;
        in EDM the model predicts the raw noise.
    sde: corruption process
    model_out: model output
    batch_idx: indicates which sample each row of model_out belongs to
    noisy_x: noisy data
    t: diffusion timestep
    batch: noisy batch, ignored except by strange SDEs
    """
    _, std = sde.marginal_prob(
        x=paddle.ones_like(x=model_out), t=t, batch_idx=batch_idx, batch=batch
    )
    if model_target == ModelTarget.score_times_std:
        return model_out / std
    elif model_target == ModelTarget.logits:
        return model_out
    else:
        raise NotImplementedError


class NoiseLevelEncoding(paddle.nn.Layer):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = paddle.exp(
            x=paddle.arange(start=0, end=d_model, step=2)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer(name="div_term", tensor=div_term)

    def forward(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = paddle.zeros(shape=(tuple(t.shape)[0], self.d_model))
        x[:, 0::2] = paddle.sin(x=t[:, None] * self.div_term[None])
        x[:, 1::2] = paddle.cos(x=t[:, None] * self.div_term[None])
        return self.dropout(x)
