import abc
from typing import Generic, TypeVar

import paddle
from mattergen.diffusion.data.batched_data import BatchedData

Diffusable = TypeVar("Diffusable", bound=BatchedData)


class ScoreModel(paddle.nn.Layer, Generic[Diffusable], abc.ABC):
    """Abstract base class for score models."""

    @abc.abstractmethod
    def forward(self, x: Diffusable, t: paddle.Tensor) -> Diffusable:
        """Args:
        x: batch of noisy data
        t: timestep. Shape (batch_size, 1)
        """
        ...
