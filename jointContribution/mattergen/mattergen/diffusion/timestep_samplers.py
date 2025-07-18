from typing import Protocol

import paddle
from mattergen.diffusion.corruption.sde_lib import SDE


class TimestepSampler(Protocol):
    min_t: float
    max_t: float

    def __call__(
        self, batch_size: int, device: (paddle.CPUPlace, paddle.CUDAPlace, str)
    ) -> paddle.float32:
        raise NotImplementedError


class UniformTimestepSampler:
    """Samples diffusion timesteps uniformly over the training time."""

    def __init__(self, *, min_t: float, max_t: float):
        """Initializes the sampler.

        Args:
            min_t (float): Smallest timestep that will be seen during training.
            max_t (float): Largest timestep that will be seen during training.
        """
        super().__init__()
        self.min_t = min_t
        self.max_t = max_t

    def __call__(
        self, batch_size: int, 
    ) -> paddle.float32:
        return paddle.rand(shape=[batch_size]) * (self.max_t - self.min_t) + self.min_t
