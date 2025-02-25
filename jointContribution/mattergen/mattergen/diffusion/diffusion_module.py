from typing import Any
from typing import Callable
from typing import TypeVar

import paddle

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import Loss
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.diffusion.timestep_samplers import TimestepSampler
from mattergen.diffusion.timestep_samplers import UniformTimestepSampler

T = TypeVar("T", bound=BatchedData)
BatchTransform = Callable[[T], T]


def identity(x: T) -> T:
    return x


class DiffusionModule(paddle.nn.Layer):
    """Denoising diffusion model for a multi-part state"""

    def __init__(
        self,
        model: ScoreModel[T],
        corruption: MultiCorruption[T],
        loss_fn: Loss,
        pre_corruption_fn: BatchTransform | None = None,
        timestep_sampler: TimestepSampler | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.corruption = corruption
        self.loss_fn = loss_fn
        self.pre_corruption_fn = pre_corruption_fn or identity
        self.model_targets = {k: ModelTarget(v) for k, v in loss_fn.model_targets.items()}  # noqa

        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-05, max_t=corruption.T
        )
        self._register_corruption_modules()

    def _register_corruption_modules(self):
        """
        Register corruptions that are instances of `paddle.nn.Layer`s for proper device,
        parameter, etc handling.
        """
        assert isinstance(self.corruption, MultiCorruption)
        for idx, (key, _corruption) in enumerate(self.corruption._corruptions.items()):
            if isinstance(_corruption, paddle.nn.Layer):
                self.add_sublayer(name=f"MultiCorruption:{idx}:{key}", sublayer=_corruption)  # noqa

    def calc_loss(
        self,
        batch: T,
        node_is_unmasked=None,  #: (paddle.int64 | None) = None todo: fix this
    ) -> tuple[paddle.Tensor, dict[str, paddle.Tensor]]:
        """
        Calculate loss and metrics given a batch of clean data which may include
        context/conditioning fields. Add noise, predict score using score model, then
        calculate loss.

        Args:
            batch: batch of training data
            node_is_unmasked: mask that has a value 1 for nodes that are included in
            the loss, and a value of 0 for nodes that should be ignored. If None, all
            nodes are included.

        Returns:
            loss: the loss for the batch
            metrics: a dictionary of metrics for the batch
        """
        batch = batch["data"]

        batch = self.pre_corruption_fn(batch)
        noisy_batch, t = self._corrupt_batch(batch)

        score_model_output = self.model(noisy_batch, t)
        loss, metrics = self.loss_fn(
            multi_corruption=self.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )
        assert loss.size == 1
        return loss, metrics

    def _corrupt_batch(self, batch: T) -> tuple[T, paddle.Tensor]:
        """
        Corrupt a batch of data for use in a training step:
        - sample a different timestep for each sample in the batch
        - add noise according to the corruption process

        Args:
            batch: Batch of clean states

        Returns:
            noisy_batch: batch of noisy samples
            t: the timestep used for each sample in the batch

        """
        t = self.sample_timesteps(batch)
        noisy_batch = self.corruption.sample_marginal(batch, t)
        return noisy_batch, t

    def score_fn(self, x: T, t: paddle.Tensor) -> T:
        """Calculate the score of a batch of data at a given timestep

        Args:
            x: batch of data
            t: timestep

        Returns:
            score: score of the batch of data at the given timestep
        """
        model_out: T = self.model(x, t)
        fns = {k: convert_model_out_to_score for k in self.corruption.sdes.keys()}
        scores = apply(
            fns=fns,
            model_out=model_out,
            broadcast=dict(t=t, batch=x),
            sde=self.corruption.sdes,
            model_target=self.model_targets,
            batch_idx=self.corruption._get_batch_indices(x),
        )
        return model_out.replace(**scores)

    def sample_timesteps(self, batch: T) -> paddle.Tensor:
        """Sample the timesteps, which will be used to determine how much noise
        to add to data.

        Args:
           batch: batch of data to be corrupted

        Returns: sampled timesteps
        """
        return self.timestep_sampler(batch_size=batch.get_batch_size())

    def forward(self, batch) -> Any:
        return self.calc_loss(batch)
