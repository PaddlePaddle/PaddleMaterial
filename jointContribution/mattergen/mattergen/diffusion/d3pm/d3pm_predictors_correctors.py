from typing import Optional, cast

import paddle
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.sampling.predictors import Predictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class D3PMAncestralSamplingPredictor(Predictor):
    """
    Ancestral sampling predictor for D3PM.
    """

    def __init__(
        self,
        *,
        corruption: D3PMCorruption,
        score_fn: ScoreFunction,
        predict_x0: bool = True
    ):
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.predict_x0 = predict_x0

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return isinstance(corruption, D3PMCorruption)

    @property
    def N(self) -> int:
        self.corruption = cast(D3PMCorruption, self.corruption)
        return self.corruption.N

    def update_given_score(
        self,
        *,
        x: paddle.Tensor,
        t: paddle.Tensor,
        dt: paddle.Tensor,
        batch_idx: paddle.Tensor,
        score: paddle.Tensor,
        batch: Optional[BatchedData]
    ) -> SampleAndMean:
        """
        Takes the atom coordinates, cell vectors and atom types at time t and
        returns the atom types at time t-1, sampled using the learned reverse
        atom diffusion model.

        Look at https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion.py

        lines 3201-3229. NOTE: we do implement the taking the softmax of the initial
        sample as per 3226-3227. This could be to avoid weird behaving for picking
        initial states that happened to have very low probability in latent space.
        Try adding if there proves to be a problem generating samples.
        """
        t = to_discrete_time(t=t, N=self.N, T=self.corruption.T)
        class_logits = score
        assert isinstance(self.corruption, D3PMCorruption)
        x_sample = self.corruption._to_non_zero_based(
            paddle.distribution.Categorical(logits=class_logits).sample()
        )
        class_probs = paddle.nn.functional.softmax(x=class_logits, axis=-1)
        class_expected = self.corruption._to_non_zero_based(
            paddle.argmax(x=class_probs, axis=-1)
        )
        if self.predict_x0:
            assert isinstance(self.corruption, D3PMCorruption)
            class_logits, _ = self.corruption.d3pm.sample_and_compute_posterior_q(
                x_0=class_probs,
                t=t[batch_idx].to("int64"),
                make_one_hot=False,
                samples=self.corruption._to_zero_based(x),
                return_logits=True,
            )
            x_sample = self.corruption._to_non_zero_based(
                paddle.distribution.Categorical(logits=class_logits).sample()
            )
            class_expected = self.corruption._to_non_zero_based(
                paddle.argmax(
                    x=paddle.nn.functional.softmax(
                        x=class_logits.to(class_probs.dtype), axis=-1
                    ),
                    axis=-1,
                )
            )
        return x_sample, class_expected
