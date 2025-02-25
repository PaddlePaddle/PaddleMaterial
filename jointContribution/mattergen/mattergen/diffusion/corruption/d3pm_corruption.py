from typing import Optional, Tuple, Union

import paddle
from mattergen.diffusion.corruption.corruption import (B, Corruption,
                                                       maybe_expand)
from mattergen.diffusion.d3pm import d3pm
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time
from paddle_scatter import scatter_add


class D3PMCorruption(Corruption):
    """D3PM discrete corruption process. Has discret time and discrete (categorical) values."""

    def __init__(self, d3pm: d3pm.DiscreteDiffusionBase, offset: int = 0):
        super().__init__()
        self.d3pm = d3pm
        self.offset = offset

    @property
    def N(self) -> int:
        """Number of diffusion timesteps i.e. number of noise levels.
        Must match number of noise levels used for sampling. To change this, we'd need to implement continuous-time diffusion for discrete things
        as in e.g. Campbell et al. https://arxiv.org/abs/2205.14987"""
        return self.d3pm.num_steps

    def _to_zero_based(self, x: paddle.Tensor) -> paddle.Tensor:
        """Convert from non-zero-based indices to zero-based indices."""
        return x - self.offset

    def _to_non_zero_based(self, x: paddle.Tensor) -> paddle.Tensor:
        """Convert from zero-based indices to non-zero-based indices."""
        return x + self.offset

    @property
    def T(self) -> float:
        """End time of the Corruption process."""
        return 1

    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Parameters to determine the marginal distribution of the corruption process, $p_t(x | x_0)$."""
        t_discrete = (
            maybe_expand(to_discrete_time(t, N=self.N, T=self.T), batch_idx) + 1
        )
        _, logits = d3pm.q_sample(
            self._to_zero_based(x.astype(dtype="int64")),
            t_discrete,
            diffusion=self.d3pm,
            return_logits=True,
        )
        return logits, None

    def prior_sampling(
        self,
        shape: Union[list, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> paddle.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""
        return self._to_non_zero_based(self.d3pm.sample_stationary(shape))

    def prior_logp(
        self, z: paddle.Tensor, batch_idx: B = None, batch: Optional[BatchedData] = None
    ) -> paddle.Tensor:
        """Compute log-density of the prior distribution.

        Args:
          z: samples, non-zero-based indices, i.e., we first need to subtract the offset
        Returns:
          log probability density
        """
        probs = self.d3pm.stationary_probs(tuple(z.shape)).to(z.place)
        log_probs = (probs + 1e-08).log()
        log_prob_per_sample = log_probs[:, self._to_zero_based(z.astype(dtype="int64"))]
        log_prob_per_structure = scatter_add(log_prob_per_sample, batch_idx, dim=0)
        return log_prob_per_structure

    def sample_marginal(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> paddle.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t), non-zero-based indices
          where raw_noise is drawn from standard Gaussian
        """
        logits = self.marginal_prob(x=x, t=t, batch_idx=batch_idx, batch=batch)[0]
        sample = paddle.distribution.Categorical(logits=logits).sample()
        return self._to_non_zero_based(sample)
