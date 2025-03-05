import paddle

"""
Based on code from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
which is released under Apache licence.

Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Key changes:
-  Rename SDE => Corruption
-  Remove several methods like .reverse(), .discretize()
"""
import abc
import logging
from typing import Optional, Tuple, Union

from mattergen.diffusion.data.batched_data import BatchedData

B = Optional[paddle.Tensor]


def _broadcast_like(x, like):
    """
    add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return x
    return x[(...,) + (None,) * (like.ndim - x.ndim)]


def maybe_expand(
    x: paddle.Tensor, batch: B, like: paddle.Tensor = None
) -> paddle.Tensor:
    """

    Args:
        x: shape (batch_size, ...)
        batch: shape (num_thingies,) with integer entries in the range [0, batch_size), indicating which sample each thingy belongs to
        like: shape x.shape + potential additional dimensions
    Returns:
        expanded x with shape (num_thingies,), or if given like.shape, containing value of x for each thingy.
        If `batch` is None, just returns `x` unmodified, to avoid pointless work if you have exactly one thingy per sample.
    """
    x = _broadcast_like(x, like)
    if batch is None:
        return x
    else:
        if tuple(x.shape)[0] == tuple(batch.shape)[0]:
            logging.warn(
                "Warning: batch shape is == x shape, are you trying to expand something that is already expanded?"
            )
        return x[batch]


class Corruption(abc.ABC):
    """Abstract base class for corruption processes"""

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the corruption process."""
        pass

    @abc.abstractmethod
    def marginal_prob(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(
        self,
        shape: Union[list, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> paddle.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(
        self, z: paddle.Tensor, batch_idx: B = None, batch: Optional[BatchedData] = None
    ) -> paddle.Tensor:
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    @abc.abstractmethod
    def sample_marginal(
        self,
        x: paddle.Tensor,
        t: paddle.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> paddle.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t) (same shape as input x).
        """
        pass
