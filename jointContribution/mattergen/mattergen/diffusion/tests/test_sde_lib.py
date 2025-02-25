from typing import Type

import paddle
import pytest
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.tests.conftest import SDE_TYPES


def _check_batch_shape(x: paddle.Tensor, batch_size: paddle.Tensor):
    """Checks sde outputs that should be (batch_size, )"""
    assert len(tuple(x.shape)) == 1
    assert tuple(x.shape)[0] == batch_size


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("sdetype", SDE_TYPES)
def test_sde(tiny_state_batch, sdetype: Type[SDE], sparse, EPS):
    """Tests correct shapes for all methods of the SDE class"""
    x: paddle.Tensor = tiny_state_batch["foo"]
    sde: SDE = sdetype()
    if sparse:
        batch_size = tiny_state_batch.get_batch_size()
        batch_idx = tiny_state_batch.get_batch_idx("foo")
    else:
        batch_size = tuple(x.shape)[0]
        batch_idx = None
    t = paddle.rand(shape=batch_size) * (sde.T - EPS) + EPS

    def _check_shapes(drift, diffusion):
        assert tuple(drift.shape) == tuple(x.shape)
        assert tuple(diffusion.shape)[0] == tuple(x.shape)[0]

    drift, diffusion = sde.sde(x, t, batch_idx)
    _check_shapes(drift, diffusion)
    mean, std = sde.marginal_prob(x, t, batch_idx)
    _check_shapes(mean, std)
    z = sde.prior_sampling(tuple(x.shape))
    assert tuple(z.shape) == tuple(x.shape)
    prior_logp = sde.prior_logp(z, batch_idx=batch_idx)
    _check_batch_shape(prior_logp, batch_size)


def dummy_score_fn(x, t, batch_idx):
    return paddle.zeros_like(x=x)
