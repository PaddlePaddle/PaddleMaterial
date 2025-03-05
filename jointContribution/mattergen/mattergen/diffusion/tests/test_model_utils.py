from functools import partial

import paddle
import pytest
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.tests.conftest import SDE_TYPES


@pytest.mark.parametrize("sde_type", SDE_TYPES)
def test_conversions_match(sde_type):
    """Check that we get the same score whether the model output is interpreted as prediction of clean data, noise, or minus noise."""
    sde = sde_type()
    t = paddle.linspace(start=0.1, stop=0.9, num=10)
    clean = paddle.randn(shape=[10, 3])
    z = paddle.randn(shape=clean.shape, dtype=clean.dtype)
    mean, std = sde.marginal_prob(
        x=clean, t=t, batch_idx=paddle.arange(end=10), batch=None
    )
    noisy = mean + std * z
    _convert = partial(
        convert_model_out_to_score,
        sde=sde,
        batch_idx=paddle.arange(end=10),
        noisy_x=noisy,
        t=t,
        batch=None,
    )
    score1 = _convert(model_target=ModelTarget.score_times_std, model_out=-z)
    score2 = _convert(model_target=ModelTarget.noise, model_out=z)
    score3 = _convert(model_target=ModelTarget.clean_data, model_out=clean)
    assert paddle.allclose(x=score1, y=score2).item()
    assert paddle.allclose(x=score1, y=score3, atol=0.0001).item()
