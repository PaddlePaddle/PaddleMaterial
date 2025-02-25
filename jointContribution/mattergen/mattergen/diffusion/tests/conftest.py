import random
from functools import wraps
from typing import Callable, Dict, List

import numpy
import paddle
import pytest
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.sde_lib import SDE, VESDE, VPSDE
from mattergen.diffusion.data.batched_data import (BatchedData,
                                                   SimpleBatchedData,
                                                   collate_fn)
from mattergen.diffusion.sampling import predictors
from mattergen.diffusion.sampling import predictors_correctors as pc
from mattergen.diffusion.wrapped.wrapped_predictors_correctors import (
    WrappedAncestralSamplingPredictor, WrappedLangevinCorrector)
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedVPSDE

SDE_TYPES = [VPSDE, VESDE, WrappedVPSDE, WrappedVESDE]
DISCRETE_CORRUPTION_TYPES = [D3PMCorruption]
CORRUPTION_TYPES = SDE_TYPES + DISCRETE_CORRUPTION_TYPES
DEFAULT_PREDICTORS = [predictors.AncestralSamplingPredictor]
WRAPPED_PREDICTORS = [WrappedAncestralSamplingPredictor]
WRAPPED_CORRECTORS = [WrappedLangevinCorrector]
DEFAULT_CORRECTORS = [pc.LangevinCorrector]
DummyState = Dict[str, paddle.Tensor]


def seed_all(seed):
    """Set the seed of all computational frameworks."""
    random.seed(seed)
    numpy.random.seed(seed)
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)


@pytest.fixture(autouse=True)
def seed_random_state(seed: int = 42):
    """
    Fixture for seeding random states of every unit test. Is invoked automatically before each test.

    Args:
        seed (int, optional): Random seed. Defaults to 42.
    """
    seed_all(seed)
    yield


@pytest.fixture
def EPS():
    return 1e-05


def dummy_score_fn(
    batch: SimpleBatchedData, t: paddle.Tensor, train: bool
) -> SimpleBatchedData:
    return batch.replace(**{k: paddle.ones_like(x=batch[k]) for k in batch.data})


@pytest.fixture
def diffusion_mocks():
    class Mocks:
        DummyState = DummyState
        dummy_score_fn = dummy_score_fn

    return Mocks


@pytest.fixture(scope="function")
def make_state_batch():
    def make_batch(sde_type):
        return collate_fn([_make_sample(i) for i in range(0, 10)])

    return make_batch


@pytest.fixture(scope="function")
def tiny_state_batch() -> BatchedData:
    return collate_fn([_make_sample(i) for i in range(0, 10)])


def _make_sample(bigness) -> DummyState:
    foo_per_sample = 3 * (bigness + 1)
    bar_per_sample = 1 * (bigness + 1)
    return dict(
        foo=paddle.randn(shape=[foo_per_sample, 3]),
        bar=paddle.randn(shape=[bar_per_sample, 4]),
    )


@pytest.fixture
def get_multi_corruption():
    from mattergen.diffusion.corruption.multi_corruption import MultiCorruption

    def factory(corruption_type, keys: List[str]):
        discrete_corruptions = {
            k: corruption_type()
            for k in keys
            if issubclass(corruption_type, Corruption)
            and not issubclass(corruption_type, SDE)
        }
        sdes = {k: corruption_type() for k in keys if issubclass(corruption_type, SDE)}
        return MultiCorruption(sdes=sdes, discrete_corruptions=discrete_corruptions)

    return factory


@pytest.fixture
def dummy_state() -> DummyState:
    return _make_sample(3)
