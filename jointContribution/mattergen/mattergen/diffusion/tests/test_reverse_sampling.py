import sys


import paddle
from paddle_utils import *

"""
This is an integeratation test of reverse sampling. For a known data distribution that
is Gaussian, we substitute the known ground truth score for an approximate model
prediction and reverse sample to check we retrieve correct moments of the data distribution.
"""
from argparse import Namespace
from contextlib import nullcontext
from functools import partial
from typing import List, Type

import pytest
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import (BatchedData,
                                                   SimpleBatchedData)
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.tests.conftest import (DEFAULT_CORRECTORS,
                                                DEFAULT_PREDICTORS, SDE_TYPES,
                                                WRAPPED_CORRECTORS,
                                                WRAPPED_PREDICTORS)
from mattergen.diffusion.tests.test_sampling import INCOMPATIBLE_SAMPLERS
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedVPSDE


def score_given_xt(
    x: BatchedData,
    t: paddle.Tensor,
    multi_corruption: MultiCorruption,
    x0_mean: paddle.Tensor,
    x0_std: paddle.Tensor,
) -> BatchedData:
    def _score_times_std(x_t: paddle.Tensor, sde: SDE) -> paddle.Tensor:
        a_t, s_t = sde.marginal_prob(x=paddle.ones_like(x=x_t), t=t)
        mean = a_t * x0_mean
        std = paddle.sqrt(x=a_t**2 * x0_std**2 + s_t**2)
        score_times_std = -(x_t - mean) / std**2 * s_t
        return score_times_std

    return x.replace(
        **{
            k: _score_times_std(x_t=x[k], sde=multi_corruption.sdes[k])
            for k in multi_corruption.sdes.keys()
        }
    )


def get_diffusion_module(
    x0_mean, x0_std, multi_corruption: MultiCorruption
) -> DiffusionModule:
    return DiffusionModule(
        model=partial(
            score_given_xt,
            x0_mean=x0_mean,
            x0_std=x0_std,
            multi_corruption=multi_corruption,
        ),
        corruption=multi_corruption,
        loss_fn=Namespace(
            model_targets={
                k: ModelTarget.score_times_std for k in multi_corruption.sdes.keys()
            }
        ),
    )


predictor_corrector_pairs = [(p, None) for p in DEFAULT_PREDICTORS] + [
    (None, c) for c in DEFAULT_CORRECTORS
]


@pytest.mark.parametrize("predictor_type,corrector_type", predictor_corrector_pairs)
@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_reverse_sampling(
    corruption_type: Type, predictor_type: Type, corrector_type: Type
):
    N = 1000 if corrector_type is None else 200
    if predictor_type is None and corrector_type is None:
        return
    fields = ["x", "y", "z", "a"]
    batch_size = 10000
    x0_mean = paddle.to_tensor(data=-3.0)
    x0_std = paddle.to_tensor(data=4.3)
    multi_corruption: MultiCorruption = MultiCorruption(
        sdes={f: corruption_type() for f in fields}
    )
    with (
        pytest.raises(IncompatibleSampler)
        if predictor_type in INCOMPATIBLE_SAMPLERS[corruption_type]
        or corrector_type in INCOMPATIBLE_SAMPLERS[corruption_type]
        else nullcontext()
    ):
        multi_sampler = PredictorCorrector(
            diffusion_module=get_diffusion_module(
                multi_corruption=multi_corruption, x0_mean=x0_mean, x0_std=x0_std
            ),
            device=paddle.CPUPlace(),
            predictor_partials={}
            if predictor_type is None
            else {k: predictor_type for k in fields},
            corrector_partials={}
            if corrector_type is None
            else {k: corrector_type for k in fields},
            n_steps_corrector=5,
            N=N,
            eps_t=0.001,
            max_t=None,
        )
        conditioning_data = _get_conditioning_data(batch_size=batch_size, fields=fields)
        samples, _ = multi_sampler.sample(conditioning_data=conditioning_data)
        means = paddle.to_tensor(
            data=[samples[k].mean() for k in multi_corruption.corruptions.keys()]
        )
        stds = paddle.to_tensor(
            data=[samples[k].std() for k in multi_corruption.corruptions.keys()]
        )
        assert paddle.isclose(x=means.mean(), y=x0_mean, atol=0.1)
        assert paddle.isclose(x=stds.mean(), y=x0_std, atol=0.1)


wrapped_pc_pairs = [(p, None) for p in WRAPPED_PREDICTORS] + [
    (None, c) for c in WRAPPED_CORRECTORS
]


@pytest.mark.parametrize("predictor_type, corrector_type", wrapped_pc_pairs)
@pytest.mark.parametrize("sde_type", [WrappedVESDE, WrappedVPSDE])
def test_wrapped_reverse_sampling(
    sde_type: Type, predictor_type: Type, corrector_type: Type
):
    if predictor_type is None and corrector_type is None:
        return
    N = 50
    fields = ["x", "y", "z", "a"]
    batch_size = 10000
    x0_mean = paddle.to_tensor(data=-2.0)
    x0_std = paddle.to_tensor(data=2.3)
    wrapping_boundary = -2.4
    empirical_samples = paddle.remainder(
        x=paddle.randn(shape=batch_size) * x0_std + x0_mean,
        y=paddle.to_tensor(wrapping_boundary),
    )
    empirical_x0_mean = empirical_samples.mean()
    empirical_x0_std = empirical_samples.std()
    multi_corruption: MultiCorruption = MultiCorruption(
        sdes={k: sde_type(wrapping_boundary=wrapping_boundary) for k in fields}
    )
    predictor_partials = (
        {} if predictor_type is None else {k: predictor_type for k in fields}
    )
    corrector_partials = (
        {} if corrector_type is None else {k: corrector_type for k in fields}
    )
    n_steps_corrector = 5
    multi_sampler: PredictorCorrector = PredictorCorrector(
        diffusion_module=get_diffusion_module(
            x0_mean=x0_mean, x0_std=x0_std, multi_corruption=multi_corruption
        ),
        n_steps_corrector=n_steps_corrector,
        predictor_partials=predictor_partials,
        corrector_partials=corrector_partials,
        device=None,
        N=N,
    )
    conditioning_data = _get_conditioning_data(batch_size=batch_size, fields=fields)
    samples, _ = multi_sampler.sample(conditioning_data=conditioning_data, mask=None)
    assert (
        min(samples[k].min() for k in multi_corruption.corruptions.keys())
        >= wrapping_boundary
    )
    assert max(samples[k].max() for k in multi_corruption.corruptions.keys()) <= 0.0
    means = paddle.to_tensor(
        data=[samples[k].mean() for k in multi_corruption.corruptions.keys()]
    )
    stds = paddle.to_tensor(
        data=[samples[k].std() for k in multi_corruption.corruptions.keys()]
    )
    assert paddle.isclose(x=means.mean(), y=empirical_x0_mean, atol=0.1)
    assert paddle.isclose(x=stds.mean(), y=empirical_x0_std, atol=0.1)


def _get_conditioning_data(batch_size: int, fields: List[str]) -> SimpleBatchedData:
    return SimpleBatchedData(
        data={k: paddle.randn(shape=[batch_size, 1]) for k in fields},
        batch_idx={k: None for k in fields},
    )
