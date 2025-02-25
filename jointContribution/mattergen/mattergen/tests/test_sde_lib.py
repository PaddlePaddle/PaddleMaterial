from abc import ABC, abstractmethod
from math import pi
from typing import Optional, Tuple

import paddle
import pytest
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE, expand, make_noise_symmetric_preserve_variance)
from mattergen.diffusion.corruption.sde_lib import VPSDE


class TestVPSDE(VPSDE, ABC):
    @classmethod
    @abstractmethod
    def get_random_data(cls, N: int) -> paddle.Tensor:
        pass

    @abstractmethod
    def get_limit_mean(
        self, x: paddle.Tensor, limit_info: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        pass

    @abstractmethod
    def get_limit_var(
        self, x: paddle.Tensor, limit_info: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        pass

    @abstractmethod
    def assert_discretize_ok(self, x: paddle.Tensor) -> None:
        pass


def test_LatticeVPSDE_get_limit_mean():
    density = 15.0
    sde = LatticeVPSDE(limit_density=density, limit_mean="scaled")
    n_atoms = paddle.to_tensor(data=[1, 2])
    batch = ChemGraph(num_atoms=n_atoms)
    lattices = paddle.eye(num_rows=3).expand(shape=[2, 3, 3])
    lattice_mean = sde.get_limit_mean(x=lattices, batch=batch)
    expected_val = paddle.pow(x=n_atoms / density, y=1 / 3)
    assert paddle.allclose(
        x=lattice_mean[0], y=expected_val[0] * paddle.eye(num_rows=3)
    ).item()
    assert paddle.allclose(
        x=lattice_mean[1], y=expected_val[1] * paddle.eye(num_rows=3)
    ).item()


def test_LatticeVPSDE_get_var_mean():
    density = 20.0
    sde = LatticeVPSDE(limit_density=density)
    n_atoms = paddle.to_tensor(data=[1, 2])
    batch = ChemGraph(num_atoms=n_atoms)
    lattices = paddle.eye(num_rows=3).expand(shape=[2, 3, 3])
    lattice_var = sde.get_limit_var(x=lattices, batch=batch)
    expected_val = (
        expand(paddle.pow(x=n_atoms, y=2 / 3), (2, 3, 3)).tile(1, 3, 3)
        * sde.limit_var_scaling_constant
    )
    assert paddle.allclose(x=lattice_var, y=expected_val).item()


def test_LatticeVPSDE_prior_sampling():
    density = 20.0
    Nbatch = 1000
    n_atoms = paddle.ones(shape=(Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)
    sde = LatticeVPSDE(limit_density=density)
    x = sde.prior_sampling(shape=(Nbatch, 3, 3), conditioning_data=batch)
    expected_mean = sde.get_limit_mean(x=x, batch=batch).mean(axis=0)
    expected_var = sde.get_limit_var(x=x, batch=batch).mean(axis=0)[0, 0]
    assert tuple(x.shape) == (Nbatch, 3, 3)
    assert paddle.allclose(x=x.mean(axis=0), y=expected_mean, atol=0.1).item()
    assert paddle.allclose(x=x.var(axis=0).mean(), y=expected_var, atol=0.1).item()


def test_LatticeVPSDE_prior_logp():
    density = 20.0
    Nbatch = 100
    n_atoms = paddle.ones(shape=(Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)
    sde = LatticeVPSDE(limit_density=density, limit_var_scaling_constant=1.0)
    x = sde.prior_sampling(shape=(Nbatch, 3, 3), conditioning_data=batch)
    expected_log_likelihood = -0.5 * paddle.pow(x=x, y=2) - 0.5 * paddle.log(
        x=paddle.to_tensor(data=[2.0 * pi])
    )
    expected_log_likelihood = paddle.sum(x=expected_log_likelihood, axis=(-2, -1))
    assert paddle.allclose(
        x=sde.prior_logp(z=x, batch=batch), y=expected_log_likelihood
    ).item()


def test_LatticeVPSDE_marginal_prob():
    density = 20.0
    Nbatch = 100
    n_atoms = paddle.ones(shape=(Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)
    sde = LatticeVPSDE(limit_density=density, limit_var_scaling_constant=1.0)
    t = paddle.ones(shape=(1,)) * 0.5
    x = paddle.ones(shape=[Nbatch, 3, 3])
    mean, std = sde.marginal_prob(x=x, t=t, batch=batch)
    coeff = paddle.exp(
        x=-0.25 * t**2 * (sde.beta_1 - sde.beta_0) - 0.5 * t * sde.beta_0
    )
    expected_mean = coeff * x + (1 - coeff)[:, None, None] * (
        paddle.eye(num_rows=3)[None] * batch.num_atoms[:, None, None] / density
    ).pow(y=1.0 / 3)
    expected_var = 1 - paddle.exp(
        x=-0.5 * t**2 * (sde.beta_1 - sde.beta_0) - t * sde.beta_0
    )
    expected_var = expected_var * sde.get_limit_var(x=x, batch=batch)
    assert tuple(mean.shape) == (Nbatch, 3, 3)
    assert tuple(std.shape) == (Nbatch, 3, 3)
    assert paddle.allclose(x=expected_mean, y=mean).item()
    assert paddle.allclose(x=expected_var.sqrt(), y=std).item()


def test_make_noise_symmetric_preserve_variance():
    noise = paddle.randn(shape=[100000, 3, 3])
    symmetric_noise = make_noise_symmetric_preserve_variance(noise)
    assert paddle.allclose(x=noise.var(), y=symmetric_noise.var(), atol=0.01).item()
    assert paddle.allclose(x=noise.mean(), y=symmetric_noise.mean(), atol=0.01).item()
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(paddle.randn(shape=[100000, 3, 4]))
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(paddle.randn(shape=[100000, 3]))
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(paddle.randn(shape=[100000, 3, 1]))


@pytest.mark.parametrize(
    "output_shape", [(10, 3, 3), (10, 3, 1), (10, 3), (10, 2), (10, 3, 9, 1)]
)
def test_expand(output_shape: Tuple):
    unexpanded_data = paddle.randn(shape=(10,))
    expanded_data = expand(unexpanded_data, output_shape)
    assert len(tuple(expanded_data.shape)) == len(output_shape)
    assert tuple(expanded_data.shape) != output_shape
