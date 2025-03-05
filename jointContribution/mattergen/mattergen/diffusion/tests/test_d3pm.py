import paddle

"""Tests for d3pm.py."""
import functools

import numpy as np
import pytest
from mattergen.diffusion.d3pm import d3pm as diffusion


@pytest.mark.parametrize("schedule_kind", ["linear", "standard", "cosine"])
def test_prior_kl(schedule_kind: str):
    """Test the prior KL computation."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind=schedule_kind, beta_min=0.001, beta_max=0.1, num_steps=1000
    )
    dim = 100
    num_samples = 71
    x_in = paddle.randint(low=0, high=dim, shape=(num_samples,))
    diff = diffusion.MaskDiffusion(dim=dim + 1, schedule=schedule)
    prior_kl = diffusion.compute_prior_kl(x_in, diff)
    assert paddle.isclose(x=prior_kl, y=paddle.to_tensor(data=0.0), atol=1e-05)


def test_product_the_hard_way():
    """Tests that the discrete transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}) are equivalent
    for t in {0, 1}. Uses the slow iterative method of computing the transition matrix q(x_t | x_0).
    """
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule, use_fast_inference=False)
    assert not diff.supports_efficient_inference()
    product = diff.get_qt_matrix(paddle.to_tensor(data=0))
    np.testing.assert_array_almost_equal(product, paddle.eye(num_rows=100))
    product = diff.get_qt_matrix(paddle.to_tensor(data=1)[None])
    np.testing.assert_array_almost_equal(product, diff.get(paddle.to_tensor(data=0)))


def test_product_fast():
    """Tests that the discrete transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}) are equivalent
    for t in {0, 1}. Uses the fast closed-form method of computing the transition matrix q(x_t | x_0).
    """
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule, use_fast_inference=True)
    assert diff.supports_efficient_inference()
    product = diff.get_qt_matrix(paddle.to_tensor(data=0))
    np.testing.assert_array_almost_equal(product, paddle.eye(num_rows=100))
    product = diff.get_qt_matrix(paddle.to_tensor(data=1))
    np.testing.assert_array_almost_equal(product, diff.get(paddle.to_tensor(data=0)))


def test_product_constant():
    """Tests, when we have a constant beta schedule (transition probabilities don't change over time),
    whether the transition matrices computed via q(x_t | x_0) and q(x_t|x_{t-1}), and via explicit matrix
    multiplication are equivalent."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    assert diff.supports_efficient_inference()
    product = diff.get_qt_matrix(0)
    np.testing.assert_array_almost_equal(product, paddle.eye(num_rows=100))
    product = diff.get_qt_matrix(1)
    np.testing.assert_array_almost_equal(product, diff.get(paddle.to_tensor(data=0)))
    product = diff.get_qt_matrix(10)
    expected = np.linalg.matrix_power(diff.get(paddle.to_tensor(data=0)), 10)
    np.testing.assert_array_almost_equal(product, expected)


def test_sample_and_posterior():
    """Tests whether the samples and posterior are as expected when providing timestep 0 for the sampling."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    inputs = paddle.ones(shape=(1,), dtype="int64")
    probs, sample = diff.sample_and_compute_posterior_q(
        inputs, paddle.to_tensor(data=[0]), return_logits=False
    )
    assert tuple(probs.shape) == (1, 100)
    assert paddle.allclose(
        x=probs[0, 1], y=paddle.to_tensor(data=1.0), atol=1e-05
    ).item()
    assert tuple(sample.shape) == (1,)
    np.testing.assert_array_equal(sample, np.array([1]))


def test_compute_posterior():
    """Tests that the forward diffusion probabilities are correct for t=0."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    inputs = paddle.ones(shape=(2,), dtype="int64")
    q_t = diff.get_qt_given_q0(inputs, paddle.to_tensor(data=[0, 0]), make_one_hot=True)
    assert tuple(q_t.shape) == (2, 100)
    assert paddle.allclose(x=q_t[0][1], y=paddle.to_tensor(data=1.0)).item()
    assert paddle.allclose(x=q_t[0][0], y=paddle.to_tensor(data=0.0)).item()


def test_model():
    """Test the Diffusion noise diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="standard", beta_min=0.001, beta_max=0.001, num_steps=100
    )
    dim = 100
    length = 100
    x0 = paddle.randint(low=0, high=dim, shape=(length,))
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    if hasattr(diffusion, "get"):
        np.testing.assert_allclose(diff.get(0).sum(axis=0), 1.0, rtol=1e-06)
        np.testing.assert_allclose(diff.get(10).sum(axis=0), 1.0, rtol=1e-06)
        np.testing.assert_allclose(diff.get(99).sum(axis=0), 1.0, rtol=1e-06)
        np.testing.assert_allclose(
            diff.get_qt_matrix(0), paddle.eye(num_rows=100), rtol=1e-06
        )
    expected = paddle.eye(num_rows=dim)[x0]
    result = diff.get_qt_given_q0(
        q0=x0, t=paddle.to_tensor(data=[0]), make_one_hot=True
    )
    np.testing.assert_allclose(result, expected)
    expected = paddle.nn.functional.softmax(paddle.randn(shape=(length, dim)), axis=-1)
    result = diff.get_qt_given_q0(
        q0=expected, t=paddle.to_tensor(data=[0]), make_one_hot=False
    )
    np.testing.assert_allclose(result, expected)
    q0 = paddle.nn.functional.softmax(paddle.randn(shape=(length, dim)), axis=-1)
    result = diff.get_qt_given_q0(
        q0=q0, t=paddle.to_tensor(data=[0]), make_one_hot=False
    )
    np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-06)
    expected = diff.stationary_probs(tuple(x0.shape))
    result = diff.get_qt_given_q0(
        q0=x0, t=paddle.to_tensor(data=[100]), make_one_hot=True
    )
    np.testing.assert_allclose(result, expected)


def test_mask_diffusion():
    """Test the Diffusion noise diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.1, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim=100, schedule=schedule)
    np.testing.assert_allclose(
        diff.get(paddle.to_tensor(data=0)).sum(axis=0), 1.0, rtol=1e-06
    )
    np.testing.assert_allclose(
        diff.get(paddle.to_tensor(data=10)).sum(axis=0), 1.0, rtol=1e-06
    )
    np.testing.assert_allclose(
        diff.get(paddle.to_tensor(data=0))[0, 0], 1.0 - schedule(0), rtol=1e-06
    )
    np.testing.assert_allclose(
        diff.get(paddle.to_tensor(data=1))[0, 0], 1.0 - schedule(1), rtol=1e-06
    )
    np.testing.assert_allclose(
        diff.get_qt_matrix(0), paddle.eye(num_rows=100), rtol=1e-06
    )


def test_mask_diffusion_slow_and_fast():
    """Compares fast and slow inference for mask diffusion."""
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="standard", beta_min=0.0005, beta_max=0.05, num_steps=100
    )
    dim = 16
    length = 16
    fast_diff = diffusion.MaskDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=True
    )
    slow_diff = diffusion.MaskDiffusion(
        dim=dim, schedule=schedule, use_fast_inference=False
    )
    x0 = paddle.randint(low=0, high=dim, shape=(length,))
    for _t in range(100):
        t = paddle.to_tensor(data=[_t]).expand_as(y=x0)
        _t_item = paddle.to_tensor(data=_t)
        qt_slow = slow_diff.get_qt_matrix(_t_item)
        qt_fast = fast_diff.get_qt_matrix(t)
        np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)
        qt_slow = slow_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
        qt_fast = fast_diff.get_qt_given_q0(q0=x0, t=t, make_one_hot=True)
        np.testing.assert_array_almost_equal(qt_slow, qt_fast, decimal=3)
        np.testing.assert_array_almost_equal(qt_slow.sum(axis=-1), 1.0, decimal=3)
        np.testing.assert_array_almost_equal(qt_fast.sum(axis=-1), 1.0, decimal=3)
        paddle.seed(seed=234)
        posterior_slow, samples_slow = slow_diff.sample_and_compute_posterior_q(
            x_0=x0, t=t, make_one_hot=True
        )
        paddle.seed(seed=234)
        posterior_fast, samples_fast = fast_diff.sample_and_compute_posterior_q(
            x_0=x0, t=t, make_one_hot=True
        )
        np.testing.assert_array_almost_equal(posterior_slow, posterior_fast, decimal=3)
        np.testing.assert_array_equal(samples_slow, samples_fast)
    t_100 = paddle.to_tensor(data=[100]).expand_as(y=x0)
    qt = fast_diff.get_qt_given_q0(q0=x0, t=t_100, make_one_hot=True)
    np.testing.assert_allclose(
        qt,
        paddle.eye(num_rows=dim)[
            paddle.full(shape=tuple(x0.shape), fill_value=dim - 1)
        ],
        rtol=1e-06,
    )
    qt = slow_diff.get_qt_given_q0(q0=x0, t=t_100, make_one_hot=True)
    np.testing.assert_allclose(
        qt,
        paddle.eye(num_rows=dim)[
            paddle.full(shape=tuple(x0.shape), fill_value=dim - 1)
        ],
        rtol=1e-06,
    )


def test_large_matrices():
    """Tests precision for large matrices."""
    dim = 1000
    length = 64
    x0 = paddle.randint(low=0, high=dim, shape=(length,))
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.0005, beta_max=0.05, num_steps=100
    )
    diff = diffusion.MaskDiffusion(dim, schedule, use_fast_inference=True)
    fn = functools.partial(diff.get_qt_given_q0, make_one_hot=True)
    result = fn(x0, paddle.to_tensor(data=[100]))
    np.testing.assert_array_almost_equal(result.sum(axis=-1), 1.0)


def test_loss_computation():
    """Tests whether the loss computation uses the right terms (KL / cross-entropy) and broadcasts correctly."""
    paddle.seed(seed=234)
    num_steps = 100
    num_classes = 7
    hybrid_lambda = 0.0
    schedule = diffusion.create_discrete_diffusion_schedule(
        kind="linear", beta_min=0.001, beta_max=0.001, num_steps=num_steps
    )
    t = paddle.arange(start=0, end=100)
    diff = diffusion.MaskDiffusion(dim=num_classes, schedule=schedule)
    inputs = paddle.ones(shape=(num_steps,), dtype="int64")
    q_t_minus_one, x_t_samples = diff.sample_and_compute_posterior_q(
        inputs, t, make_one_hot=True, return_logits=True
    )

    def denoise_fn(targets, timestep):
        return q_t_minus_one

    loss_dict = diffusion.compute_kl_reverse_process(
        x_start=inputs,
        t=t,
        x_t_plus_1=x_t_samples,
        diffusion=diff,
        denoise_fn=denoise_fn,
        predict_x0=False,
        hybrid_lambda=hybrid_lambda,
    )
    loss = loss_dict.pop("loss")
    kl_loss = loss_dict.pop("kl/kl_loss")
    cross_entropy_loss = loss_dict.pop("kl/cross_entropy_loss")
    assert tuple(loss.shape) == tuple(t.shape)
    assert paddle.allclose(x=kl_loss[1:], y=loss[1:]).item()
    assert paddle.allclose(x=cross_entropy_loss[:1], y=loss[:1]).item()
    assert paddle.allclose(x=kl_loss, y=paddle.zeros_like(x=kl_loss), atol=1e-06).item()
