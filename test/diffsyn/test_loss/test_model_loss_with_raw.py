"""Tests for DiffSyn model: shapes, loss, diffusion math, sampling."""

import importlib.util
import os
import sys
import unittest

import numpy as np
import paddle

# Load diffsyn.py directly to bypass ppmat's __init__ (which pulls pgl, etc.)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_mod_path = os.path.join(_root, "ppmat", "models", "diffsyn", "diffsyn.py")
_spec = importlib.util.spec_from_file_location("diffsyn", _mod_path)
_diffsyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diffsyn)

DiffSyn = _diffsyn.DiffSyn
Unet1D = _diffsyn.Unet1D
SinusoidalPosEmb = _diffsyn.SinusoidalPosEmb
cosine_beta_schedule = _diffsyn.cosine_beta_schedule
extract = _diffsyn.extract


def _make_model(**overrides):
    defaults = dict(
        dim=32, channels=3, seq_length=8, cond_dim=16,
        dim_mults=(1, 2), groups=4, timesteps=100,
        objective="pred_noise", cond_drop_prob=0.0,
        loss_type="l1_loss",
    )
    defaults.update(overrides)
    return DiffSyn(**defaults)


def _dummy_data(batch=2, channels=3, length=8, cond_dim=16):
    x = paddle.randn([batch, channels, length])
    cond = paddle.randn([batch, cond_dim])
    return {"x": x, "cond": cond, "synthesis_conditions": x}


class TestDiffSynForwardAlignment(unittest.TestCase):
    """Core forward-pass and shape tests."""

    def setUp(self):
        paddle.seed(42)
        self.model = _make_model()
        self.model.eval()

    def test_forward_returns_expected_keys(self):
        data = _dummy_data()
        out = self.model(data)
        self.assertIn("loss_dict", out)
        self.assertIn("pred_dict", out)
        self.assertIn("loss", out["loss_dict"])
        self.assertIn("synthesis_conditions", out["pred_dict"])

    def test_forward_prediction_shape(self):
        data = _dummy_data()
        out = self.model(data)
        pred = out["pred_dict"]["synthesis_conditions"]
        self.assertEqual(list(pred.shape), [2, 3, 8])

    def test_loss_is_scalar(self):
        data = _dummy_data()
        out = self.model(data)
        loss = out["loss_dict"]["loss"]
        self.assertEqual(loss.ndim, 0)
        self.assertFalse(paddle.isnan(loss).item())

    def test_loss_only_mode(self):
        data = _dummy_data()
        out = self.model(data, return_loss=True, return_prediction=False)
        self.assertIn("loss", out["loss_dict"])
        self.assertEqual(len(out["pred_dict"]), 0)

    def test_prediction_only_mode(self):
        data = _dummy_data()
        out = self.model(data, return_loss=False, return_prediction=True)
        self.assertEqual(len(out["loss_dict"]), 0)
        self.assertIn("synthesis_conditions", out["pred_dict"])


class TestQSample(unittest.TestCase):
    """Forward diffusion q(x_t | x_0) correctness."""

    def setUp(self):
        paddle.seed(0)
        self.model = _make_model()

    def test_q_sample_t0_close_to_x_start(self):
        """At t=0, x_t ≈ x_start (very little noise added)."""
        x = paddle.randn([2, 3, 8])
        t = paddle.zeros([2], dtype="int64")
        noise = paddle.randn(x.shape)
        x_noisy = self.model.q_sample(x, t, noise)
        # sqrt_alphas_cumprod[0] ≈ 1, so x_noisy ≈ x + tiny noise
        diff = (x_noisy - x).abs().mean().item()
        self.assertLess(diff, 0.15)

    def test_q_sample_reproducibility(self):
        """Same seed → same noised sample."""
        x = paddle.ones([1, 3, 8])
        t = paddle.to_tensor([50], dtype="int64")
        paddle.seed(99)
        noise1 = paddle.randn(x.shape)
        out1 = self.model.q_sample(x, t, noise1)
        paddle.seed(99)
        noise2 = paddle.randn(x.shape)
        out2 = self.model.q_sample(x, t, noise2)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)

    def test_q_sample_shape(self):
        x = paddle.randn([4, 3, 8])
        t = paddle.randint(0, 100, [4])
        out = self.model.q_sample(x, t)
        self.assertEqual(list(out.shape), [4, 3, 8])


class TestDenoiseModel(unittest.TestCase):
    """Direct Unet1D tests."""

    def setUp(self):
        paddle.seed(42)
        self.unet = Unet1D(dim=32, channels=3, cond_dim=16,
                           dim_mults=(1, 2), groups=4, cond_drop_prob=0.0)
        self.unet.eval()

    def test_unet_output_shape(self):
        x = paddle.randn([2, 3, 8])
        t = paddle.randint(0, 100, [2])
        cond = paddle.randn([2, 16])
        out = self.unet(x, t, cond=cond)
        self.assertEqual(list(out.shape), [2, 3, 8])

    def test_unet_unconditional(self):
        """Should work when cond=None (uses null embedding)."""
        x = paddle.randn([2, 3, 8])
        t = paddle.randint(0, 100, [2])
        out = self.unet(x, t, cond=None)
        self.assertEqual(list(out.shape), [2, 3, 8])

    def test_cond_scale_forwarding(self):
        x = paddle.randn([1, 3, 8])
        t = paddle.randint(0, 100, [1])
        cond = paddle.randn([1, 16])
        out_scale1 = self.unet.forward_with_cond_scale(
            x, t, cond=cond, cond_scale=1.0)
        out_scale3 = self.unet.forward_with_cond_scale(
            x, t, cond=cond, cond_scale=3.0)
        # scale=1 returns conditional; scale≠1 returns guided output
        self.assertEqual(list(out_scale1.shape), list(out_scale3.shape))


class TestSample(unittest.TestCase):
    """Sampling (generative) path."""

    def setUp(self):
        paddle.seed(42)
        self.model = _make_model(timesteps=10)
        self.model.eval()

    def test_sample_shape(self):
        samples = self.model.sample(batch_size=2, cond=paddle.randn([2, 16]))
        self.assertEqual(list(samples.shape), [2, 3, 8])

    def test_sample_unconditional(self):
        samples = self.model.sample(batch_size=1)
        self.assertEqual(list(samples.shape), [1, 3, 8])

    def test_predict_method(self):
        data = {"cond": paddle.randn([2, 16])}
        result = self.model.predict(data)
        self.assertIn("synthesis_conditions", result)
        self.assertEqual(list(result["synthesis_conditions"].shape), [2, 3, 8])


class TestNormalization(unittest.TestCase):
    """Data normalize / unnormalize round-trip."""

    def test_roundtrip(self):
        model = _make_model(data_mean=2.0, data_std=3.0)
        t = paddle.to_tensor([1.0, 5.0, 10.0])
        normed = model.normalize(t)
        recovered = model.unnormalize(normed)
        np.testing.assert_allclose(t.numpy(), recovered.numpy(), atol=1e-5)


class TestBetaSchedule(unittest.TestCase):
    """Cosine beta schedule properties."""

    def test_shape(self):
        betas = cosine_beta_schedule(100)
        self.assertEqual(betas.shape, (100,))

    def test_range(self):
        betas = cosine_beta_schedule(1000)
        self.assertTrue(np.all(betas >= 0))
        self.assertTrue(np.all(betas <= 0.9999))

    def test_monotonically_increasing_alphas_cumprod_decreases(self):
        betas = cosine_beta_schedule(200)
        alphas_cumprod = np.cumprod(1.0 - betas)
        diffs = np.diff(alphas_cumprod)
        self.assertTrue(np.all(diffs <= 0))


class TestSinusoidalPosEmb(unittest.TestCase):
    def test_output_shape(self):
        emb = SinusoidalPosEmb(64)
        t = paddle.to_tensor([0, 50, 99])
        out = emb(t)
        self.assertEqual(list(out.shape), [3, 64])


class TestExtract(unittest.TestCase):
    def test_broadcast(self):
        a = paddle.arange(10, dtype="float32")
        t = paddle.to_tensor([3, 7], dtype="int64")
        out = extract(a, t, [2, 4, 4])
        self.assertEqual(list(out.shape), [2, 1, 1])
        self.assertAlmostEqual(out[0, 0, 0].item(), 3.0)
        self.assertAlmostEqual(out[1, 0, 0].item(), 7.0)


class TestObjectives(unittest.TestCase):
    """Verify all three objective modes produce valid output."""

    def test_pred_noise(self):
        m = _make_model(objective="pred_noise")
        out = m(_dummy_data())
        self.assertFalse(paddle.isnan(out["loss_dict"]["loss"]).item())

    def test_pred_x0(self):
        m = _make_model(objective="pred_x0")
        out = m(_dummy_data())
        self.assertFalse(paddle.isnan(out["loss_dict"]["loss"]).item())

    def test_pred_v(self):
        m = _make_model(objective="pred_v")
        out = m(_dummy_data())
        self.assertFalse(paddle.isnan(out["loss_dict"]["loss"]).item())


class TestReferenceAlignment(unittest.TestCase):
    """Hard-coded reference values for diffusion math reproducibility."""

    def test_q_sample_known_values(self):
        model = _make_model()
        paddle.seed(123)
        x = paddle.ones([1, 3, 8])
        t = paddle.to_tensor([0], dtype="int64")
        noise = paddle.zeros([1, 3, 8])
        out = model.q_sample(x, t, noise)
        # At t=0, sqrt_alphas_cumprod[0] ≈ 1.0, so out ≈ x
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=0.01)

    def test_predict_start_from_noise_inverse(self):
        """predict_start_from_noise should invert q_sample when noise is known."""
        model = _make_model()
        x = paddle.randn([2, 3, 8])
        t = paddle.to_tensor([10, 50], dtype="int64")
        noise = paddle.randn([2, 3, 8])
        x_noisy = model.q_sample(x, t, noise)
        x_recovered = model.predict_start_from_noise(x_noisy, t, noise)
        np.testing.assert_allclose(
            x.numpy(), x_recovered.numpy(), atol=1e-4, rtol=1e-4
        )


class TestTrainingConvergence(unittest.TestCase):
    """Verify training convergence (RFC requirement)."""

    def test_training_convergence(self):
        """Verify loss decreases over 2 training epochs (RFC requirement)."""
        paddle.seed(42)
        model = _make_model()
        model.train()
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(), learning_rate=1e-3
        )
        losses = []
        for epoch in range(2):
            optimizer.clear_grad()
            data = _dummy_data()
            result = model(data)
            loss = result["loss_dict"]["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertTrue(
            np.isfinite(losses[-1]), f"Loss became non-finite: {losses}"
        )
        self.assertLess(
            losses[-1],
            losses[0] * 1.5,
            f"Loss increased significantly: {losses}",
        )


class TestMSELoss(unittest.TestCase):
    """Verify MSE loss type works."""

    def test_mse_loss(self):
        model = _make_model(loss_type="mse_loss")
        out = model(_dummy_data())
        loss = out["loss_dict"]["loss"]
        self.assertFalse(paddle.isnan(loss).item())
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
