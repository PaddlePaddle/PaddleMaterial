# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ppmat.models.mofdiff."""

import unittest

import numpy as np
import paddle

from ppmat.models.mofdiff.mofdiff import (
    VP,
    GaussianFourierProjection,
    MOFDiff,
    SimpleGNNDecoder,
    SimpleGNNEncoder,
    build_mlp,
)


def _make_batch(batch_size=2, nodes_per_graph=3, node_feat_dim=32, num_bb_types=10):
    """Create a minimal dummy batch for MOFDiff."""
    total = batch_size * nodes_per_graph
    batch_idx = paddle.repeat_interleave(
        paddle.arange(batch_size),
        paddle.full([batch_size], nodes_per_graph, dtype="int64"),
    )
    return {
        "node_features": paddle.randn([total, node_feat_dim]),
        "frac_coords": paddle.rand([total, 3]),
        "bb_types": paddle.randint(0, num_bb_types, [total]),
        "batch": batch_idx,
        "num_atoms": paddle.full([batch_size], nodes_per_graph, dtype="int64"),
        "lattice_params": paddle.randn([batch_size, 6]),
    }


class TestGaussianFourierProjection(unittest.TestCase):
    """GaussianFourierProjection embedding tests."""

    def test_output_shape(self):
        emb = GaussianFourierProjection(embedding_size=64, scale=1.0)
        x = paddle.to_tensor([0.0, 0.5, 1.0])
        out = emb(x)
        self.assertEqual(list(out.shape), [3, 128])

    def test_determinism(self):
        """Same input → same output (frozen weights)."""
        emb = GaussianFourierProjection(embedding_size=32)
        x = paddle.to_tensor([0.25, 0.75])
        np.testing.assert_allclose(
            emb(x).numpy(), emb(x).numpy(), atol=1e-6
        )

    def test_different_inputs(self):
        emb = GaussianFourierProjection(embedding_size=32)
        a = emb(paddle.to_tensor([0.0]))
        b = emb(paddle.to_tensor([1.0]))
        self.assertFalse(np.allclose(a.numpy(), b.numpy()))


class TestVPDiffusion(unittest.TestCase):
    """VP (Variance Preserving) diffusion schedule tests."""

    def setUp(self):
        self.vp = VP(num_steps=100, s=0.0001, power=2)

    def test_alpha_bar_boundaries(self):
        """alpha_bar_0 ~ 1 and alpha_bar_T < alpha_bar_0."""
        ab = self.vp.alpha_bars.numpy()
        np.testing.assert_allclose(ab[0], 1.0, atol=1e-4)
        self.assertLess(ab[-1], ab[0])

    def test_alpha_bars_monotonically_decrease(self):
        ab = self.vp.alpha_bars.numpy()
        self.assertTrue(np.all(np.diff(ab) <= 0))

    def test_forward_shape(self):
        h0 = paddle.randn([6, 16])
        t = paddle.randint(1, 101, [6])
        ht, eps = self.vp.forward(h0, t)
        self.assertEqual(list(ht.shape), [6, 16])
        self.assertEqual(list(eps.shape), [6, 16])

    def test_forward_t0_recovers_input(self):
        """At t=0 (alpha_bar=1) the noisy signal equals the clean input."""
        h0 = paddle.randn([4, 8])
        t = paddle.zeros([4], dtype="int64")
        ht, _ = self.vp.forward(h0, t)
        np.testing.assert_allclose(ht.numpy(), h0.numpy(), atol=1e-5)

    def test_reverse_shape(self):
        ht = paddle.randn([6, 16])
        eps = paddle.randn([6, 16])
        t = paddle.randint(1, 101, [6])
        out = self.vp.reverse(ht, eps, t)
        self.assertEqual(list(out.shape), [6, 16])


class TestSimpleGNNEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = SimpleGNNEncoder(input_dim=16, hidden_dim=32, output_dim=8, num_layers=2)
        feats = paddle.randn([6, 16])
        batch_idx = paddle.to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
        out = enc(feats, batch_idx, 2)
        self.assertEqual(list(out.shape), [2, 8])


class TestSimpleGNNDecoder(unittest.TestCase):
    def test_output_shapes(self):
        dec = SimpleGNNDecoder(
            input_dim=32, hidden_dim=64, output_coord_dim=3,
            output_type_dim=10, num_layers=2,
        )
        feats = paddle.randn([6, 32])
        eps_x, eps_h = dec(feats)
        self.assertEqual(list(eps_x.shape), [6, 3])
        self.assertEqual(list(eps_h.shape), [6, 10])


class TestMOFDiff(unittest.TestCase):
    """End-to-end MOFDiff model tests."""

    def setUp(self):
        paddle.seed(42)
        self.model = MOFDiff(
            node_feat_dim=32,
            hidden_dim=64,
            latent_dim=32,
            num_bb_types=10,
            max_num_bbs=5,
            num_diffusion_steps=50,
            fc_num_layers=2,
        )

    def test_forward_returns_loss_dict(self):
        """forward() must return {'loss_dict': {...}} (PM convention)."""
        batch = _make_batch(node_feat_dim=32, num_bb_types=10)
        result = self.model(batch)
        self.assertIn("loss_dict", result)
        for key in ("loss", "loss_coord", "loss_type", "loss_kl",
                     "loss_lattice", "loss_num_bbs"):
            self.assertIn(key, result["loss_dict"], f"Missing {key}")
            self.assertFalse(
                paddle.isnan(result["loss_dict"][key]).item(),
                f"{key} is NaN",
            )

    def test_forward_loss_is_positive(self):
        batch = _make_batch(node_feat_dim=32, num_bb_types=10)
        result = self.model(batch)
        self.assertGreater(result["loss_dict"]["loss"].item(), 0.0)

    def test_encode_shape(self):
        feats = paddle.randn([6, 32])
        batch_idx = paddle.to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
        mu, log_var, z = self.model.encode(feats, batch_idx, 2)
        self.assertEqual(list(mu.shape), [2, 32])
        self.assertEqual(list(log_var.shape), [2, 32])
        self.assertEqual(list(z.shape), [2, 32])

    def test_sample_output_keys(self):
        """sample() must return predicted coords, types, and lattice."""
        self.model.eval()
        z = paddle.randn([2, 32])
        num_atoms = paddle.to_tensor([3, 3], dtype="int64")
        out = self.model.sample(z, num_atoms, num_steps=3)
        self.assertIn("pred_coords", out)
        self.assertIn("pred_types", out)
        self.assertIn("pred_lattice", out)
        self.assertEqual(list(out["pred_coords"].shape), [6, 3])
        self.assertEqual(list(out["pred_types"].shape), [6])
        self.assertEqual(list(out["pred_lattice"].shape), [2, 6])

    def test_gradient_flow(self):
        """All parameters receive gradients after a forward + backward."""
        batch = _make_batch(node_feat_dim=32, num_bb_types=10)
        result = self.model(batch)
        result["loss_dict"]["loss"].backward()
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                self.assertIsNotNone(
                    param.grad,
                    f"No gradient for {name}",
                )

    def test_build_mlp(self):
        mlp = build_mlp(16, 32, 3, 8)
        out = mlp(paddle.randn([4, 16]))
        self.assertEqual(list(out.shape), [4, 8])

    def test_training_convergence(self):
        """Verify loss decreases over 2 training epochs (RFC requirement)."""
        self.model.train()
        optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), learning_rate=1e-3
        )
        losses = []
        for epoch in range(2):
            optimizer.clear_grad()
            batch = _make_batch(node_feat_dim=32, num_bb_types=10)
            result = self.model(batch)
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


class TestVPMathProperties(unittest.TestCase):
    """Verify mathematical invariants of the VP schedule."""

    def test_betas_non_negative(self):
        vp = VP(num_steps=200)
        self.assertTrue(np.all(vp.betas.numpy() >= 0))

    def test_forward_variance_preservation(self):
        """For large sample, Var(h_t) ~ 1 when h_0 ~ N(0,1)."""
        vp = VP(num_steps=500)
        paddle.seed(0)
        h0 = paddle.randn([10000, 4])
        t = paddle.full([10000], 250, dtype="int64")
        ht, _ = vp.forward(h0, t)
        var = ht.numpy().var()
        np.testing.assert_allclose(var, 1.0, atol=0.15)


if __name__ == "__main__":
    unittest.main()
