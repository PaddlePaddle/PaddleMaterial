# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alignment and regression tests for CrystalFormer."""

import importlib.util
import os
import sys
import unittest

import numpy as np
import paddle

# Direct import of the crystalformer module, bypassing ppmat.__init__
# which pulls heavy deps (pgl, etc.) that may not be installed.
_mod_path = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "ppmat", "models", "crystalformer", "crystalformer.py",
    )
)
_spec = importlib.util.spec_from_file_location(
    "ppmat.models.crystalformer.crystalformer", _mod_path
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CrystalFormer = _mod.CrystalFormer
CrystalformerEncoderLayer = _mod.CrystalformerEncoderLayer
GaussianRBF = _mod.GaussianRBF
LatticeAttention = _mod.LatticeAttention
LatticeDistanceComputer = _mod.LatticeDistanceComputer


def _make_si_crystal(B=1):
    """Create a silicon diamond-cubic unit cell (2 atoms).

    Returns a data dict suitable for ``CrystalFormer.forward``.
    """
    N = 2
    x = paddle.zeros([B, N, 98], dtype="float32")
    for b in range(B):
        x[b, 0, 13] = 1.0  # Si (Z=14, 0-indexed)
        x[b, 1, 13] = 1.0

    pos = paddle.to_tensor(
        [[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]] * B, dtype="float32"
    )
    a = 5.43
    trans_vec = paddle.to_tensor(
        [[[a, 0, 0], [0, a, 0], [0, 0, a]]] * B, dtype="float32"
    )
    mask = paddle.ones([B, N], dtype="bool")
    label = paddle.to_tensor([[-0.5]] * B, dtype="float32")

    return {
        "x": x,
        "pos": pos,
        "trans_vec": trans_vec,
        "mask": mask,
        "sizes": paddle.to_tensor([N] * B, dtype="int64"),
        "formation_energy_per_atom": label,
    }


def _make_small_model(**overrides):
    """Build a small CrystalFormer for tests."""
    defaults = dict(
        atom_feat_dim=98,
        model_dim=64,
        n_heads=4,
        ff_dim=128,
        n_layers=2,
        n_gaussians=20,
        cutoff=8.0,
        lattice_range=1,
        embedding_dim=[64, 32],
        dropout=0.0,
        property_names="formation_energy_per_atom",
    )
    defaults.update(overrides)
    return CrystalFormer(**defaults)


class TestGaussianRBF(unittest.TestCase):
    """Test the Gaussian radial basis function layer."""

    def test_output_shape(self):
        rbf = GaussianRBF(n_gaussians=30, cutoff=6.0)
        d = paddle.to_tensor([0.0, 1.0, 3.0, 6.0], dtype="float32")
        out = rbf(d)
        self.assertEqual(list(out.shape), [4, 30])

    def test_peak_at_centre(self):
        """RBF at offset i should peak when distance == offsets[i]."""
        rbf = GaussianRBF(n_gaussians=50, cutoff=10.0)
        d = rbf.offsets[25:26]  # exact centre of one basis
        out = rbf(d)  # [1, 50]
        self.assertEqual(out.numpy().argmax(), 25)


class TestLatticeDistanceComputer(unittest.TestCase):
    """Test periodic distance computation."""

    def test_self_distance_zero(self):
        """Distance from an atom to itself (same image) should be zero."""
        comp = LatticeDistanceComputer(lattice_range=1)
        pos = paddle.to_tensor([[[0.0, 0.0, 0.0]]], dtype="float32")
        tv = paddle.to_tensor([[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]], dtype="float32")
        d = comp(pos, tv)
        self.assertAlmostEqual(d[0, 0, 0].item(), 0.0, places=5)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        comp = LatticeDistanceComputer(lattice_range=1)
        data = _make_si_crystal()
        d = comp(data["pos"], data["trans_vec"])
        np.testing.assert_allclose(
            d.numpy(), d.numpy().transpose(0, 2, 1), atol=1e-5
        )

    def test_periodic_closer_than_direct(self):
        """Two atoms near opposite cell faces should be closer via PBC."""
        comp = LatticeDistanceComputer(lattice_range=1)
        pos = paddle.to_tensor([[[0.01, 0.0, 0.0], [0.99, 0.0, 0.0]]], dtype="float32")
        tv = paddle.to_tensor([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]], dtype="float32")
        d = comp(pos, tv)
        periodic_dist = d[0, 0, 1].item()
        direct_dist = 0.98 * 10.0  # 9.8 Å without PBC
        self.assertLess(periodic_dist, direct_dist)
        np.testing.assert_allclose(periodic_dist, 0.2, atol=0.05)


class TestLatticeAttention(unittest.TestCase):
    """Test the lattice-distance-biased attention module."""

    def test_output_shape(self):
        attn = LatticeAttention(model_dim=64, n_heads=4, n_gaussians=20)
        B, N, D = 2, 5, 64
        x = paddle.randn([B, N, D])
        dist_rbf = paddle.randn([B, N, N, 20])
        out = attn(x, dist_rbf)
        self.assertEqual(list(out.shape), [B, N, D])


class TestCrystalFormerForwardAlignment(unittest.TestCase):
    """End-to-end tests for CrystalFormer."""

    def setUp(self):
        paddle.seed(42)
        self.model = _make_small_model()
        self.model.eval()

    def test_forward_shape(self):
        """Output prediction has shape [B, 1]."""
        data = _make_si_crystal()
        result = self.model(data)
        pred = result["pred_dict"]["formation_energy_per_atom"]
        self.assertEqual(list(pred.shape), [1, 1])

    def test_forward_determinism(self):
        """Two forward passes with same input yield identical output."""
        data = _make_si_crystal()
        r1 = self.model(data)["pred_dict"]["formation_energy_per_atom"]
        r2 = self.model(data)["pred_dict"]["formation_energy_per_atom"]
        np.testing.assert_allclose(r1.numpy(), r2.numpy(), atol=1e-6)

    def test_loss_computation(self):
        """Loss is a finite positive scalar."""
        data = _make_si_crystal()
        result = self.model(data, return_loss=True)
        loss = result["loss_dict"]["loss"]
        self.assertEqual(list(loss.shape), [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_predict_method(self):
        """``predict()`` returns un-normalised predictions without loss."""
        data = _make_si_crystal()
        pred = self.model.predict(data)
        self.assertIn("formation_energy_per_atom", pred)
        self.assertEqual(list(pred["formation_energy_per_atom"].shape), [1, 1])

    def test_batched_crystals(self):
        """Batched forward produces correct output shape."""
        data = _make_si_crystal(B=4)
        result = self.model(data)
        pred = result["pred_dict"]["formation_energy_per_atom"]
        self.assertEqual(list(pred.shape), [4, 1])

    def test_normalization_roundtrip(self):
        """normalize then unnormalize should be identity."""
        model = _make_small_model(data_mean=-0.5, data_std=2.0)
        t = paddle.to_tensor([1.0, 2.0, 3.0])
        recovered = model.unnormalize(model.normalize(t))
        np.testing.assert_allclose(t.numpy(), recovered.numpy(), atol=1e-5)

    def test_loss_type_l1(self):
        """L1 loss variant works."""
        model = _make_small_model(loss_type="l1_loss")
        model.eval()
        data = _make_si_crystal()
        result = model(data, return_loss=True)
        loss = result["loss_dict"]["loss"]
        self.assertTrue(paddle.isfinite(loss).item())

    def test_mean_pooling(self):
        """Mean-pooling variant produces valid output."""
        model = _make_small_model(pooling="mean")
        model.eval()
        data = _make_si_crystal()
        result = model(data)
        pred = result["pred_dict"]["formation_energy_per_atom"]
        self.assertEqual(list(pred.shape), [1, 1])
        self.assertTrue(paddle.isfinite(pred).item())


if __name__ == "__main__":
    unittest.main()
