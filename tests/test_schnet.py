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

"""Unit tests for SchNet model (PaddleMaterials implementation).

Tests model instantiation, weight loading, forward pass correctness,
and component-level behavior without requiring GPU or pretrained weights.

Run with:
    python tests/test_schnet.py
"""

import importlib.util
import os
import sys
import types
import unittest

# Pre-register ppmat packages as stubs to avoid __init__.py import chain
# which requires pgl, pymatgen, etc.
for _pkg in [
    "ppmat",
    "ppmat.datasets",
    "ppmat.losses",
    "ppmat.metrics",
    "ppmat.models",
    "ppmat.optimizer",
    "ppmat.schedulers",
    "ppmat.trainer",
    "ppmat.utils",
    "ppmat.sampler",
]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        sys.modules[_pkg] = _m

# Project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_mod("ppmat.utils.paddle_aux", "ppmat/utils/paddle_aux.py")
_load_mod("ppmat.utils.crystal", "ppmat/utils/crystal.py")
_load_mod("ppmat.utils.scatter", "ppmat/utils/scatter.py")
_load_mod("ppmat.models.schnet.schnet", "ppmat/models/schnet/schnet.py")

import numpy as np
import paddle

from ppmat.models.schnet.schnet import CFConv
from ppmat.models.schnet.schnet import GaussianRBF
from ppmat.models.schnet.schnet import SchNet
from ppmat.models.schnet.schnet import SchNetInteraction
from ppmat.models.schnet.schnet import shifted_softplus
from ppmat.utils.scatter import scatter


class TestSchNetComponents(unittest.TestCase):
    """Test individual SchNet building blocks."""

    def test_shifted_softplus(self):
        """shifted_softplus(0) should be 0, and output should be smooth."""
        x = paddle.to_tensor([0.0, 1.0, -1.0, 5.0])
        y = shifted_softplus(x)
        # ssp(0) = softplus(0) - ln2 = ln2 - ln2 = 0
        np.testing.assert_allclose(y[0].numpy(), 0.0, atol=1e-6)
        # ssp should be monotonically increasing
        self.assertGreater(y[1].item(), y[0].item())
        self.assertGreater(y[3].item(), y[1].item())

    def test_gaussian_rbf(self):
        """GaussianRBF should expand distances to n_gaussians features."""
        n_gaussians = 25
        cutoff = 5.0
        rbf = GaussianRBF(n_gaussians=n_gaussians, cutoff=cutoff)

        dist = paddle.to_tensor([0.5, 1.0, 2.5, 4.9])
        out = rbf(dist)
        self.assertEqual(out.shape, [4, n_gaussians])
        # All values should be in [0, 1] (Gaussian)
        self.assertTrue((out.numpy() >= 0.0).all())
        self.assertTrue((out.numpy() <= 1.0).all())
        # Distance at offset[0]=0.0 should give peak at first Gaussian
        single = rbf(paddle.to_tensor([0.0]))
        self.assertGreater(single[0, 0].item(), 0.9)

    def test_cfconv_shape(self):
        """CFConv should produce correct output shape."""
        n_atom_basis = 64
        n_filters = 64
        n_gaussians = 25
        cfconv = CFConv(n_atom_basis, n_filters, n_gaussians)

        num_atoms = 5
        num_edges = 12
        x = paddle.randn([num_atoms, n_atom_basis])
        rbf = paddle.randn([num_edges, n_gaussians])
        edge_src = paddle.to_tensor(
            np.random.randint(0, num_atoms, num_edges).astype(np.int64)
        )
        edge_dst = paddle.to_tensor(
            np.random.randint(0, num_atoms, num_edges).astype(np.int64)
        )

        out = cfconv(x, rbf, edge_src, edge_dst, num_atoms)
        self.assertEqual(out.shape, [num_atoms, n_atom_basis])

    def test_interaction_residual(self):
        """SchNetInteraction should preserve input shape and apply residual."""
        n_atom_basis = 64
        interaction = SchNetInteraction(n_atom_basis, n_atom_basis, 25)

        num_atoms = 3
        x = paddle.ones([num_atoms, n_atom_basis])
        rbf = paddle.randn([6, 25])
        edge_src = paddle.to_tensor([0, 0, 1, 1, 2, 2], dtype="int64")
        edge_dst = paddle.to_tensor([1, 2, 0, 2, 0, 1], dtype="int64")

        out = interaction(x, rbf, edge_src, edge_dst, num_atoms)
        self.assertEqual(out.shape, [num_atoms, n_atom_basis])
        # Residual connection: output should differ from input
        self.assertFalse(np.allclose(out.numpy(), 1.0, atol=1e-6))


class TestSchNetModel(unittest.TestCase):
    """Test the full SchNet model."""

    def test_model_instantiation(self):
        """Model should instantiate with default parameters."""
        model = SchNet()
        self.assertIsInstance(model, paddle.nn.Layer)
        self.assertEqual(len(model.interactions), 6)
        self.assertEqual(model.cutoff, 10.0)

    def test_model_qm9_param_count(self):
        """QM9 config (128-dim, 6 interactions) should have ~456K params."""
        model = SchNet(n_atom_basis=128, n_interactions=6, n_gaussians=50)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 400000)
        self.assertLess(n_params, 600000)

    def test_model_md17_param_count(self):
        """MD17 config (64-dim, 6 interactions, 25 gaussians) should be ~63K."""
        model = SchNet(n_atom_basis=64, n_interactions=6, n_gaussians=25, cutoff=5.0)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 50000)
        self.assertLess(n_params, 200000)

    def test_weight_loading_qm9(self):
        """Model should load converted QM9 weights without errors."""
        params_path = os.path.join(
            _ROOT, "checkpoints", "schnet_qm9_U0", "schnet_qm9_U0.pdparams"
        )
        if not os.path.exists(params_path):
            self.skipTest(f"Weights not found at {params_path}")

        state_dict = paddle.load(params_path)
        model = SchNet(
            n_atom_basis=128,
            n_interactions=6,
            n_gaussians=50,
            cutoff=10.0,
            data_mean=float(state_dict.get("data_mean", 0.0)),
            data_std=float(state_dict.get("data_std", 1.0)),
        )
        missing, unexpected = model.set_state_dict(state_dict)
        unexpected_real = [k for k in unexpected if "atomref" not in k]
        self.assertEqual(len(missing), 0, f"Missing keys: {missing}")
        self.assertEqual(len(unexpected_real), 0, f"Unexpected keys: {unexpected_real}")

    def test_weight_loading_md17(self):
        """Model should load converted MD17 weights without errors."""
        params_path = os.path.join(
            _ROOT, "checkpoints", "schnet_md17_ethanol", "schnet_md17_ethanol.pdparams"
        )
        if not os.path.exists(params_path):
            self.skipTest(f"Weights not found at {params_path}")

        state_dict = paddle.load(params_path)
        model = SchNet(
            n_atom_basis=64,
            n_interactions=6,
            n_gaussians=25,
            cutoff=5.0,
            data_mean=float(state_dict.get("data_mean", 0.0)),
            data_std=float(state_dict.get("data_std", 1.0)),
        )
        missing, unexpected = model.set_state_dict(state_dict)
        self.assertEqual(len(missing), 0, f"Missing keys: {missing}")
        self.assertEqual(len(unexpected), 0, f"Unexpected keys: {unexpected}")

    def test_core_forward_pass(self):
        """Test forward pass through core layers (bypassing PGL graph)."""
        model = SchNet(n_atom_basis=64, n_interactions=3, n_gaussians=25, cutoff=5.0)
        model.eval()

        # Simulate a 3-atom molecule (water)
        num_atoms = 3
        atom_types = paddle.to_tensor([8, 1, 1], dtype="int64")
        edge_src = paddle.to_tensor([0, 0, 1, 1, 2, 2], dtype="int64")
        edge_dst = paddle.to_tensor([1, 2, 0, 2, 0, 1], dtype="int64")
        dist = paddle.to_tensor([0.96, 0.96, 0.96, 1.52, 0.96, 1.52])
        batch = paddle.zeros([num_atoms], dtype="int64")

        with paddle.no_grad():
            rbf = model.rbf(dist)
            x = model.embedding(atom_types)
            for interaction in model.interactions:
                x = interaction(x, rbf, edge_src, edge_dst, num_atoms)
            x = shifted_softplus(model.output_network[0](x))
            atom_energy = model.output_network[1](x)
            energy = scatter(atom_energy, batch, dim=0, reduce="sum")

        self.assertEqual(energy.shape, [1, 1])
        self.assertTrue(np.isfinite(energy.numpy()).all())

    def test_core_forward_with_pretrained_weights(self):
        """Forward pass with pretrained QM9 weights should give finite energy."""
        params_path = os.path.join(
            _ROOT, "checkpoints", "schnet_qm9_U0", "schnet_qm9_U0.pdparams"
        )
        if not os.path.exists(params_path):
            self.skipTest("QM9 weights not available")

        state_dict = paddle.load(params_path)
        model = SchNet(
            n_atom_basis=128,
            n_interactions=6,
            n_gaussians=50,
            cutoff=10.0,
            data_mean=float(state_dict.get("data_mean", 0.0)),
            data_std=float(state_dict.get("data_std", 1.0)),
        )
        model.set_state_dict(state_dict)
        model.eval()

        # Water molecule
        atom_types = paddle.to_tensor([8, 1, 1], dtype="int64")
        edge_src = paddle.to_tensor([0, 0, 1, 1, 2, 2], dtype="int64")
        edge_dst = paddle.to_tensor([1, 2, 0, 2, 0, 1], dtype="int64")
        dist = paddle.to_tensor([0.96, 0.96, 0.96, 1.52, 0.96, 1.52])
        batch = paddle.zeros([3], dtype="int64")

        with paddle.no_grad():
            rbf = model.rbf(dist)
            x = model.embedding(atom_types)
            for interaction in model.interactions:
                x = interaction(x, rbf, edge_src, edge_dst, 3)
            x = shifted_softplus(model.output_network[0](x))
            atom_energy = model.output_network[1](x)
            energy = scatter(atom_energy, batch, dim=0, reduce="sum")
            # Unnormalize
            energy = energy * model.data_std + model.data_mean

        energy_val = energy.numpy().flatten()[0]
        self.assertTrue(np.isfinite(energy_val), "Energy should be finite")
        # Without atomref correction, absolute value won't match physical reality
        # but should still be in a reasonable range for QM9 Hartree units
        self.assertGreater(abs(energy_val), 0.01, "Energy should be non-trivial")
        print(f"\n  QM9 U0 energy for water (no atomref): {energy_val:.6f} Ha")

    def test_normalize_unnormalize(self):
        """Normalization/unnormalization should be inverse operations."""
        model = SchNet(data_mean=5.0, data_std=2.0)
        x = paddle.to_tensor([10.0, 20.0, 30.0])
        normalized = model.normalize(x)
        unnormalized = model.unnormalize(normalized)
        np.testing.assert_allclose(x.numpy(), unnormalized.numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
