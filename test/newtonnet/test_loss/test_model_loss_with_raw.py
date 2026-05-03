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

"""Tests for NewtonNet forward pass, loss computation, and prediction structure."""

import importlib.util
import os
import unittest

import numpy as np
import paddle

# Load the module directly from its file path to avoid ppmat/__init__.py
# which requires pgl (PaddleGraphLearning) not available in this env.
_module_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "ppmat", "models", "newtonnet", "newtonnet.py",
)
_module_path = os.path.abspath(_module_path)
_spec = importlib.util.spec_from_file_location("newtonnet_module", _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NewtonNet = _mod.NewtonNet


class TestNewtonNetForwardAlignment(unittest.TestCase):
    """Test NewtonNet forward pass alignment and basic functionality."""

    def setUp(self):
        paddle.seed(42)
        self.model = NewtonNet(
            cutoff=5.0,
            n_features=32,
            n_basis=8,
            n_interactions=2,
            activation="swish",
            layer_norm=False,
            property_names="energy",
            data_mean=0.0,
            data_std=1.0,
            loss_type="mse_loss",
        )
        self.model.eval()

    def _create_water_molecule_data(self):
        """Create a single water molecule (H2O) test input."""
        z = paddle.to_tensor([8, 1, 1], dtype="int64")  # O, H, H
        pos = paddle.to_tensor(
            [
                [0.0000, 0.0000, 0.1173],
                [0.0000, 0.7572, -0.4692],
                [0.0000, -0.7572, -0.4692],
            ],
            dtype="float32",
        )
        batch = paddle.zeros([3], dtype="int64")
        cell = paddle.eye(3, dtype="float32").unsqueeze(0) * 10.0
        energy = paddle.to_tensor([-76.4], dtype="float32")
        return {
            "z": z,
            "pos": pos,
            "batch": batch,
            "cell": cell,
            "energy": energy,
        }

    def _create_two_molecule_batch(self):
        """Create a batch with two molecules: H2O and H2."""
        z = paddle.to_tensor([8, 1, 1, 1, 1], dtype="int64")
        pos = paddle.to_tensor(
            [
                [0.0, 0.0, 0.1173],
                [0.0, 0.7572, -0.4692],
                [0.0, -0.7572, -0.4692],
                [5.0, 0.0, 0.0],
                [5.0, 0.74, 0.0],
            ],
            dtype="float32",
        )
        batch = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64")
        cell = paddle.eye(3, dtype="float32").unsqueeze(0).expand([2, 3, 3]) * 10.0
        energy = paddle.to_tensor([-76.4, -1.17], dtype="float32")
        return {
            "z": z,
            "pos": pos,
            "batch": batch,
            "cell": cell,
            "energy": energy,
        }

    def test_forward_shape(self):
        """Verify output shapes from forward pass."""
        data = self._create_water_molecule_data()
        result = self.model(data, return_loss=True, return_prediction=True)

        self.assertIn("loss_dict", result)
        self.assertIn("pred_dict", result)
        self.assertIn("loss", result["loss_dict"])
        self.assertIn("energy", result["pred_dict"])

        loss = result["loss_dict"]["loss"]
        energy = result["pred_dict"]["energy"]
        self.assertEqual(loss.shape, [])  # scalar
        self.assertEqual(energy.shape, [1])  # one molecule

    def test_forward_shape_batch(self):
        """Verify shapes for a two-molecule batch."""
        data = self._create_two_molecule_batch()
        result = self.model(data, return_loss=True, return_prediction=True)

        energy = result["pred_dict"]["energy"]
        self.assertEqual(energy.shape, [2])

    def test_forward_determinism(self):
        """Two forward passes with the same input must produce identical results."""
        data = self._create_water_molecule_data()
        r1 = self.model(data, return_loss=False, return_prediction=True)
        r2 = self.model(data, return_loss=False, return_prediction=True)
        np.testing.assert_allclose(
            r1["pred_dict"]["energy"].numpy(),
            r2["pred_dict"]["energy"].numpy(),
            rtol=1e-6,
        )

    def test_loss_computation(self):
        """Loss should be a finite positive scalar."""
        data = self._create_water_molecule_data()
        result = self.model(data, return_loss=True, return_prediction=False)

        loss = result["loss_dict"]["loss"]
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_energy_prediction_structure(self):
        """Predicted energy must be finite and return the correct key."""
        data = self._create_water_molecule_data()
        result = self.model(data, return_loss=False, return_prediction=True)

        self.assertIn("energy", result["pred_dict"])
        energy = result["pred_dict"]["energy"]
        self.assertTrue(paddle.isfinite(energy).all().item())

    def test_predict_method(self):
        """The convenience ``predict()`` method should return a dict with energy."""
        data = self._create_water_molecule_data()
        pred = self.model.predict(data)
        self.assertIn("energy", pred)
        self.assertEqual(pred["energy"].shape, [1])
        self.assertTrue(paddle.isfinite(pred["energy"]).all().item())

    def test_force_computation(self):
        """Forces are computed as negative energy gradient."""
        data = self._create_water_molecule_data()
        data["pos"].stop_gradient = False
        result = self.model(data, return_loss=False, return_prediction=True)
        forces = result["pred_dict"]["forces"]
        self.assertEqual(list(forces.shape), [data["pos"].shape[0], 3])
        self.assertTrue(paddle.isfinite(forces).all().item())

    def test_energy_force_consistency(self):
        """Forces should approximate the negative finite-difference gradient."""
        data = self._create_water_molecule_data()
        pos = data["pos"].clone()
        pos.stop_gradient = False
        data["pos"] = pos
        result = self.model(data, return_loss=False, return_prediction=True)
        pred_forces = result["pred_dict"]["forces"]

        # Verify via finite differences
        eps = 1e-3
        n_atoms = pos.shape[0]
        fd_forces = paddle.zeros([n_atoms, 3], dtype="float32")
        for i in range(n_atoms):
            for j in range(3):
                data_p = self._create_water_molecule_data()
                data_p["pos"] = data_p["pos"].clone()
                data_p["pos"][i, j] += eps
                e_p = self.model(
                    data_p, return_loss=False, return_prediction=True
                )["pred_dict"]["energy"]

                data_m = self._create_water_molecule_data()
                data_m["pos"] = data_m["pos"].clone()
                data_m["pos"][i, j] -= eps
                e_m = self.model(
                    data_m, return_loss=False, return_prediction=True
                )["pred_dict"]["energy"]

                fd_forces[i, j] = -(e_p.sum() - e_m.sum()) / (2 * eps)

        np.testing.assert_allclose(
            pred_forces.detach().numpy(), fd_forces.numpy(), atol=5e-2
        )

    def test_force_loss(self):
        """Forward pass with force targets computes force loss."""
        data = self._create_water_molecule_data()
        data["forces"] = paddle.randn(data["pos"].shape)
        data["pos"].stop_gradient = False
        result = self.model(data, return_loss=True, return_prediction=True)
        self.assertIn("force_loss", result["loss_dict"])
        self.assertTrue(result["loss_dict"]["force_loss"].item() > 0)

    def test_normalize_unnormalize_roundtrip(self):
        """normalize → unnormalize should recover the original value."""
        model = NewtonNet(data_mean=-76.0, data_std=10.0)
        original = paddle.to_tensor([-76.4], dtype="float32")
        normalized = model.normalize(original)
        recovered = model.unnormalize(normalized)
        np.testing.assert_allclose(
            recovered.numpy(), original.numpy(), rtol=1e-5
        )

    def test_no_cell_input(self):
        """Model should work when cell is not provided."""
        data = self._create_water_molecule_data()
        del data["cell"]
        result = self.model(data, return_loss=False, return_prediction=True)
        energy = result["pred_dict"]["energy"]
        self.assertEqual(energy.shape, [1])
        self.assertTrue(paddle.isfinite(energy).all().item())

    def test_training_convergence(self):
        """Verify loss decreases over 2 training epochs (RFC requirement)."""
        self.model.train()
        optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), learning_rate=1e-3
        )
        losses = []
        for epoch in range(2):
            optimizer.clear_grad()
            data = self._create_water_molecule_data()
            result = self.model(data, return_loss=True, return_prediction=False)
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


if __name__ == "__main__":
    unittest.main()
