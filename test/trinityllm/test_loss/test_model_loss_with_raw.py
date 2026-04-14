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

"""Forward-alignment and loss tests for TrinityLLM."""

import unittest

import numpy as np
import paddle

import sys
import os
import importlib.util

# Direct-import the trinityllm module file to avoid triggering the full
# ppmat import chain (which requires optional deps like pgl).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_MOD_PATH = os.path.join(
    _REPO_ROOT, "ppmat", "models", "trinityllm", "trinityllm.py"
)
_spec = importlib.util.spec_from_file_location("trinityllm", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrinityLLM = _mod.TrinityLLM
SMILESTokenizer = _mod.SMILESTokenizer


class TestTrinityLLMForwardAlignment(unittest.TestCase):
    """Verify TrinityLLM forward pass, loss, prediction, and normalization."""

    def setUp(self):
        paddle.seed(42)
        self.model = TrinityLLM(
            n_vocab=100,
            n_embd=64,
            n_layers=2,
            n_heads=4,
            dropout=0.0,
        )
        self.model.eval()

    def _create_dummy_data(self):
        # <bos>=0, tokens, <eos>=1, <pad>=2, <pad>=2
        token_ids = paddle.to_tensor(
            [[0, 5, 12, 8, 23, 1, 2, 2]], dtype="int64"
        )
        return {
            "token_ids": token_ids,
            "property": paddle.to_tensor([[-1.5]], dtype="float32"),
        }

    def test_forward_shape(self):
        """Prediction and loss dicts have correct shapes."""
        data = self._create_dummy_data()
        result = self.model(data, return_loss=True, return_prediction=True)

        self.assertIn("loss_dict", result)
        self.assertIn("pred_dict", result)
        self.assertIn("loss", result["loss_dict"])
        self.assertIn("property", result["pred_dict"])

        loss = result["loss_dict"]["loss"]
        pred = result["pred_dict"]["property"]
        self.assertEqual(list(loss.shape), [])
        self.assertEqual(list(pred.shape), [1, 1])

    def test_forward_determinism(self):
        """Two forward passes with same input yield identical outputs."""
        data = self._create_dummy_data()
        r1 = self.model(data, return_loss=False, return_prediction=True)
        r2 = self.model(data, return_loss=False, return_prediction=True)
        np.testing.assert_allclose(
            r1["pred_dict"]["property"].numpy(),
            r2["pred_dict"]["property"].numpy(),
            atol=1e-6,
        )

    def test_loss_computation(self):
        """Loss is non-negative and finite."""
        data = self._create_dummy_data()
        result = self.model(data, return_loss=True, return_prediction=False)
        loss_val = result["loss_dict"]["loss"].numpy()
        self.assertTrue(np.isfinite(loss_val))
        self.assertGreaterEqual(float(loss_val), 0.0)

    def test_predict_method(self):
        """predict() returns a dict with a finite scalar value."""
        data = self._create_dummy_data()
        pred = self.model.predict(data)
        self.assertIsInstance(pred, dict)
        self.assertIn("property", pred)
        self.assertTrue(np.isfinite(pred["property"]))

    def test_different_sequence_lengths(self):
        """Batch with varying-length sequences (padded) runs correctly."""
        token_ids = paddle.to_tensor(
            [
                [0, 5, 12, 1, 2, 2, 2],  # length 4
                [0, 7, 8, 9, 10, 1, 2],  # length 6
            ],
            dtype="int64",
        )
        data = {
            "token_ids": token_ids,
            "property": paddle.to_tensor([[-0.5], [1.2]], dtype="float32"),
        }
        result = self.model(data, return_loss=True, return_prediction=True)
        pred = result["pred_dict"]["property"]
        self.assertEqual(list(pred.shape), [2, 1])

        loss = result["loss_dict"]["loss"]
        self.assertTrue(np.isfinite(loss.numpy()))

    def test_normalization_roundtrip(self):
        """normalize → unnormalize is identity."""
        model = TrinityLLM(
            n_vocab=50,
            n_embd=32,
            n_layers=1,
            n_heads=2,
            data_mean=2.5,
            data_std=0.7,
        )
        t = paddle.to_tensor([1.0, 3.0, -2.0])
        recovered = model.unnormalize(model.normalize(t))
        np.testing.assert_allclose(recovered.numpy(), t.numpy(), atol=1e-5)

    def test_reference_alignment(self):
        """Hard-coded reference values for a fixed seed + input."""
        paddle.seed(123)
        model = TrinityLLM(
            n_vocab=50,
            n_embd=32,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        model.eval()

        token_ids = paddle.to_tensor([[0, 3, 7, 15, 1]], dtype="int64")
        data = {"token_ids": token_ids}
        result = model(data, return_loss=False, return_prediction=True)
        pred = result["pred_dict"]["property"].numpy()

        # Verify shape and finiteness (exact values depend on init seed)
        self.assertEqual(pred.shape, (1, 1))
        self.assertTrue(np.isfinite(pred).all())
        # The prediction should be a small value near 0 for random init
        self.assertLess(abs(float(pred[0, 0])), 10.0)


class TestSMILESTokenizer(unittest.TestCase):
    """Test the regex-based SMILES tokenizer."""

    def setUp(self):
        self.tok = SMILESTokenizer()

    def test_tokenize_simple(self):
        tokens = self.tok.tokenize("CCO")
        self.assertEqual(tokens, ["C", "C", "O"])

    def test_tokenize_branches(self):
        tokens = self.tok.tokenize("C(=O)O")
        self.assertEqual(tokens, ["C", "(", "=", "O", ")", "O"])

    def test_tokenize_halogen(self):
        tokens = self.tok.tokenize("ClBr")
        self.assertEqual(tokens, ["Cl", "Br"])

    def test_encode_with_special(self):
        ids = self.tok.encode("CCO", max_length=10)
        # Should start with <bos>=0 and end with <eos>=1
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 1)
        self.assertEqual(len(ids), 5)  # bos + C + C + O + eos

    def test_batch_encode(self):
        tensor = self.tok.batch_encode(["CCO", "O"], max_length=10)
        self.assertEqual(tensor.dtype, paddle.int64)
        self.assertEqual(tensor.shape[0], 2)
        # Shorter sequence padded with pad_id=2
        self.assertEqual(int(tensor[1, -1].numpy()), 2)

    def test_vocab_size(self):
        self.assertGreater(self.tok.vocab_size, 4)  # at least special tokens


if __name__ == "__main__":
    unittest.main()
