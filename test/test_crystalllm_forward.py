#!/usr/bin/env python3
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

"""
Forward alignment test for CrystalLLM Paddle implementation.

Tests:
1. Model instantiation (small config)
2. Forward pass shape correctness
3. Loss computation
4. Generate method
5. Tokenizer vocab size consistency
6. Parameter count validation
7. ppmat-interface compliance (loss_dict/pred_dict)
"""

import importlib.util
import os
import sys

import paddle

# Load modules directly from file to bypass ppmat/__init__.py which eagerly
# imports pgl-dependent subpackages that aren't available in all environments.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tok_mod = _load_module(
    "cif_tokenizer",
    os.path.join(_repo_root, "ppmat", "models", "crystalllm", "cif_tokenizer.py"),
)
CIFTokenizer = _tok_mod.CIFTokenizer

_model_mod = _load_module(
    "crystalllm",
    os.path.join(_repo_root, "ppmat", "models", "crystalllm", "crystalllm.py"),
)
CrystalLLM = _model_mod.CrystalLLM
GPTConfig = _model_mod.GPTConfig


def test_tokenizer():
    """Test CIFTokenizer initialization and vocab size."""
    print("=" * 60)
    print("Test 1: CIFTokenizer")
    tok = CIFTokenizer()
    assert tok.vocab_size == 371, f"Expected vocab_size=371, got {tok.vocab_size}"
    # 89 atoms + 10 digits + 31 keywords + 13 symbols + 227 space groups + 1 UNK = 371

    # Test encode/decode roundtrip for simple tokens
    tokens = ["Si", "O", "\n"]
    ids = tok.encode(tokens)
    decoded = tok.decode(ids)
    assert decoded == "SiO\n", f"Roundtrip failed: {decoded!r}"
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  encode(['Si','O','\\n']): {ids}")
    print("  decode roundtrip: OK")
    print("  PASSED")


def test_model_instantiation():
    """Test model creation with small config."""
    print("=" * 60)
    print("Test 2: Model instantiation (small config)")
    model = CrystalLLM(
        block_size=1024,
        vocab_size=371,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=True,
    )
    n_params = model.get_num_params(non_embedding=True)
    print(f"  Parameters (non-embedding): {n_params:,}")
    # Small config should be ~33M params
    assert 25_000_000 < n_params < 50_000_000, f"Unexpected param count: {n_params}"
    print("  PASSED")
    return model


def test_forward_shape(model):
    """Test forward pass output shapes."""
    print("=" * 60)
    print("Test 3: Forward pass shape")
    B, T = 2, 64
    input_ids = paddle.randint(0, 371, [B, T])
    target_ids = paddle.randint(0, 371, [B, T])

    data = {"input_ids": input_ids, "target_ids": target_ids}
    result = model(data)

    assert "loss_dict" in result, "Missing loss_dict"
    assert "pred_dict" in result, "Missing pred_dict"
    assert "loss" in result["loss_dict"], "Missing loss in loss_dict"
    assert "logits" in result["pred_dict"], "Missing logits in pred_dict"

    loss = result["loss_dict"]["loss"]
    logits = result["pred_dict"]["logits"]

    assert logits.shape == [B, T, 371], f"Wrong logits shape: {logits.shape}"
    assert loss.shape == [], f"Loss should be scalar, got {loss.shape}"
    assert not paddle.isnan(loss).item(), "Loss is NaN"

    print(f"  logits shape: {logits.shape}")
    print(f"  loss: {loss.item():.4f}")
    print("  PASSED")


def test_forward_no_targets(model):
    """Test forward without targets (inference mode)."""
    print("=" * 60)
    print("Test 4: Forward without targets")
    B, T = 2, 32
    input_ids = paddle.randint(0, 371, [B, T])

    data = {"input_ids": input_ids}
    result = model(data)

    assert (
        result["loss_dict"] == {}
    ), f"Expected empty loss_dict, got {result['loss_dict']}"
    assert "logits" in result["pred_dict"]
    logits = result["pred_dict"]["logits"]
    assert logits.shape == [B, T, 371]
    print(f"  logits shape: {logits.shape}")
    print("  PASSED")


def test_generate(model):
    """Test autoregressive generation."""
    print("=" * 60)
    print("Test 5: Generate")
    # Start with a newline token (token ID for '\n')
    tok = CIFTokenizer()
    newline_id = tok.token_to_id["\n"]
    idx = paddle.to_tensor([[newline_id]], dtype="int64")

    generated = model.generate(idx, max_new_tokens=20, temperature=1.0, top_k=10)
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 2  # at least start + 1 generated token
    assert generated.shape[1] <= 21  # at most start + 20

    print(f"  Input length: 1, Output length: {generated.shape[1]}")
    # Decode the generated tokens
    gen_ids = generated[0].numpy().tolist()
    decoded = tok.decode(gen_ids)
    print(f"  Generated text (first 100 chars): {decoded[:100]!r}")
    print("  PASSED")


def test_weight_tying(model):
    """Test that lm_head uses wte embedding weights (via matmul in _forward)."""
    print("=" * 60)
    print("Test 6: Weight tying (implicit via matmul)")
    # The model uses paddle.matmul(x, wte.weight^T) as the LM head,
    # so there is no separate lm_head parameter — weights are tied by construction.
    assert not hasattr(
        model, "lm_head"
    ), "lm_head should not exist (weight tied via matmul)"
    wte_shape = model.wte.weight.shape
    assert wte_shape == [371, model.config.n_embd], f"wte shape: {wte_shape}"
    print(f"  wte.weight shape: {wte_shape} (used as LM head)")
    print("  PASSED")


def test_crop_block_size(model):
    """Test block size cropping."""
    print("=" * 60)
    print("Test 7: Crop block size")
    original_bs = model.config.block_size
    model.crop_block_size(512)
    assert model.config.block_size == 512

    # Forward pass with shorter sequence should work
    idx = paddle.randint(0, 371, [1, 256])
    logits = model._forward(idx)
    assert logits.shape == [1, 256, 371]

    print(f"  Cropped from {original_bs} to 512: OK")
    print("  Forward with seq_len=256: OK")
    print("  PASSED")


if __name__ == "__main__":
    paddle.set_device("cpu")
    paddle.seed(42)

    test_tokenizer()
    model = test_model_instantiation()
    test_forward_shape(model)
    test_forward_no_targets(model)
    test_generate(model)
    test_weight_tying(model)
    test_crop_block_size(model)

    print("=" * 60)
    print("ALL TESTS PASSED")
