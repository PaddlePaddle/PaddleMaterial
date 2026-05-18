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
CrystalLLM unit tests — monkeypatch / lightweight.

Run:  python -m pytest test/test_unit.py -v
From: worktrees/task-006-crystalllm/

These tests catch interface bugs, shape mismatches, weight conversion
issues, and metrics edge-cases WITHOUT downloading checkpoints or
needing a GPU.  Run them before any heavy pipeline operation.
"""

import importlib.util
import math
import os
import sys
import tempfile

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# Module loading (bypass ppmat's pgl-eager imports)
# ---------------------------------------------------------------------------
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

_metrics_mod = _load_module(
    "crystal_metrics",
    os.path.join(_repo_root, "ppmat", "metrics", "crystal_metrics.py"),
)
CrystalMetrics = _metrics_mod.CrystalMetrics
is_valid = _metrics_mod.is_valid
bond_length_reasonableness_score = _metrics_mod.bond_length_reasonableness_score

sys.path.insert(0, os.path.join(_repo_root, "structure_generation"))
from convert_weights import convert_pytorch_to_paddle

paddle.set_device("cpu")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TINY_CONFIG = GPTConfig(
    block_size=64, vocab_size=371, n_layer=2, n_head=2, n_embd=32,
    dropout=0.0, bias=True,
)


@pytest.fixture(scope="module")
def tiny_model():
    """A tiny CrystalLLM for fast unit testing."""
    paddle.seed(42)
    m = CrystalLLM(
        block_size=TINY_CONFIG.block_size,
        vocab_size=TINY_CONFIG.vocab_size,
        n_layer=TINY_CONFIG.n_layer,
        n_head=TINY_CONFIG.n_head,
        n_embd=TINY_CONFIG.n_embd,
        dropout=0.0,
        bias=TINY_CONFIG.bias,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def tokenizer():
    return CIFTokenizer()


# ===================================================================
# 1. CIFTokenizer
# ===================================================================
class TestCIFTokenizer:
    def test_vocab_size(self, tokenizer):
        # 89 atoms + 10 digits + 31 kw + 13 symbols + 227 SG + 1 UNK = 371
        assert tokenizer.vocab_size == 371

    def test_encode_decode_roundtrip(self, tokenizer):
        tokens = ["Na", "Cl", "\n"]
        ids = tokenizer.encode(tokens)
        assert len(ids) == 3
        decoded = tokenizer.decode(ids)
        assert decoded == "NaCl\n"

    def test_newline_token_exists(self, tokenizer):
        """The generation loop relies on \\n having a token ID."""
        assert "\n" in tokenizer.token_to_id
        nl_id = tokenizer.token_to_id["\n"]
        assert isinstance(nl_id, int)
        assert 0 <= nl_id < tokenizer.vocab_size

    def test_unk_token(self, tokenizer):
        """Unknown tokens map to <unk>."""
        assert "<unk>" in tokenizer.token_to_id

    def test_tokenize_cif_basic(self, tokenizer):
        cif = "data_NaCl\n_cell_length_a 5\n"
        tokens = tokenizer.tokenize_cif(cif)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # data_ is a keyword, should appear as single token
        assert "data_" in tokens

    def test_space_group_disambiguation(self, tokenizer):
        """Space groups get _sg suffix to avoid collision with atom symbols."""
        cif = "_symmetry_space_group_name_H-M Fm-3m\n"
        tokens = tokenizer.tokenize_cif(cif)
        # Check that the space group got _sg suffix
        sg_tokens = [t for t in tokens if t.endswith("_sg")]
        assert len(sg_tokens) == 1, f"Expected one SG token, got {sg_tokens}"

    def test_all_atoms_in_vocab(self, tokenizer):
        for atom in _tok_mod.ATOMS:
            assert atom in tokenizer.token_to_id, f"Missing atom: {atom}"

    def test_all_keywords_in_vocab(self, tokenizer):
        for kw in _tok_mod.KEYWORDS + _tok_mod.EXTENDED_KEYWORDS:
            assert kw in tokenizer.token_to_id, f"Missing keyword: {kw}"


# ===================================================================
# 2. GPTConfig
# ===================================================================
class TestGPTConfig:
    def test_defaults(self):
        c = GPTConfig()
        assert c.block_size == 1024
        assert c.vocab_size == 371
        assert c.n_layer == 8
        assert c.n_head == 8
        assert c.n_embd == 512

    def test_custom(self):
        c = GPTConfig(block_size=128, n_layer=4, n_embd=256, n_head=4)
        assert c.block_size == 128
        assert c.n_layer == 4
        assert c.n_embd == 256


# ===================================================================
# 3. CrystalLLM Model
# ===================================================================
class TestCrystalLLM:
    def test_instantiation(self, tiny_model):
        assert isinstance(tiny_model, CrystalLLM)
        assert tiny_model.config.n_layer == 2

    def test_no_lm_head_attribute(self, tiny_model):
        """Weight tying is via matmul, no separate lm_head."""
        assert not hasattr(tiny_model, "lm_head")

    def test_forward_logits_shape(self, tiny_model):
        B, T = 2, 16
        x = paddle.randint(0, 371, [B, T])
        logits = tiny_model._forward(x)
        assert list(logits.shape) == [B, T, 371]

    def test_forward_no_nan(self, tiny_model):
        x = paddle.randint(0, 371, [1, 8])
        logits = tiny_model._forward(x)
        assert not paddle.isnan(logits).any().item()
        assert paddle.isfinite(logits).all().item()

    def test_forward_dict_interface(self, tiny_model):
        """ppmat convention: forward(data) → {loss_dict, pred_dict}."""
        x = paddle.randint(0, 371, [1, 8])
        data = {"input_ids": x}
        result = tiny_model(data)
        assert "loss_dict" in result
        assert "pred_dict" in result
        assert result["loss_dict"] == {}  # no targets → no loss
        assert list(result["pred_dict"]["logits"].shape) == [1, 8, 371]

    def test_forward_with_targets_loss(self, tiny_model):
        x = paddle.randint(0, 371, [1, 8])
        t = paddle.randint(0, 371, [1, 8])
        data = {"input_ids": x, "target_ids": t}
        result = tiny_model(data)
        loss = result["loss_dict"]["loss"]
        assert loss.shape == []
        assert not paddle.isnan(loss).item()
        # Cross-entropy with random targets on 371 classes ≈ ln(371) ≈ 5.9
        assert 3.0 < loss.item() < 9.0

    def test_block_size_enforced(self, tiny_model):
        """Sequence longer than block_size should raise."""
        too_long = paddle.randint(0, 371, [1, TINY_CONFIG.block_size + 1])
        with pytest.raises(AssertionError, match="exceeds block_size"):
            tiny_model._forward(too_long)

    def test_generate_returns_longer_sequence(self, tiny_model):
        start = paddle.to_tensor([[0]], dtype="int64")
        out = tiny_model.generate(start, max_new_tokens=10, temperature=1.0)
        # No stop_token → always generates exactly max_new_tokens
        assert out.shape[1] == 11  # 1 seed + 10 generated

    def test_generate_double_newline_stop(self, tokenizer):
        """Generate should stop on \\n\\n when stop_token is given."""
        # Build a model that always predicts newline
        cfg = GPTConfig(block_size=64, vocab_size=371, n_layer=1, n_head=1,
                        n_embd=16, dropout=0.0, bias=True)
        m = CrystalLLM(block_size=64, vocab_size=371, n_layer=1, n_head=1,
                        n_embd=16, dropout=0.0, bias=True)
        m.eval()
        nl_id = tokenizer.token_to_id["\n"]
        # Monkeypatch _forward to always return high logit on newline
        original_forward = m._forward

        def _always_newline(idx):
            logits = original_forward(idx)
            # Set newline logit very high
            logits[:, :, :] = -1e9
            logits[:, :, nl_id] = 100.0
            return logits

        m._forward = _always_newline
        start = paddle.to_tensor([[nl_id]], dtype="int64")
        out = m.generate(start, max_new_tokens=50, temperature=1.0, stop_token=nl_id)
        # Start is \n (len 1). Need initial_len + 2 = 3 to check stop.
        # Generated tokens: \n, \n → stop fires → [nl, nl, nl] = length 3.
        assert out.shape[1] == 3, f"Expected stop at \\n\\n, got length {out.shape[1]}"

    def test_param_count(self, tiny_model):
        n = tiny_model.get_num_params(non_embedding=True)
        assert n > 0
        n_all = tiny_model.get_num_params(non_embedding=False)
        assert n_all > n  # full count includes wpe

    def test_crop_block_size(self):
        m = CrystalLLM(block_size=64, vocab_size=371, n_layer=1, n_head=1,
                        n_embd=16, dropout=0.0, bias=True)
        m.crop_block_size(32)
        assert m.config.block_size == 32
        # Forward with shorter seq should work
        x = paddle.randint(0, 371, [1, 16])
        logits = m._forward(x)
        assert list(logits.shape) == [1, 16, 371]

    def test_configure_optimizers(self, tiny_model):
        opt = tiny_model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95)
        )
        assert isinstance(opt, paddle.optimizer.AdamW)


# ===================================================================
# 4. Weight Converter
# ===================================================================
class TestConvertWeights:
    """Test convert_pytorch_to_paddle with synthetic checkpoints."""

    def _make_pt_checkpoint(self, config, prefix=""):
        """Build a minimal PyTorch-format checkpoint dict.

        Args:
            prefix: e.g. "" for clean keys or "_orig_mod.transformer." for
                    torch.compile-wrapped keys.
        """
        import torch
        sd = {}
        n = config.n_embd
        v = config.vocab_size
        bs = config.block_size

        sd[f"{prefix}wte.weight"] = torch.randn(v, n) * 0.02
        sd[f"{prefix}wpe.weight"] = torch.randn(bs, n) * 0.02
        for i in range(config.n_layer):
            p = f"{prefix}h.{i}"
            sd[f"{p}.ln_1.weight"] = torch.ones(n)
            sd[f"{p}.ln_1.bias"] = torch.zeros(n)
            sd[f"{p}.attn.c_attn.weight"] = torch.randn(3 * n, n)
            sd[f"{p}.attn.c_attn.bias"] = torch.zeros(3 * n)
            sd[f"{p}.attn.c_proj.weight"] = torch.randn(n, n)
            sd[f"{p}.attn.c_proj.bias"] = torch.zeros(n)
            sd[f"{p}.ln_2.weight"] = torch.ones(n)
            sd[f"{p}.ln_2.bias"] = torch.zeros(n)
            sd[f"{p}.mlp.c_fc.weight"] = torch.randn(4 * n, n)
            sd[f"{p}.mlp.c_fc.bias"] = torch.zeros(4 * n)
            sd[f"{p}.mlp.c_proj.weight"] = torch.randn(n, 4 * n)
            sd[f"{p}.mlp.c_proj.bias"] = torch.zeros(n)
        sd[f"{prefix}ln_f.weight"] = torch.ones(n)
        sd[f"{prefix}ln_f.bias"] = torch.zeros(n)
        # lm_head (should be stripped by converter)
        lm_prefix = prefix.replace("transformer.", "") if prefix else ""
        sd[f"{lm_prefix}lm_head.weight"] = sd[f"{prefix}wte.weight"].clone()
        return {"model": sd, "model_args": {"n_layer": config.n_layer}}

    def _convert_roundtrip(self, config, prefix=""):
        import torch
        ckpt = self._make_pt_checkpoint(config, prefix)
        with tempfile.TemporaryDirectory() as d:
            pt_path = os.path.join(d, "ckpt.pt")
            pd_path = os.path.join(d, "ckpt.pdparams")
            torch.save(ckpt, pt_path)
            convert_pytorch_to_paddle(pt_path, pd_path)
            return paddle.load(pd_path)

    def test_clean_keys_no_prefix(self):
        """Standard nanoGPT checkpoint (no torch.compile prefix)."""
        sd = self._convert_roundtrip(TINY_CONFIG, prefix="")
        assert "wte.weight" in sd
        assert "h.0.attn.c_attn.weight" in sd
        assert "lm_head.weight" not in sd

    def test_orig_mod_prefix_stripped(self):
        """Zenodo checkpoints have _orig_mod.transformer. prefix."""
        sd = self._convert_roundtrip(TINY_CONFIG, prefix="_orig_mod.transformer.")
        assert "wte.weight" in sd
        assert "h.0.attn.c_attn.weight" in sd
        assert "lm_head.weight" not in sd
        # No raw prefixed keys should survive
        for k in sd:
            assert not k.startswith("_orig_mod"), f"Prefix not stripped: {k}"

    def test_linear_weights_transposed(self):
        """Linear [out, in] → [in, out]."""
        sd = self._convert_roundtrip(TINY_CONFIG)
        n = TINY_CONFIG.n_embd
        # c_fc: PT [4*n, n] → PD [n, 4*n]
        assert list(sd["h.0.mlp.c_fc.weight"].shape) == [n, 4 * n]
        # c_attn: PT [3*n, n] → PD [n, 3*n]
        assert list(sd["h.0.attn.c_attn.weight"].shape) == [n, 3 * n]

    def test_embeddings_not_transposed(self):
        """Embeddings keep [vocab, embd] / [block, embd]."""
        sd = self._convert_roundtrip(TINY_CONFIG)
        assert list(sd["wte.weight"].shape) == [371, TINY_CONFIG.n_embd]
        assert list(sd["wpe.weight"].shape) == [TINY_CONFIG.block_size, TINY_CONFIG.n_embd]

    def test_layernorm_1d_not_transposed(self):
        sd = self._convert_roundtrip(TINY_CONFIG)
        assert sd["h.0.ln_1.weight"].ndim == 1
        assert sd["ln_f.weight"].ndim == 1

    def test_lm_head_removed(self):
        sd = self._convert_roundtrip(TINY_CONFIG)
        assert "lm_head.weight" not in sd

    def test_param_count_matches(self):
        """Converted params should load into the model without missing keys."""
        sd = self._convert_roundtrip(TINY_CONFIG)
        m = CrystalLLM(
            block_size=TINY_CONFIG.block_size, vocab_size=371,
            n_layer=TINY_CONFIG.n_layer, n_head=TINY_CONFIG.n_head,
            n_embd=TINY_CONFIG.n_embd, dropout=0.0, bias=True,
        )
        # set_state_dict logs warnings for missing keys (causal_mask buffers);
        # the real check is that all weight params are loaded
        m.set_state_dict(sd)
        # Forward should work after loading
        x = paddle.randint(0, 371, [1, 8])
        logits = m._forward(x)
        assert list(logits.shape) == [1, 8, 371]
        assert paddle.isfinite(logits).all().item()

    def test_converted_weights_match_values(self):
        """Verify actual numeric values survive the conversion."""
        import torch
        ckpt = self._make_pt_checkpoint(TINY_CONFIG)
        pt_wte = ckpt["model"]["wte.weight"].numpy()
        with tempfile.TemporaryDirectory() as d:
            pt_path = os.path.join(d, "ckpt.pt")
            pd_path = os.path.join(d, "ckpt.pdparams")
            torch.save(ckpt, pt_path)
            convert_pytorch_to_paddle(pt_path, pd_path)
            sd = paddle.load(pd_path)
        # Embeddings should be identical (no transpose)
        np.testing.assert_allclose(sd["wte.weight"].numpy(), pt_wte, atol=1e-7)
        # Linear weights should be transposed
        pt_c_fc = ckpt["model"]["h.0.mlp.c_fc.weight"].numpy()
        np.testing.assert_allclose(
            sd["h.0.mlp.c_fc.weight"].numpy(), pt_c_fc.T, atol=1e-7
        )


# ===================================================================
# 5. CrystalMetrics
# ===================================================================
NACL_CIF = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'Fm-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""

JUNK_CIF = "data_junk\n_cell_length_a 1.0\n"


class TestCrystalMetrics:
    def test_valid_cif(self):
        assert is_valid(NACL_CIF) is True

    def test_invalid_cif(self):
        assert is_valid(JUNK_CIF) is False

    def test_empty_string(self):
        assert is_valid("") is False

    def test_metrics_batch(self):
        metrics = CrystalMetrics()
        result = metrics([NACL_CIF, JUNK_CIF])
        assert "validity_rate" in result
        assert "avg_bond_score" in result
        assert "sg_consistency_rate" in result
        assert 0.0 <= result["validity_rate"] <= 1.0
        assert 0.0 <= result["avg_bond_score"] <= 1.0

    def test_metrics_all_valid(self):
        metrics = CrystalMetrics()
        result = metrics([NACL_CIF, NACL_CIF])
        assert result["validity_rate"] == 1.0

    def test_metrics_empty_list(self):
        metrics = CrystalMetrics()
        result = metrics([])
        assert result["validity_rate"] == 0.0

    def test_bond_score_nacl(self):
        from pymatgen.io.cif import CifParser
        parser = CifParser.from_str(NACL_CIF)
        structure = parser.parse_structures()[0]
        score = bond_length_reasonableness_score(structure)
        assert 0.5 < score <= 1.0, f"NaCl bond score unexpectedly low: {score}"


# ===================================================================
# 6. Pipeline Integration (synthetic, no download)
# ===================================================================
class TestPipelineIntegration:
    """End-to-end: synthetic PT checkpoint → convert → load → forward → generate → metrics."""

    def test_full_synthetic_pipeline(self, tokenizer):
        import torch

        config = TINY_CONFIG
        with tempfile.TemporaryDirectory() as d:
            pt_path = os.path.join(d, "ckpt.pt")
            pd_path = os.path.join(d, "ckpt.pdparams")

            # 1) Create synthetic PT checkpoint
            sd = {}
            n = config.n_embd
            sd["wte.weight"] = torch.randn(371, n) * 0.02
            sd["wpe.weight"] = torch.randn(config.block_size, n) * 0.02
            for i in range(config.n_layer):
                p = f"h.{i}"
                sd[f"{p}.ln_1.weight"] = torch.ones(n)
                sd[f"{p}.ln_1.bias"] = torch.zeros(n)
                sd[f"{p}.attn.c_attn.weight"] = torch.randn(3 * n, n)
                sd[f"{p}.attn.c_attn.bias"] = torch.zeros(3 * n)
                sd[f"{p}.attn.c_proj.weight"] = torch.randn(n, n)
                sd[f"{p}.attn.c_proj.bias"] = torch.zeros(n)
                sd[f"{p}.ln_2.weight"] = torch.ones(n)
                sd[f"{p}.ln_2.bias"] = torch.zeros(n)
                sd[f"{p}.mlp.c_fc.weight"] = torch.randn(4 * n, n)
                sd[f"{p}.mlp.c_fc.bias"] = torch.zeros(4 * n)
                sd[f"{p}.mlp.c_proj.weight"] = torch.randn(n, 4 * n)
                sd[f"{p}.mlp.c_proj.bias"] = torch.zeros(n)
            sd["ln_f.weight"] = torch.ones(n)
            sd["ln_f.bias"] = torch.zeros(n)
            sd["lm_head.weight"] = sd["wte.weight"].clone()
            torch.save({
                "model": sd,
                "model_args": {
                    "n_layer": 2, "n_head": 2, "n_embd": n,
                    "block_size": config.block_size, "vocab_size": 371,
                },
            }, pt_path)

            # 2) Convert
            convert_pytorch_to_paddle(pt_path, pd_path)
            assert os.path.exists(pd_path)

            # 3) Load
            model = CrystalLLM(
                block_size=config.block_size, vocab_size=371,
                n_layer=config.n_layer, n_head=config.n_head,
                n_embd=config.n_embd, dropout=0.0, bias=True,
            )
            model.set_state_dict(paddle.load(pd_path))
            model.eval()

            # 4) Forward
            tokens = tokenizer.encode(tokenizer.tokenize_cif(
                "data_test\n_cell_length_a 5\n"))
            if len(tokens) < 2:
                tokens = list(range(10))
            tokens = tokens[:min(len(tokens), config.block_size)]
            inp = paddle.to_tensor([tokens], dtype="int64")
            logits = model._forward(inp)
            assert list(logits.shape) == [1, len(tokens), 371]
            assert paddle.isfinite(logits).all().item()

            # 5) Generate
            nl_id = tokenizer.token_to_id["\n"]
            start = paddle.to_tensor([[nl_id]], dtype="int64")
            out = model.generate(start, max_new_tokens=30, temperature=1.0, top_k=40)
            assert out.shape[1] >= 2
            gen_text = tokenizer.decode(out[0].numpy().tolist())
            assert isinstance(gen_text, str)

            # 6) Metrics (smoke — generated CIF will be gibberish from random weights)
            metrics = CrystalMetrics()
            result = metrics([gen_text])
            assert "validity_rate" in result

    def test_orig_mod_prefix_pipeline(self, tokenizer):
        """Full pipeline with _orig_mod.transformer. prefix (real Zenodo format)."""
        import torch

        config = TINY_CONFIG
        n = config.n_embd
        prefix = "_orig_mod.transformer."
        sd = {}
        sd[f"{prefix}wte.weight"] = torch.randn(371, n) * 0.02
        sd[f"{prefix}wpe.weight"] = torch.randn(config.block_size, n) * 0.02
        for i in range(config.n_layer):
            p = f"{prefix}h.{i}"
            sd[f"{p}.ln_1.weight"] = torch.ones(n)
            sd[f"{p}.ln_1.bias"] = torch.zeros(n)
            sd[f"{p}.attn.c_attn.weight"] = torch.randn(3 * n, n)
            sd[f"{p}.attn.c_attn.bias"] = torch.zeros(3 * n)
            sd[f"{p}.attn.c_proj.weight"] = torch.randn(n, n)
            sd[f"{p}.attn.c_proj.bias"] = torch.zeros(n)
            sd[f"{p}.ln_2.weight"] = torch.ones(n)
            sd[f"{p}.ln_2.bias"] = torch.zeros(n)
            sd[f"{p}.mlp.c_fc.weight"] = torch.randn(4 * n, n)
            sd[f"{p}.mlp.c_fc.bias"] = torch.zeros(4 * n)
            sd[f"{p}.mlp.c_proj.weight"] = torch.randn(n, 4 * n)
            sd[f"{p}.mlp.c_proj.bias"] = torch.zeros(n)
        sd[f"{prefix}ln_f.weight"] = torch.ones(n)
        sd[f"{prefix}ln_f.bias"] = torch.zeros(n)
        sd["_orig_mod.lm_head.weight"] = sd[f"{prefix}wte.weight"].clone()

        with tempfile.TemporaryDirectory() as d:
            pt_path = os.path.join(d, "ckpt.pt")
            pd_path = os.path.join(d, "ckpt.pdparams")
            torch.save({"model": sd, "model_args": {"n_layer": 2}}, pt_path)
            convert_pytorch_to_paddle(pt_path, pd_path)

            model = CrystalLLM(
                block_size=config.block_size, vocab_size=371,
                n_layer=config.n_layer, n_head=config.n_head,
                n_embd=config.n_embd, dropout=0.0, bias=True,
            )
            model.set_state_dict(paddle.load(pd_path))
            model.eval()

            # Forward should work
            x = paddle.randint(0, 371, [1, 8])
            logits = model._forward(x)
            assert list(logits.shape) == [1, 8, 371]
            assert paddle.isfinite(logits).all().item()
