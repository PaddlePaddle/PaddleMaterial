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

"""TrinityLLM: MoLFormer-based SMILES language model for polymer property prediction.

This module is adapted from
https://github.com/IBM/molformer (MoLFormer) and the TrinityLLM reference implementation.

The model tokenizes SMILES strings, encodes them through a Transformer
encoder with Rotary Position Embeddings (RoPE), and predicts scalar
molecular / polymer properties via a feedforward regression head.

Key design choices
------------------
* **Standard multi-head attention** with RoPE instead of linear attention
  (``fast_transformers`` dependency removed; linear attention can be added
  later for efficiency).
* Self-contained — no external dependencies beyond PaddlePaddle and NumPy.
* Follows PaddleMaterials conventions (``forward`` / ``predict`` /
  ``normalize`` / ``unnormalize`` protocol).
"""

import math
import re
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Layer):
    """Rotary Position Embedding (RoPE).

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.

    Args:
        dim (int): Dimension of each head (must be even).
        base (float): Base for the inverse-frequency spectrum.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (paddle.arange(0, dim, 2).astype("float32") / dim)
        )
        self.register_buffer(tensor=inv_freq, name="inv_freq")

    def forward(self, x: paddle.Tensor, seq_dim: int = 1):
        """Compute cos/sin embeddings for a sequence length derived from *x*.

        Args:
            x: Input tensor whose ``seq_dim`` axis defines the length.
            seq_dim: Which axis carries the sequence dimension.

        Returns:
            Tuple ``(cos, sin)`` each of shape ``[1, L, 1, dim]``.
        """
        seq_len = x.shape[seq_dim]
        t = paddle.arange(seq_len).astype("float32")
        freqs = paddle.outer(t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        cos_emb = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, dim]
        sin_emb = emb.sin().unsqueeze(0).unsqueeze(2)
        return cos_emb, sin_emb


def _rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotate the last dimension by half — helper for RoPE."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
):
    """Apply RoPE to query and key tensors.

    Args:
        q: [B, L, H, D]
        k: [B, L, H, D]
        cos: [1, L, 1, D]
        sin: [1, L, 1, D]

    Returns:
        Tuple of rotated ``(q, k)``.
    """
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (
        _rotate_half(k) * sin
    )


# ---------------------------------------------------------------------------
# Multi-Head Attention with RoPE
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Layer):
    """Standard scaled dot-product multi-head attention with RoPE.

    Args:
        d_model (int): Total model dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout on attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5

    def forward(
        self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward pass.

        Args:
            x: [B, L, D]
            mask: [B, L] boolean mask (True = keep, False = pad).

        Returns:
            [B, L, D]
        """
        B, L, _ = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).reshape([B, L, H, D])
        k = self.k_proj(x).reshape([B, L, H, D])
        v = self.v_proj(x).reshape([B, L, H, D])

        cos, sin = self.rope(x, seq_dim=1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose to [B, H, L, D]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale

        if mask is not None:
            # mask: [B, L] → [B, 1, 1, L]
            attn_mask = mask.unsqueeze(1).unsqueeze(2).astype("float32")
            attn = attn + (1.0 - attn_mask) * (-1e9)

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = paddle.matmul(attn, v)  # [B, H, L, D]
        out = out.transpose([0, 2, 1, 3]).reshape([B, L, H * D])
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Layer):
    """Pre-norm Transformer encoder layer with RoPE attention.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward inner dimension.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Pre-norm forward: LN → Attn → residual → LN → FFN → residual."""
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.drop1(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


class TransformerEncoder(nn.Layer):
    """Stack of :class:`TransformerEncoderLayer` blocks.

    Args:
        n_layers (int): Number of layers.
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward inner dimension.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.LayerList(
            [
                TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# SMILES Tokenizer
# ---------------------------------------------------------------------------


class SMILESTokenizer:
    """Regex-based SMILES tokenizer compatible with MoLFormer / TrinityLLM.

    The vocabulary is built from a fixed set of common SMILES tokens.  The
    four special tokens ``<bos>``, ``<eos>``, ``<pad>``, ``<mask>`` occupy
    indices 0–3.

    Args:
        extra_tokens: Optional list of additional tokens to add to the vocab.
    """

    PATTERN = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
        r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    SPECIAL_TOKENS = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<mask>": 3}

    # Common SMILES character-level tokens (sorted for reproducibility)
    _COMMON_TOKENS = sorted(
        set(
            list("CNOSPFIBrcnospb()=#-+/\\@.123456789")
            + ["Br", "Cl", "0", "[C@@H]", "[C@H]", "[nH]", "[N+]", "[O-]"]
        )
    )

    def __init__(self, extra_tokens: Optional[List[str]] = None):
        self.regex = re.compile(self.PATTERN)
        self.vocab = dict(self.SPECIAL_TOKENS)
        for tok in self._COMMON_TOKENS:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        if extra_tokens:
            for tok in extra_tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.SPECIAL_TOKENS["<pad>"]
        self.bos_id = self.SPECIAL_TOKENS["<bos>"]
        self.eos_id = self.SPECIAL_TOKENS["<eos>"]
        self.mask_id = self.SPECIAL_TOKENS["<mask>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into tokens."""
        return self.regex.findall(smiles)

    def encode(
        self,
        smiles: str,
        max_length: int = 512,
        add_special: bool = True,
    ) -> List[int]:
        """Encode a SMILES string to a list of token ids.

        Args:
            smiles: Input SMILES string.
            max_length: Maximum sequence length (including special tokens).
            add_special: Whether to prepend ``<bos>`` and append ``<eos>``.

        Returns:
            List of integer token ids (unpadded).
        """
        tokens = self.tokenize(smiles)
        if add_special:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        ids = [self.vocab.get(t, self.mask_id) for t in tokens]
        return ids[:max_length]

    def batch_encode(
        self,
        smiles_list: List[str],
        max_length: int = 512,
        add_special: bool = True,
    ) -> paddle.Tensor:
        """Encode a batch of SMILES and return a padded int64 tensor.

        Args:
            smiles_list: List of SMILES strings.
            max_length: Maximum sequence length.
            add_special: Whether to add ``<bos>`` / ``<eos>``.

        Returns:
            ``paddle.Tensor`` of shape ``[B, max_len]`` (``int64``).
        """
        encoded = [
            self.encode(s, max_length, add_special) for s in smiles_list
        ]
        max_len = max(len(e) for e in encoded)
        padded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return paddle.to_tensor(padded, dtype="int64")


# ---------------------------------------------------------------------------
# TrinityLLM Model
# ---------------------------------------------------------------------------


class TrinityLLM(nn.Layer):
    """TrinityLLM — MoLFormer-based SMILES model for property prediction.

    The architecture consists of:

    1. Token embedding (``nn.Embedding``).
    2. Transformer encoder with RoPE.
    3. Language-model head (for MLM pre-training, optional).
    4. Property-prediction head (FFN regressor with skip connections).

    Follows PaddleMaterials conventions (``forward`` / ``predict`` /
    ``normalize`` / ``unnormalize``).

    Args:
        n_vocab (int): Vocabulary size.
        n_embd (int): Embedding / model dimension.
        n_layers (int): Number of Transformer encoder layers.
        n_heads (int): Number of attention heads.
        ff_dim (int or None): FFN inner dimension (defaults to ``n_embd``).
        dropout (float): Dropout rate.
        max_length (int): Maximum supported sequence length.
        property_names (str or list): Target property name(s).
        data_mean (float): Mean for label normalization.
        data_std (float): Std for label normalization.
        loss_type (str): ``'l1_loss'`` or ``'mse_loss'``.
    """

    def __init__(
        self,
        n_vocab: int = 600,
        n_embd: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_length: int = 512,
        property_names: Union[str, List[str]] = "property",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "l1_loss",
        pad_id: int = 2,
    ):
        super().__init__()

        ff_dim = ff_dim or n_embd
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.max_length = max_length
        self.pad_id = pad_id

        # --- property / loss conventions (PaddleMaterials) ----------------
        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            assert isinstance(property_names, str)
            self.property_names = property_names

        self.register_buffer(
            tensor=paddle.to_tensor(data_mean, dtype="float32"),
            name="data_mean",
        )
        self.register_buffer(
            tensor=paddle.to_tensor(data_std, dtype="float32"),
            name="data_std",
        )

        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

        # --- model layers -------------------------------------------------
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.drop = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            d_model=n_embd,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Language-model head (MLM pre-training)
        self.lm_head = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_vocab, bias_attr=False),
        )

        # Property-prediction head (FFN with skip connections)
        self.property_head = PropertyHead(n_embd, dropout)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return tensor * self.data_std + self.data_mean

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _forward(self, data: dict) -> paddle.Tensor:
        """Core forward: tokens → embedding → transformer → mean pool → head.

        Args:
            data: Dict with at least ``'token_ids'`` key of shape ``[B, L]``.

        Returns:
            Predicted property values of shape ``[B, 1]``.
        """
        token_ids = data["token_ids"]  # [B, L]
        mask = (token_ids != self.pad_id)  # [B, L] True = real token

        x = self.tok_emb(token_ids)
        x = self.drop(x)
        x = self.encoder(x, mask)

        # Mean pool over non-padding positions
        mask_f = mask.unsqueeze(-1).astype("float32")  # [B, L, 1]
        x = (x * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1.0)

        return self.property_head(x)  # [B, 1]

    # ------------------------------------------------------------------
    # PaddleMaterials standard forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data: dict,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict:
        """Training / evaluation forward pass.

        Args:
            data: Dict with ``'token_ids'`` and optionally the property label.
            return_loss: Compute and return the loss.
            return_prediction: Return the (un-normalized) prediction.

        Returns:
            ``{"loss_dict": {...}, "pred_dict": {...}}``
        """
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(input=pred, label=label)
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_names] = pred

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @paddle.no_grad()
    def predict(self, data: dict) -> dict:
        """Inference helper — returns un-normalized predictions.

        Args:
            data: Dict with ``'token_ids'`` of shape ``[B, L]``.

        Returns:
            Dict ``{property_name: value}`` with scalar numpy value.
        """
        pred = self._forward(data)
        pred = self.unnormalize(pred).numpy()[0, 0]
        return {self.property_names: pred}


# ---------------------------------------------------------------------------
# Property prediction FFN (with skip connections, matching finetune reference)
# ---------------------------------------------------------------------------


class PropertyHead(nn.Layer):
    """Two-layer FFN with skip connections for scalar property regression.

    Architecture mirrors the reference finetune ``Net``:

    .. code-block:: text

        x → FC1 → GELU → Dropout → (+x) → FC2 → GELU → Dropout → (+prev) → Linear → out

    Args:
        dim (int): Input / hidden dimension.
        dropout (float): Dropout rate.
    """

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)
        self.act2 = nn.GELU()
        self.final = nn.Linear(dim, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h = self.act1(self.drop1(self.fc1(x)))
        h = h + x  # skip connection 1
        z = self.act2(self.drop2(self.fc2(h)))
        z = self.final(z + h)  # skip connection 2
        return z
