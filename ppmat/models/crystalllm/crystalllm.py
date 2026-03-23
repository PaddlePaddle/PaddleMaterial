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
CrystalLLM: Crystal Structure Generation with Autoregressive Large Language Modeling.

Ported from lantunes/CrystaLLM (MIT License).
Reference: Antunes et al., Nature Communications, 2024.
DOI: 10.1038/s41467-024-54639-7
"""

import math
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration for the CrystalLLM GPT model."""

    block_size: int = 1024
    vocab_size: int = 371
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Layer):
    """LayerNorm with optional bias (Paddle's built-in always includes bias)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[ndim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        if bias:
            self.bias = paddle.create_parameter(
                shape=[ndim],
                dtype="float32",
                default_initializer=nn.initializer.Constant(0.0),
            )
        else:
            self.bias = None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)


def gelu(x):
    """Exact GELU activation (not approximate)."""
    return (
        0.5
        * x
        * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    )


class CausalSelfAttention(nn.Layer):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias_attr=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias_attr=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        # causal mask
        self.register_buffer(
            "causal_mask",
            paddle.tril(paddle.ones([config.block_size, config.block_size])).reshape(
                [1, 1, config.block_size, config.block_size]
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        # compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = paddle.split(qkv, 3, axis=2)

        # reshape to (B, n_head, T, head_dim)
        q = q.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])

        # manual attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        att = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
        att = att + (1.0 - self.causal_mask[:, :, :T, :T]) * (-1e9)
        att = F.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = paddle.matmul(att, v)

        # reshape back to (B, T, C)
        y = y.transpose([0, 2, 1, 3]).reshape([B, T, C])
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Layer):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias_attr=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias_attr=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Layer):
    """Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrystalLLM(nn.Layer):
    """
    CrystalLLM: GPT-based autoregressive model for crystal structure generation.

    Follows the ppmat model interface:
    - forward(data) -> {"loss_dict": {...}, "pred_dict": {...}}
    - _forward(data) -> logits tensor
    - generate(idx, max_new_tokens, ...) -> generated token indices
    """

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 371,
        n_layer: int = 8,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )

        # token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # transformer blocks
        self.h = nn.LayerList([Block(self.config) for _ in range(n_layer)])

        # final layer norm
        self.ln_f = LayerNorm(n_embd, bias=bias)

        # language model head — NOT nn.Linear because Paddle Linear stores
        # weight as [in, out] while Embedding stores as [vocab, embd].
        # For weight tying we compute x @ wte.weight^T directly in _forward.
        # This avoids shape mismatch and saves parameters.

        # init weights
        self.apply(self._init_weights)
        # apply special scaled init to residual projections (c_proj)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                with paddle.no_grad():
                    nn.initializer.Normal(mean=0.0, std=0.02 / math.sqrt(2 * n_layer))(
                        p
                    )

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.initializer.Normal(mean=0.0, std=0.02)(layer.weight)
            if layer.bias is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.initializer.Normal(mean=0.0, std=0.02)(layer.weight)

    def _forward(self, idx: paddle.Tensor) -> paddle.Tensor:
        """Core forward pass: token indices -> logits (no loss).

        Args:
            idx: (B, T) int64 tensor of token indices.

        Returns:
            logits: (B, T, vocab_size) float32 tensor.
        """
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = paddle.arange(0, T, dtype="int64")
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        # Weight-tied LM head: logits = x @ wte.weight^T
        logits = paddle.matmul(x, self.wte.weight, transpose_y=True)
        return logits

    def forward(
        self,
        data,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict:
        """Training forward pass following ppmat convention.

        Args:
            data: dict with "input_ids" (B, T) and optionally "target_ids" (B, T).
            return_loss: whether to compute and return loss.
            return_prediction: whether to return logits as predictions.

        Returns:
            dict with "loss_dict" and "pred_dict".
        """
        if isinstance(data, dict):
            idx = data["input_ids"]
            targets = data.get("target_ids", None)
        else:
            # fallback: assume data is a tuple (input_ids, target_ids)
            idx, targets = data[0], data[1] if len(data) > 1 else None

        logits = self._forward(idx)

        loss_dict = {}
        if return_loss and targets is not None:
            loss = F.cross_entropy(
                logits.reshape([-1, logits.shape[-1]]),
                targets.reshape([-1]),
                ignore_index=-1,
            )
            loss_dict["loss"] = loss

        pred_dict = {}
        if return_prediction:
            pred_dict["logits"] = logits

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @paddle.no_grad()
    def generate(
        self,
        idx: paddle.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token: Optional[int] = None,
    ) -> paddle.Tensor:
        """Autoregressive generation.

        Args:
            idx: (B, T) starting token indices.
            max_new_tokens: maximum tokens to generate.
            temperature: sampling temperature.
            top_k: if set, only sample from top-k tokens.
            stop_token: if set, stop when last two generated tokens both equal
                this value (double-token stop). Requires ≥2 new tokens before
                checking. Default ``None`` means generate full ``max_new_tokens``.

        Returns:
            (B, T + generated) tensor of token indices.
        """
        initial_len = idx.shape[1]
        for _ in range(max_new_tokens):
            # crop to block_size
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits = self._forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                topk_val, _ = paddle.topk(logits, k)
                threshold = topk_val[:, -1:]
                logits = paddle.where(
                    logits < threshold,
                    paddle.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, axis=-1)
            idx_next = paddle.multinomial(probs, num_samples=1)
            idx = paddle.concat([idx, idx_next], axis=1)

            # stop on double token (only after generating at least 2 new tokens)
            if stop_token is not None and idx.shape[1] >= initial_len + 2:
                if idx[0, -1].item() == stop_token and idx[0, -2].item() == stop_token:
                    break

        return idx

    def crop_block_size(self, block_size: int):
        """Reduce the block size (for fine-tuning on shorter sequences)."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # crop position embeddings
        self.wpe.weight = paddle.create_parameter(
            shape=[block_size, self.config.n_embd],
            dtype="float32",
            default_initializer=nn.initializer.Assign(self.wpe.weight[:block_size]),
        )
        # crop causal masks in attention blocks
        for block in self.h:
            if hasattr(block.attn, "causal_mask"):
                block.attn.causal_mask = block.attn.causal_mask[
                    :, :, :block_size, :block_size
                ]

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters.

        Args:
            non_embedding: if True, subtract position embeddings
                (token embeddings are shared with lm_head, so not subtracted).
        """
        n_params = sum(p.numel().item() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel().item()
        return n_params

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> paddle.optimizer.AdamW:
        """Configure AdamW optimizer with weight decay only on 2D params.

        This mirrors the original nanoGPT pattern: bias, LayerNorm, and
        Embedding parameters are excluded from weight decay.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.stop_gradient:
                if param.ndim >= 2 and "wte" not in name and "wpe" not in name:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            beta1=betas[0],
            beta2=betas[1],
            parameters=[
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            apply_decay_param_fun=lambda name: True,
        )
        return optimizer
