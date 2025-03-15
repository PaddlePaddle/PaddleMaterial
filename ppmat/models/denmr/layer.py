import math

import paddle
import paddle.nn as nn
from einops import rearrange
from einops import repeat

from .utils.diffprior_utils import exists
from .utils.diffprior_utils import l2norm


class Xtoy(nn.Layer):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * dx, out_features=dy)

    def forward(self, X, x_mask):
        """X: bs, n, dx."""
        x_mask = paddle.expand(x_mask, shape=[-1, -1, X.shape[-1]])
        float_imask = 1 - x_mask.astype("float32")
        m = paddle.sum(X, axis=1) / paddle.sum(x_mask, axis=1)
        mi = paddle.min(X + 1e6 * float_imask, axis=1)
        ma = paddle.max(X - 1e6 * float_imask, axis=1)
        std = paddle.sum((X - m.unsqueeze(1)) ** 2 * x_mask, axis=1) / paddle.sum(
            x_mask, axis=1
        )
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


class Etoy(nn.Layer):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * d, out_features=dy)

    def forward(self, E, e_mask1, e_mask2):
        """
        E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        mask = paddle.expand(e_mask1 * e_mask2, shape=[-1, -1, -1, E.shape[-1]])
        float_imask = 1 - mask.astype("float32")
        divide = paddle.sum(mask, axis=(1, 2))
        m = paddle.sum(E, axis=(1, 2)) / divide
        mi = paddle.min(paddle.min(E + 1e6 * float_imask, axis=2), axis=1)
        ma = paddle.max(paddle.max(E - 1e6 * float_imask, axis=2), axis=1)
        std = (
            paddle.sum((E - m.unsqueeze(1).unsqueeze(1)) ** 2 * mask, axis=(1, 2))
            / divide
        )
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, axis=-1):
    """
    Perform softmax over masked values in `x`.

    Args:
        x: Tensor, the input data.
        mask: Tensor, the binary mask of the same shape as `x`.
        axis: The axis to apply softmax.

    Returns:
        Tensor with masked softmax applied.
    """
    if paddle.sum(mask) == 0:
        return x

    # TODO: ndim check: only support adding dimensions backwards now
    x_dims = x.ndim
    mask_dims = mask.ndim
    if mask_dims < x_dims:
        diff = x_dims - mask_dims
        mask = paddle.unsqueeze(mask, axis=[-1] * diff)
        repeat_times = [1] * mask_dims + [x.shape[i] for i in range(mask_dims, x_dims)]
        mask = paddle.tile(mask, repeat_times=repeat_times)

    x_masked = x.clone()
    x_masked = paddle.where(
        mask == 0, paddle.to_tensor(-float("inf"), dtype=x.dtype), x_masked
    )

    return paddle.nn.functional.softmax(x_masked, axis=axis)


class MLP(paddle.nn.Layer):
    def __init__(self, dim_in, dim_out, *, expansion_factor=2.0, depth=2, norm=False):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = (  # noqa
            lambda: paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            if norm
            else paddle.nn.Identity()
        )
        layers = [
            paddle.nn.Sequential(
                paddle.nn.Linear(in_features=dim_in, out_features=hidden_dim),
                paddle.nn.Silu(),
                norm_fn(),
            )
        ]
        for _ in range(depth - 1):
            layers.append(
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    paddle.nn.Silu(),
                    norm_fn(),
                )
            )
        layers.append(paddle.nn.Linear(in_features=hidden_dim, out_features=dim_out))
        self.net = paddle.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.astype(dtype="float32"))


class LayerNorm(paddle.nn.Layer):
    def __init__(self, dim, eps=1e-05, fp16_eps=0.001, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=dim)
        )

    def forward(self, x):
        eps = self.eps if x.dtype == "float32" else self.fp16_eps
        if self.stable:
            x = x / x.amax(axis=-1, keepdim=True).detach()
        var = paddle.var(x=x, axis=-1, unbiased=False, keepdim=True)
        mean = paddle.mean(x=x, axis=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class RelPosBias(paddle.nn.Layer):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = paddle.nn.Embedding(
            num_embeddings=num_buckets, embedding_dim=heads
        )

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = paddle.maximum(n, paddle.zeros_like(x=n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            paddle.log(x=n.astype(dtype="float32") / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(dtype="int64")
        val_if_large = paddle.min(
            paddle.stack(
                [
                    val_if_large,
                    paddle.full_like(x=val_if_large, fill_value=num_buckets - 1),
                ]
            ),
            axis=0,
        )
        return paddle.where(condition=is_small, x=n, y=val_if_large)

    def forward(self, i, j):
        q_pos = paddle.arange(dtype="int64", end=i)
        k_pos = paddle.arange(dtype="int64", end=j)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        rotary_emb=None,
        cosine_sim=True,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else dim_head**-0.5
        self.cosine_sim = cosine_sim
        self.heads = heads
        inner_dim = dim_head * heads
        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.null_kv = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[2, dim_head])
        )
        self.to_q = paddle.nn.Linear(
            in_features=dim, out_features=inner_dim, bias_attr=False
        )
        self.to_kv = paddle.nn.Linear(
            in_features=dim, out_features=dim_head * 2, bias_attr=False
        )
        self.rotary_emb = rotary_emb
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False),
            LayerNorm(dim),
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n = tuple(x.shape)[:2]  # 获取输入的batch_size和序列长度
        x = self.norm(x)  # 归一化

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(
            chunks=2, axis=-1
        )  # q linear mapping; generate concatenated representation of kv,
        # split evenly into k and v along the -1 dimension

        # Multi-head splitting and scaling
        q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
        q = q * self.scale  # 有助于数值稳定
        k = rearrange(k, "b n (h d) -> b n h d", h=1)

        # Apply rotary position encoding
        if exists(self.rotary_emb):
            q, k, _ = self.rotary_emb(q, k)
        q = rearrange(q, "b n h d -> b h n d", h=self.heads)
        k = rearrange(k, "b n h d -> b n (h d)", h=1)

        # Add empty key-value kv
        nk, nv = map(
            lambda t: repeat(t, "d -> b 1 d", b=b), self.null_kv.unbind(axis=-2)
        )
        k = paddle.concat(x=(nk, k), axis=-2)
        v = paddle.concat(x=(nv, v), axis=-2)

        # Optional cosine similarity normalization
        if self.cosine_sim:
            q, k = map(l2norm, (q, k))  # Normalize their lengths to 1,
            # and the attention score computation becomes cosine similarity

        # Quadratic scaling
        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # Compute similarity matrix
        sim = paddle.einsum("b h i d, b j d -> b h i j", q, k)
        # i represents the position in the query sequence, j represents the
        # position in the key sequence

        # Add attention bias
        if exists(attn_bias):
            sim = sim + attn_bias  # 调整注意力分数

        # Masking processing
        max_neg_value = -paddle.finfo(dtype=sim.dtype).max
        if exists(mask):
            mask = paddle.nn.functional.pad(
                x=mask, pad=(1, 0), value=True, pad_from_left_axis=False
            )
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(mask=~mask, value=max_neg_value)

        # Causal masking processing
        if self.causal:
            i, j = tuple(sim.shape)[-2:]
            causal_mask = paddle.ones(shape=(i, j), dtype="bool").triu(
                diagonal=j - i + 1
            )
            sim = sim.masked_fill(mask=causal_mask, value=max_neg_value)

        # Compute attention weights and apply Dropout
        attn = paddle.nn.functional.softmax(sim, axis=-1, dtype="float32")
        attn = attn.astype(sim.dtype)
        attn = self.dropout(attn)

        # Compute attention output
        out = paddle.einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def FeedForward(dim, mult=4, dropout=0.0, post_activation_norm=False):
    """post-activation norm https://arxiv.org/abs/2110.09456"""
    inner_dim = int(mult * dim)
    return paddle.nn.Sequential(
        LayerNorm(dim),
        paddle.nn.Linear(in_features=dim, out_features=inner_dim * 2, bias_attr=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else paddle.nn.Identity(),
        paddle.nn.Dropout(p=dropout),
        paddle.nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False),
    )


class SwiGLU(paddle.nn.Layer):
    """used successfully in https://arxiv.org/abs/2204.0231"""

    def forward(self, x):
        x, gate = x.chunk(chunks=2, axis=-1)
        return x * paddle.nn.functional.silu(x=gate)


def prob_mask_like(shape, prob):
    if prob == 1:
        return paddle.ones(shape=shape, dtype="bool")
    elif prob == 0:
        return paddle.zeros(shape=shape, dtype="bool")
    else:
        return (
            paddle.zeros(shape=shape).astype(dtype="float32").uniform_(min=0, max=1)
            < prob
        )
