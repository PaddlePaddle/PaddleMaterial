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

import math

import paddle
import paddle.nn as nn
from einops import rearrange
from einops import repeat
from einops.layers.paddle import Rearrange
from paddle.incubate.nn.functional import fused_rotary_position_embedding

from ppmat.models.common.sinusoidal_embedding import SinusoidalPosEmbeddings
from ppmat.models.diffnmr.utils.diffprior_utils import default
from ppmat.models.diffnmr.utils.diffprior_utils import exists
from ppmat.models.diffnmr.utils.diffprior_utils import l2norm


class DiffPriorNetwork(nn.Layer):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_graph_embeds=1,
        num_spectrum_embeds=1,
        max_spectrum_len=256,
        self_cond=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.num_graph_embeds = num_graph_embeds
        self.num_spectrum_embeds = num_spectrum_embeds
        self.to_spectrum_embeds = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim * num_spectrum_embeds)
            if num_spectrum_embeds > 1
            else paddle.nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_spectrum_embeds),
        )
        self.continuous_embedded_time = not exists(num_timesteps)
        self.to_time_embeds = paddle.nn.Sequential(
            paddle.nn.Embedding(
                num_embeddings=num_timesteps, embedding_dim=dim * num_time_embeds
            )
            if exists(num_timesteps)
            else paddle.nn.Sequential(
                SinusoidalPosEmbeddings(dim), MLP(dim, dim * num_time_embeds)
            ),
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )
        self.to_graph_embeds = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim * num_graph_embeds)
            if num_graph_embeds > 1
            else paddle.nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_graph_embeds),
        )
        self.learned_query = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[dim])
        )
        self.causal_transformer = CausalTransformer(dim=dim, **kwargs)
        self.max_spectrum_len = max_spectrum_len
        self.null_spectrum_encodings = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, max_spectrum_len, dim])
        )
        self.null_spectrum_embeds = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, num_spectrum_embeds, dim])
        )
        self.null_graph_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, dim])
        )
        self.self_cond = self_cond

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, spectrum_cond_drop_prob=1.0, graph_cond_drop_prob=1, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        graph_embed,
        diffusion_timesteps,
        *,
        spectrum_embed,
        spectrum_encodings=None,
        self_cond=None,
        spectrum_cond_drop_prob=0.0,
        graph_cond_drop_prob=0.0,
    ):
        batch, dim, dtype = (
            *tuple(graph_embed.shape),
            graph_embed.dtype,
        )

        # num_time_embeds, num_graph_embeds, num_spectrum_embeds = (
        #     self.num_time_embeds,
        #     self.num_graph_embeds,
        #     self.num_spectrum_embeds,
        # ) # TODO: check it from original dalle2 repo

        # setup self conditioning
        if self.self_cond:
            self_cond = default(
                self_cond, lambda: paddle.zeros(shape=[batch, self.dim], dtype=dtype)
            )
            self_cond = rearrange(self_cond, "b d -> b 1 d")

        # in section 2.2 of DALLE-2 paper, last paragraph
        # "..consisting of encoded spectrum, CLIP spectrum embedding, diffusion timestep
        # embedding, noised CLIP image embedding, final embedding for prediction"
        spectrum_embed = self.to_spectrum_embeds(spectrum_embed)
        graph_embed = self.to_graph_embeds(graph_embed)

        # classifier free guidance masks
        spectrum_keep_mask = prob_mask_like((batch,), 1 - spectrum_cond_drop_prob)
        spectrum_keep_mask = rearrange(spectrum_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like((batch,), 1 - graph_cond_drop_prob)
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")
        if not exists(spectrum_encodings):
            spectrum_encodings = paddle.empty(shape=(batch, 0, dim), dtype=dtype)

        # make spectrum encodings optional
        # although the paper seems to suggest it is present
        if not exists(spectrum_encodings):
            spectrum_encodings = paddle.empty(shape=(batch, 0, dim), dtype=dtype)

        if spectrum_encodings.shape[1] == 0:
            mask = paddle.zeros(shape=(batch, 0), dtype=bool)
        else:
            mask = paddle.any(x=spectrum_encodings != 0.0, axis=-1)

        # replace any padding in the spectrum encodings with learned
        # padding tokens unique across position
        spectrum_encodings = spectrum_encodings[:, : self.max_spectrum_len]
        mask = mask[:, : self.max_spectrum_len]

        spectrum_len = tuple(spectrum_encodings.shape)[-2]
        remainder = self.max_spectrum_len - spectrum_len

        if remainder > 0:
            spectrum_encodings = nn.functional.pad(
                x=spectrum_encodings,
                pad=(0, 0, 0, remainder),
                value=0.0,
                pad_from_left_axis=False,
            )
            mask = mask.astype(paddle.int32)
            mask = nn.functional.pad(
                x=mask, pad=(0, remainder), value=0, pad_from_left_axis=False
            ).astype("bool")

        # mask out spectrum encodings with null encodings
        null_spectrum_encodings = self.null_spectrum_encodings.to(
            spectrum_encodings.dtype
        )

        spectrum_encodings = paddle.where(
            condition=rearrange(mask, "b n -> b n 1").clone() & spectrum_keep_mask,
            x=spectrum_encodings,
            y=null_spectrum_encodings,
        )

        # mask out spectrum embeddings with null spectrum embeddings
        null_spectrum_embeds = self.null_spectrum_embeds.to(spectrum_embed.dtype)

        spectrum_embed = paddle.where(
            condition=spectrum_keep_mask, x=spectrum_embed, y=null_spectrum_embeds
        )

        # mask out image embeddings with null image embeddings
        null_graph_embed = self.null_graph_embed.to(graph_embed.dtype)

        graph_embed = paddle.where(
            condition=image_keep_mask, x=graph_embed, y=null_graph_embed
        )

        # whether spectrum embedding is used for conditioning depends on whether
        # spectrum encodings are available for attention (for classifier free guidance,
        # even though it seems from the paper it was not used in the prior ddpm,
        # as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.astype(dtype)

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch)

        if self.self_cond:
            learned_queries = paddle.concat(x=(self_cond, learned_queries), axis=-2)

        tokens = paddle.concat(
            x=(
                spectrum_encodings,
                spectrum_embed,
                time_embed,
                graph_embed,
                learned_queries,
            ),
            axis=-2,
        )

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict image embedding (per DDPM timestep)
        pred_graph_embed = tokens[..., -1, :]

        return pred_graph_embed


class CausalTransformer(nn.Layer):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
    ):
        super().__init__()
        self.init_norm = (
            LayerNorm(dim) if norm_in else nn.Identity()
        )  # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads=heads)

        rotary_emb = fused_rotary_position_embedding if rotary_emb else None

        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList(
                    [
                        Attention(
                            dim=dim,
                            causal=True,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )

        self.norm = (
            LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        )  # unclear in paper whether they projected after the classic layer norm for
        # the final denoised image embedding, or just had the transformer
        # output it directly: plan on offering both options

        self.project_out = (
            nn.Linear(dim, dim, bias_attr=False) if final_proj else nn.Identity()
        )

    def forward(self, x):
        n = x.shape[1]

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


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
