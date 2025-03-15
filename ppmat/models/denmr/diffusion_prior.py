import math

import numpy
import paddle
import paddle.nn as nn
from einops import rearrange
from einops import repeat
from einops.layers.paddle import Rearrange
from paddle.incubate.nn.functional import fused_rotary_position_embedding

from ppmat.models.common.sinusoidal_embedding import SinusoidalPosEmbeddings

from .layer import MLP
from .layer import Attention
from .layer import FeedForward
from .layer import LayerNorm
from .layer import RelPosBias
from .layer import prob_mask_like
from .utils.diffprior_utils import default
from .utils.diffprior_utils import exists
from .utils.diffprior_utils import first
from .utils.diffprior_utils import log


class DiffusionPriorNetwork(nn.Layer):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_graph_embeds=1,
        num_text_embeds=1,
        max_text_len=256,
        self_cond=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.num_graph_embeds = num_graph_embeds
        self.num_text_embeds = num_text_embeds
        self.to_text_embeds = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim * num_text_embeds)
            if num_text_embeds > 1
            else paddle.nn.Identity(),
            Rearrange("b (n d) -> b n d", n=num_text_embeds),
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
        self.max_text_len = max_text_len
        self.null_text_encodings = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, max_text_len, dim])
        )
        self.null_text_embeds = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, num_text_embeds, dim])
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
            *args, text_cond_drop_prob=1.0, graph_cond_drop_prob=1, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        graph_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings=None,
        self_cond=None,
        text_cond_drop_prob=0.0,
        graph_cond_drop_prob=0.0,
    ):
        batch, dim, dtype = (
            *tuple(graph_embed.shape),
            graph_embed.dtype,
        )

        # num_time_embeds, num_graph_embeds, num_text_embeds = (
        #     self.num_time_embeds,
        #     self.num_graph_embeds,
        #     self.num_text_embeds,
        # ) # TODO: check it from original dalle2 repo

        # setup self conditioning
        if self.self_cond:
            self_cond = default(
                self_cond, lambda: paddle.zeros(shape=[batch, self.dim], dtype=dtype)
            )
            self_cond = rearrange(self_cond, "b d -> b 1 d")

        # in section 2.2 of DALLE-2 paper, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep
        # embedding, noised CLIP image embedding, final embedding for prediction"
        text_embed = self.to_text_embeds(text_embed)
        graph_embed = self.to_graph_embeds(graph_embed)

        # classifier free guidance masks
        text_keep_mask = prob_mask_like((batch,), 1 - text_cond_drop_prob)
        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like((batch,), 1 - graph_cond_drop_prob)
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")
        if not exists(text_encodings):
            text_encodings = paddle.empty(shape=(batch, 0, dim), dtype=dtype)

        # make text encodings optional
        # although the paper seems to suggest it is present
        if not exists(text_encodings):
            text_encodings = paddle.empty(shape=(batch, 0, dim), dtype=dtype)

        if text_encodings.shape[1] == 0:
            mask = paddle.zeros(shape=(batch, 0), dtype=bool)
        else:
            mask = paddle.any(x=text_encodings != 0.0, axis=-1)

        # replace any padding in the text encodings with learned
        # padding tokens unique across position
        text_encodings = text_encodings[:, : self.max_text_len]
        mask = mask[:, : self.max_text_len]

        text_len = tuple(text_encodings.shape)[-2]
        remainder = self.max_text_len - text_len

        if remainder > 0:
            text_encodings = nn.functional.pad(
                x=text_encodings,
                pad=(0, 0, 0, remainder),
                value=0.0,
                pad_from_left_axis=False,
            )
            mask = mask.astype(paddle.int32)
            mask = nn.functional.pad(
                x=mask, pad=(0, remainder), value=0, pad_from_left_axis=False
            ).astype("bool")

        # mask out text encodings with null encodings
        null_text_encodings = self.null_text_encodings.to(text_encodings.dtype)

        text_encodings = paddle.where(
            condition=rearrange(mask, "b n -> b n 1").clone() & text_keep_mask,
            x=text_encodings,
            y=null_text_encodings,
        )

        # mask out text embeddings with null text embeddings
        null_text_embeds = self.null_text_embeds.to(text_embed.dtype)

        text_embed = paddle.where(
            condition=text_keep_mask, x=text_embed, y=null_text_embeds
        )

        # mask out image embeddings with null image embeddings
        null_graph_embed = self.null_graph_embed.to(graph_embed.dtype)

        graph_embed = paddle.where(
            condition=image_keep_mask, x=graph_embed, y=null_graph_embed
        )

        # whether text embedding is used for conditioning depends on whether text
        # encodings are available for attention (for classifier free guidance,
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
            x=(text_encodings, text_embed, time_embed, graph_embed, learned_queries),
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


class NoiseScheduler(paddle.nn.Layer):
    def __init__(
        self,
        *,
        beta_schedule,
        timesteps,
        loss_type,
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / paddle.linspace(start=timesteps, stop=1, num=timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()
        alphas = 1.0 - betas
        alphas_cumprod = paddle.cumprod(alphas, dim=0)
        alphas_cumprod_prev = paddle.nn.functional.pad(
            x=alphas_cumprod[:-1], pad=(1, 0), value=1.0, pad_from_left_axis=False
        )
        (timesteps,) = tuple(betas.shape)
        self.num_timesteps = int(timesteps)
        if loss_type == "l1":
            loss_fn = paddle.nn.functional.l1_loss
        elif loss_type == "l2":
            loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "huber":
            loss_fn = paddle.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError()
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        register_buffer = lambda name, val: self.register_buffer(  # noqa
            name=name, tensor=val.to("float32")
        )
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", paddle.sqrt(x=alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", paddle.sqrt(x=1.0 - alphas_cumprod)
        )
        register_buffer(
            "log_one_minus_alphas_cumprod", paddle.log(x=1.0 - alphas_cumprod)
        )
        register_buffer(
            "sqrt_recip_alphas_cumprod", paddle.sqrt(x=1.0 / alphas_cumprod)
        )
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", paddle.sqrt(x=1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            paddle.log(x=posterior_variance.clip(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * paddle.sqrt(x=alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * paddle.sqrt(x=alphas)
            / (1.0 - alphas_cumprod),
        )
        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def sample_random_times(self, batch):
        return paddle.randint(
            low=0, high=self.num_timesteps, shape=(batch,), dtype="int64"
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, tuple(x_t.shape)) * x_start
            + extract(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, tuple(x_t.shape))
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, tuple(x_t.shape)
        )
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def q_sample(self, x_start, t, noise=None):
        noise = default(
            noise, lambda: paddle.randn(shape=x_start.shape, dtype=x_start.dtype)
        )
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape))
            * noise
        )

    def calculate_v(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape))
            * x_start
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = tuple(x_from.shape)
        noise = default(
            noise, lambda: paddle.randn(shape=x_from.shape, dtype=x_from.dtype)
        )
        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)
        return (
            x_from * (alpha_next / alpha)
            + noise * (sigma_next * alpha - sigma * alpha_next) / alpha
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_t.shape)) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape))

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, tuple(loss.shape))


def extract(a, t, x_shape):
    b, *_ = tuple(t.shape)
    out = a.take_along_axis(axis=-1, indices=t, broadcast=False)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def meanflat(x):
    return x.mean(axis=tuple(range(1, len(tuple(x.shape)))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(x=logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * paddle.exp(x=-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0
        + paddle.nn.functional.tanh(x=(2.0 / math.pi) ** 0.5 * (x + 0.044715 * x**3))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert tuple(x.shape) == tuple(means.shape) == tuple(log_scales.shape)
    eps = 1e-12 if x.dtype == "float32" else 0.001
    centered_x = x - means
    inv_stdv = paddle.exp(x=-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps=eps)
    log_one_minus_cdf_min = log(1.0 - cdf_min, eps=eps)
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        condition=x < -thres,
        x=log_cdf_plus,
        y=paddle.where(
            condition=x > thres, x=log_one_minus_cdf_min, y=log(cdf_delta, eps=eps)
        ),
    )
    return log_probs


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = paddle.linspace(start=0, stop=timesteps, num=steps, dtype="float64")
    alphas_cumprod = paddle.cos(x=(x / timesteps + s) / (1 + s) * numpy.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return paddle.clip(x=betas, min=0, max=0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return paddle.linspace(
        start=beta_start, stop=beta_end, num=timesteps, dtype="float64"
    )


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        paddle.linspace(
            start=beta_start**0.5,
            stop=beta_end**0.5,
            num=timesteps,
            dtype="float64",
        )
        ** 2
    )


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = paddle.linspace(start=-6, stop=6, num=timesteps, dtype="float64")
    return paddle.nn.functional.sigmoid(x=betas) * (beta_end - beta_start) + beta_start


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.stop_gradient = not requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)


def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)
