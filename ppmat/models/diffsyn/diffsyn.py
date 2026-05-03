"""DiffSyn: Conditional 1D Diffusion for Zeolite Synthesis Conditions.

Adapted from https://github.com/eltonpan/zeosyn_gen (PyTorch) to PaddlePaddle.
Uses a 1D U-Net denoiser wrapped in a Gaussian diffusion process (DDPM/DDIM)
to generate synthesis conditions conditioned on zeolite/OSDA features.
"""

import math
from collections import namedtuple
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.9999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)


def extract(a, t, x_shape):
    """Gather values from *a* at indices *t*, then broadcast to *x_shape*."""
    b = t.shape[0]
    out = paddle.gather(a, t)
    return out.reshape([b] + [1] * (len(x_shape) - 1))


# ---------------------------------------------------------------------------
# Tiny building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Layer):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim, dtype="float32") * -emb)
        emb = x.unsqueeze(-1).astype("float32") * emb.unsqueeze(0)
        emb = paddle.concat([emb.sin(), emb.cos()], axis=-1)
        return emb


class LayerNorm1D(nn.Layer):
    """Channel-wise layer-norm for [B, C, L] tensors."""

    def __init__(self, dim):
        super().__init__()
        self.g = paddle.create_parameter(
            shape=[1, dim, 1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        eps = 1e-5 if x.dtype == paddle.float32 else 1e-3
        var = x.var(axis=1, keepdim=True)
        mean = x.mean(axis=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm1D(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ---------------------------------------------------------------------------
# Weight-standardised Conv1D
# ---------------------------------------------------------------------------

class WSConv1D(nn.Layer):
    """Conv1D with weight standardisation (synergistic with GroupNorm)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias_attr=True):
        super().__init__()
        self.conv = nn.Conv1D(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups,
            bias_attr=bias_attr,
        )

    def forward(self, x):
        eps = 1e-5 if x.dtype == paddle.float32 else 1e-3
        weight = self.conv.weight  # [out, in/groups, k]
        mean = weight.mean(axis=[1, 2], keepdim=True)
        var = weight.var(axis=[1, 2], keepdim=True)
        normed = (weight - mean) * (var + eps).rsqrt()
        return F.conv1d(x, normed, self.conv.bias, self.conv._stride,
                        self.conv._padding, self.conv._dilation, self.conv._groups)


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------

class Block1D(nn.Layer):
    """Conv1D → GroupNorm → SiLU, with optional FiLM scale-shift."""

    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = WSConv1D(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.Silu()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock1D(nn.Layer):
    """1D ResNet block with optional time + condition FiLM modulation."""

    def __init__(self, dim_in, dim_out, *, time_emb_dim=None,
                 cond_emb_dim=None, groups=8):
        super().__init__()
        mlp_dim = int(default(time_emb_dim, 0)) + int(default(cond_emb_dim, 0))
        self.mlp = (
            nn.Sequential(nn.Silu(), nn.Linear(mlp_dim, dim_out * 2))
            if mlp_dim > 0
            else None
        )
        self.block1 = Block1D(dim_in, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv1D(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None, cond_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(cond_emb)):
            parts = [e for e in (time_emb, cond_emb) if exists(e)]
            cond = paddle.concat(parts, axis=-1)
            cond = self.mlp(cond)
            cond = cond.unsqueeze(-1)  # [B, 2*dim_out, 1]
            scale_shift = cond.chunk(2, axis=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention1D(nn.Layer):
    """Efficient O(n) linear attention for 1D sequences."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1D(dim, hidden_dim * 3, 1, bias_attr=False)
        self.to_out = nn.Sequential(
            nn.Conv1D(hidden_dim, dim, 1),
            LayerNorm1D(dim),
        )

    def forward(self, x):
        b, c, n = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, axis=1)
        # reshape [B, hidden, N] → [B, heads, dim_head, N]
        q, k, v = [
            t.reshape([b, h, -1, n]) for t in qkv
        ]
        q = F.softmax(q, axis=-2)
        k = F.softmax(k, axis=-1)
        q = q * self.scale

        # linear attention: context = k^T v, out = context^T q
        context = paddle.matmul(k, v.transpose([0, 1, 3, 2]))  # [B,h,d,d]
        out = paddle.matmul(context, q)  # [B,h,d,N]
        out = out.reshape([b, -1, n])
        return self.to_out(out)


class Attention1D(nn.Layer):
    """Standard dot-product attention for 1D sequences."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1D(dim, hidden_dim * 3, 1, bias_attr=False)
        self.to_out = nn.Conv1D(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, axis=1)
        q, k, v = [t.reshape([b, h, -1, n]) for t in qkv]
        q = q * self.scale
        sim = paddle.einsum("bhdi,bhdj->bhij", q, k)
        attn = F.softmax(sim, axis=-1)
        # out = einsum('b h i j, b h d j -> b h i d', attn, v) → [B,h,N,d_h]
        out = paddle.einsum("bhij,bhdj->bhid", attn, v)
        # reshape to [B, hidden, N]
        out = out.transpose([0, 1, 3, 2]).reshape([b, -1, n])
        return self.to_out(out)


def _downsample_1d(dim, dim_out=None):
    """Strided Conv1D downsample (factor 2)."""
    return nn.Conv1D(dim, default(dim_out, dim), 4, stride=2, padding=1)


class _Upsample1DNearest(nn.Layer):
    """Nearest-neighbour 1D upsample (factor 2) via repeat-interleave."""

    def forward(self, x):
        # x: [B, C, L] → [B, C, L*2]
        return x.repeat_interleave(2, axis=2)


def _upsample_1d(dim, dim_out=None):
    """Nearest-neighbour upsample + Conv1D."""
    return nn.Sequential(
        _Upsample1DNearest(),
        nn.Conv1D(dim, default(dim_out, dim), 3, padding=1),
    )


# ---------------------------------------------------------------------------
# 1-D U-Net
# ---------------------------------------------------------------------------

class Unet1D(nn.Layer):
    """1D U-Net denoising model with condition encoding.

    Args:
        dim: base channel dimension.
        channels: input / output channels (synthesis-condition dimensions).
        cond_dim: raw condition feature dimension fed into the condition MLP.
        dim_mults: channel multipliers per resolution level.
        groups: GroupNorm groups.
        cond_drop_prob: probability of dropping conditions (classifier-free guidance).
    """

    def __init__(
        self,
        dim=64,
        channels=3,
        cond_dim=32,
        dim_mults=(1, 2, 4),
        groups=8,
        cond_drop_prob=0.5,
    ):
        super().__init__()
        self.channels = channels
        self.out_dim = channels
        self.cond_drop_prob = cond_drop_prob

        # initial projection
        init_dim = dim
        self.init_conv = nn.Conv1D(channels, init_dim, 7, padding=3)

        dims = [init_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        cond_inner_dim = dim * 4

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # condition MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_inner_dim),
            nn.ReLU(),
            nn.Linear(cond_inner_dim, cond_inner_dim),
        )

        # null-condition embedding for classifier-free guidance
        self.null_cond_emb = paddle.create_parameter(
            shape=[cond_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(mean=0.0, std=1.0),
        )

        block_klass = partial(
            ResnetBlock1D, time_emb_dim=time_dim, cond_emb_dim=cond_inner_dim, groups=groups
        )

        # --- encoder (down) ---
        self.downs = nn.LayerList()
        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.downs.append(
                nn.LayerList(
                    [
                        block_klass(d_in, d_in),
                        block_klass(d_in, d_in),
                        Residual(PreNorm(d_in, LinearAttention1D(d_in))),
                        _downsample_1d(d_in, d_out)
                        if not is_last
                        else nn.Conv1D(d_in, d_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention1D(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # --- decoder (up) ---
        self.ups = nn.LayerList()
        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(
                nn.LayerList(
                    [
                        block_klass(d_out + d_in, d_out),
                        block_klass(d_out + d_in, d_out),
                        Residual(PreNorm(d_out, LinearAttention1D(d_out))),
                        _upsample_1d(d_out, d_in)
                        if not is_last
                        else nn.Conv1D(d_out, d_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Conv1D(dim, self.out_dim, 1)

    # -- classifier-free guidance entry point --
    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, cond_drop_prob=0.0, **kwargs)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, cond=None, cond_drop_prob=None):
        """
        Args:
            x: [B, C, L] noisy input.
            time: [B] integer timestep indices.
            cond: [B, cond_dim] conditioning features (or ``None``).
            cond_drop_prob: override for condition dropout probability.
        """
        batch = x.shape[0]
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # --- condition encoding with classifier-free dropout ---
        if exists(cond):
            if cond_drop_prob > 0.0:
                keep_mask = (
                    paddle.rand([batch]) >= cond_drop_prob
                )  # True → keep
                null_cond = self.null_cond_emb.unsqueeze(0).expand([batch, -1])
                cond = paddle.where(
                    keep_mask.unsqueeze(-1), cond, null_cond
                )
            c = self.cond_mlp(cond)
        else:
            # unconditional — use null embedding for every sample
            null_cond = self.null_cond_emb.unsqueeze(0).expand([batch, -1])
            c = self.cond_mlp(null_cond)

        # --- time embedding ---
        t = self.time_mlp(time)

        # --- U-Net ---
        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)
            x = block2(x, t, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = paddle.concat([x, h.pop()], axis=1)
            x = block1(x, t, c)
            x = paddle.concat([x, h.pop()], axis=1)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)

        x = paddle.concat([x, r], axis=1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


# ---------------------------------------------------------------------------
# DiffSyn — top-level model following PaddleMaterials conventions
# ---------------------------------------------------------------------------

class DiffSyn(nn.Layer):
    """DiffSyn: Conditional 1D Gaussian Diffusion for zeolite synthesis conditions.

    Wraps :class:`Unet1D` in a DDPM / DDIM diffusion process and exposes
    the ``forward`` / ``predict`` API expected by PaddleMaterials.

    Args:
        dim: base U-Net channel width.
        channels: number of synthesis-condition channels.
        seq_length: temporal length of the 1D signal.
        cond_dim: condition feature dimensionality.
        dim_mults: channel multipliers per resolution level.
        timesteps: number of diffusion timesteps *T*.
        beta_schedule: ``'cosine'`` or ``'linear'``.
        objective: prediction target — ``'pred_noise'``, ``'pred_x0'``, or ``'pred_v'``.
        cond_drop_prob: classifier-free guidance dropout probability.
        property_names: key used in the returned prediction dict.
        data_mean / data_std: normalisation statistics.
        loss_type: ``'l1_loss'`` or ``'mse_loss'``.
    """

    def __init__(
        self,
        dim=64,
        channels=3,
        seq_length=8,
        cond_dim=32,
        dim_mults=(1, 2, 4),
        groups=8,
        timesteps=1000,
        sampling_timesteps=None,
        beta_schedule="cosine",
        objective="pred_noise",
        cond_drop_prob=0.5,
        ddim_sampling_eta=1.0,
        property_names="synthesis_conditions",
        data_mean=0.0,
        data_std=1.0,
        loss_type="l1_loss",
    ):
        super().__init__()

        self.seq_length = seq_length
        self.channels = channels
        self.objective = objective
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, f"objective must be pred_noise | pred_x0 | pred_v, got {objective}"

        # denoising backbone
        self.denoise_model = Unet1D(
            dim=dim,
            channels=channels,
            cond_dim=cond_dim,
            dim_mults=dim_mults,
            groups=groups,
            cond_drop_prob=cond_drop_prob,
        )

        # --- diffusion schedule ---
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        self.num_timesteps = int(betas.shape[0])

        # sampling
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def _buf(name, arr):
            self.register_buffer(name, paddle.to_tensor(arr, dtype="float32"))

        _buf("betas", betas)
        _buf("alphas_cumprod", alphas_cumprod)
        _buf("alphas_cumprod_prev", alphas_cumprod_prev)

        _buf("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        _buf("sqrt_one_minus_alphas_cumprod", np.sqrt(1.0 - alphas_cumprod))
        _buf("log_one_minus_alphas_cumprod", np.log(1.0 - alphas_cumprod))
        _buf("sqrt_recip_alphas_cumprod", np.sqrt(1.0 / alphas_cumprod))
        _buf("sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / alphas_cumprod - 1.0))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        _buf("posterior_variance", posterior_variance)
        _buf(
            "posterior_log_variance_clipped",
            np.log(np.clip(posterior_variance, 1e-20, None)),
        )
        _buf(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        _buf(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # SNR-based loss weight
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        if objective == "pred_noise":
            loss_weight = np.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr.copy()
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1.0)
        _buf("loss_weight", loss_weight)

        # PM conventions
        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            self.property_names = property_names

        if loss_type == "l1_loss":
            self.loss_fn = F.l1_loss
        elif loss_type == "mse_loss":
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.register_buffer(
            "data_mean", paddle.to_tensor(np.atleast_1d(np.float32(data_mean)))
        )
        self.register_buffer(
            "data_std", paddle.to_tensor(np.atleast_1d(np.float32(data_std)))
        )

    # -- normalisation helpers (PM convention) --
    def normalize(self, t):
        return (t - self.data_mean) / self.data_std

    def unnormalize(self, t):
        return t * self.data_std + self.data_mean

    # ------------------------------------------------------------------
    # Diffusion math
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion q(x_t | x_0)."""
        if noise is None:
            noise = paddle.randn(x_start.shape)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond=None, cond_scale=3.0, clip_x_start=False):
        model_output = self.denoise_model.forward_with_cond_scale(
            x, t, cond=cond, cond_scale=cond_scale
        )
        maybe_clip = (
            partial(paddle.clip, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond=None, cond_scale=3.0, clip_denoised=True):
        preds = self.model_predictions(x, t, cond=cond, cond_scale=cond_scale)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start = paddle.clip(x_start, -1.0, 1.0)
        model_mean, post_var, post_log_var = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, post_var, post_log_var, x_start

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @paddle.no_grad()
    def p_sample(self, x, t_int, cond=None, cond_scale=3.0, clip_denoised=True):
        b = x.shape[0]
        batched_times = paddle.full([b], t_int, dtype="int64")
        model_mean, _, model_log_var, x_start = self.p_mean_variance(
            x=x, t=batched_times, cond=cond, cond_scale=cond_scale,
            clip_denoised=clip_denoised,
        )
        noise = paddle.randn(x.shape) if t_int > 0 else paddle.zeros(x.shape)
        pred = model_mean + (0.5 * model_log_var).exp() * noise
        return pred, x_start

    @paddle.no_grad()
    def p_sample_loop(self, shape, cond=None, cond_scale=3.0):
        img = paddle.randn(shape)
        for t in reversed(range(0, self.num_timesteps)):
            img, _ = self.p_sample(img, t, cond=cond, cond_scale=cond_scale)
        return img

    @paddle.no_grad()
    def ddim_sample(self, shape, cond=None, cond_scale=3.0, clip_denoised=True):
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = np.linspace(-1, total_timesteps - 1, sampling_timesteps + 1)
        times = list(reversed(times.astype(int).tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        batch = shape[0]
        img = paddle.randn(shape)

        for time, time_next in time_pairs:
            time_cond = paddle.full([batch], time, dtype="int64")
            pred_noise, x_start = self.model_predictions(
                img, time_cond, cond=cond, cond_scale=cond_scale,
                clip_x_start=clip_denoised,
            )
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = paddle.randn(img.shape)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    @paddle.no_grad()
    def sample(self, batch_size=1, cond=None, cond_scale=3.0):
        """Full diffusion sampling (DDPM or DDIM)."""
        shape = [batch_size, self.channels, self.seq_length]
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        return sample_fn(shape, cond=cond, cond_scale=cond_scale)

    # ------------------------------------------------------------------
    # Training forward (PM convention)
    # ------------------------------------------------------------------

    def _forward(self, data):
        """Core forward: compute diffusion training loss targets."""
        x_start = data["x"]  # [B, C, L]
        cond = data.get("cond", None)
        b = x_start.shape[0]
        t = paddle.randint(0, self.num_timesteps, [b])
        noise = paddle.randn(x_start.shape)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = self.denoise_model(x_noisy, t, cond=cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)

        return pred, target

    def forward(self, data, return_loss=True, return_prediction=True):
        assert return_loss or return_prediction, (
            "At least one of return_loss or return_prediction must be True."
        )
        pred, target = self._forward(data)

        loss_dict = {}
        if return_loss:
            loss = self.loss_fn(input=pred, label=target)
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            prediction[self.property_names] = pred

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, data):
        """Generate samples via the full reverse diffusion process."""
        cond = data.get("cond", None)
        batch_size = cond.shape[0] if exists(cond) else 1
        samples = self.sample(batch_size=batch_size, cond=cond)
        return {self.property_names: self.unnormalize(samples)}
