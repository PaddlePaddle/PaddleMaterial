# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from ppmat.schedulers import build_scheduler


class GaussianDiffusion:
    def __init__(self, model_mean_type="epsilon", model_var_type="fixed_small", loss_type="mse", **kwargs):
        sched_kwargs = {k: v for k, v in kwargs.items() if k in (
            "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
            "variance_type", "prediction_type", "clip_sample",
        )}
        sched_kwargs.setdefault("variance_type", "fixed_small")
        sched_kwargs.setdefault("prediction_type", "epsilon")
        sched_kwargs.setdefault("beta_schedule", "linear")
        self.scheduler = build_scheduler({
            "__class_name__": "DDPMScheduler",
            "__init_params__": sched_kwargs,
        })
        self.num_timesteps = self.scheduler.num_train_timesteps

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn_like(x_start)
        return self.scheduler.add_noise(x_start, noise, t)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if noise is None:
            noise = paddle.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t, **(model_kwargs or {}))

        # Model may output 2x channels when learn_sigma=True (noise + variance)
        if model_output.shape[-1] == x_start.shape[-1] * 2:
            model_output, _ = paddle.split(model_output, 2, axis=-1)

        target = noise
        mask = (model_kwargs or {}).get("mask")

        if mask is not None:
            while mask.ndim < target.ndim:
                mask = mask.unsqueeze(-1)
            mse = ((target - model_output) ** 2 * mask.astype(target.dtype)).sum(axis=list(range(1, target.ndim))) / mask.astype(target.dtype).sum(axis=list(range(1, target.ndim))).clip(min=1e-6)
        else:
            mse = paddle.mean((target - model_output) ** 2, axis=list(range(1, target.ndim)))

        return {"mse": mse.mean(), "loss": mse.mean()}

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, t, **(model_kwargs or {}))
        if model_output.shape[-1] == x.shape[-1] * 2:
            model_output, _ = paddle.split(model_output, 2, axis=-1)
        out = self.scheduler.step(model_output, int(t[0].item()), x)
        return {"sample": out.prev_sample, "pred_xstart": out.pred_original_sample}

    def _loop(self, model, shape, noise, model_kwargs, sampler_fn, progress, clip_denoised, eta=0.0):
        img = noise if noise is not None else paddle.randn(shape)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            try:
                from tqdm.auto import tqdm
                indices = tqdm(indices)
            except ImportError:
                pass
        for i in indices:
            t = paddle.to_tensor([i] * shape[0], dtype='int64')
            with paddle.no_grad():
                img = sampler_fn(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, eta=eta)["sample"]
        return img

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, progress=False, device=None):
        return self._loop(model, shape, noise, model_kwargs, lambda m, x, t, **kw: self.p_sample(m, x, t, **kw), progress, clip_denoised)

    def _extract(self, arr, t, shape):
        res = paddle.to_tensor(arr, dtype='float32')[t]
        while res.ndim < len(shape):
            res = res.unsqueeze(-1)
        return paddle.broadcast_to(res, shape)

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, eta=0.0):
        model_output = model(x, t, **(model_kwargs or {}))
        if model_output.shape[-1] == x.shape[-1] * 2:
            model_output, _ = paddle.split(model_output, 2, axis=-1)
        sched = self.scheduler
        alpha_bar = self._extract(sched.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract(sched.alphas_cumprod.clip(min=1e-8), (t - 1).clip(min=0), x.shape)

        pred_x0 = (x - (1 - alpha_bar).sqrt() * model_output) / alpha_bar.sqrt()
        if clip_denoised:
            pred_x0 = pred_x0.clip(-1, 1)

        eps = (x - alpha_bar.sqrt() * pred_x0) / (1 - alpha_bar).sqrt()

        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt() * (1 - alpha_bar / alpha_bar_prev).sqrt()
        noise = paddle.randn_like(x) * (t != 0).astype('float32').reshape([-1] + [1] * (x.ndim - 1))
        mean = pred_x0 * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev - sigma ** 2).clip(min=0).sqrt() * eps
        return {"sample": mean + sigma * noise, "pred_xstart": pred_x0, "mean": mean, "std": sigma}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, progress=False, device=None, eta=0.0):
        return self._loop(model, shape, noise, model_kwargs, lambda m, x, t, **kw: self.ddim_sample(m, x, t, **kw), progress, clip_denoised, eta=eta)


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(**kwargs)
        use_timesteps = sorted(set(use_timesteps))
        self.use_timesteps = use_timesteps
        self.original_num_steps = self.num_timesteps

        sched = self.scheduler
        last_alpha = 1.0
        new_betas = []
        self.timestep_map = []
        for i in range(sched.num_train_timesteps):
            if i in use_timesteps:
                new_betas.append(float(1 - sched.alphas_cumprod[i] / last_alpha))
                last_alpha = float(sched.alphas_cumprod[i])
                self.timestep_map.append(i)
        self.num_timesteps = len(new_betas)
        self._map_tensor = paddle.to_tensor(self.timestep_map)

    def _wrap_model(self, model):
        map_t = self._map_tensor
        def wrapped(x, ts, **kw):
            return model(x, map_t[ts], **kw)
        return wrapped

    def p_sample(self, model, *args, **kwargs):
        return super().p_sample(self._wrap_model(model), *args, **kwargs)

    def ddim_sample(self, model, *args, **kwargs):
        return super().ddim_sample(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def p_sample_loop(self, model, *args, **kwargs):
        return super().p_sample_loop(self._wrap_model(model), *args, **kwargs)

    def ddim_sample_loop(self, model, *args, **kwargs):
        return super().ddim_sample_loop(self._wrap_model(model), *args, **kwargs)


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str) and section_counts.startswith("ddim"):
        desired = int(section_counts[len("ddim"):])
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired:
                return set(range(0, num_timesteps, i))
        raise ValueError(f"cannot create {num_timesteps} steps with integer stride")
    counts = [int(x) for x in section_counts.split(",")] if isinstance(section_counts, str) else section_counts
    size_per = num_timesteps // len(counts)
    extra = num_timesteps % len(counts)
    all_steps = []
    start = 0
    for i, cnt in enumerate(counts):
        size = size_per + (1 if i < extra else 0)
        if size < cnt:
            raise ValueError(f"cannot divide {size} steps into {cnt}")
        stride = (size - 1) / (cnt - 1) if cnt > 1 else 1
        all_steps += [start + round(j * stride) for j in range(cnt)]
        start += size
    return set(all_steps)


def create_diffusion(timestep_respacing, noise_schedule="linear", sigma_small=False,
                      learn_sigma=True, diffusion_steps=1000, **kwargs):
    beta_schedule_map = {"linear": "linear", "squaredcos_cap_v2": "squaredcos_cap_v2"}
    bs = beta_schedule_map.get(noise_schedule, "linear")
    var_type = "learned_range" if learn_sigma else ("fixed_small" if sigma_small else "fixed_large")

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    kw = dict(num_train_timesteps=diffusion_steps, beta_schedule=bs, variance_type=var_type, prediction_type="epsilon")
    if isinstance(timestep_respacing, (list, tuple)) or isinstance(timestep_respacing, str):
        use_ts = space_timesteps(diffusion_steps, timestep_respacing)
        return SpacedDiffusion(use_timesteps=use_ts, **kw)

    return GaussianDiffusion(**kw)
