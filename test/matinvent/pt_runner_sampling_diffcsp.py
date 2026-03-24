#!/usr/bin/env python3
"""DiffCSP PyTorch 采样子脚本 -- 在 matinvent conda 环境下执行。

使用 numpy 随机数（而非 torch 随机数）生成所有噪声，以保证与
Paddle 侧使用相同种子时噪声序列完全一致。
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

RAW_ROOT = Path(sys.argv[1])
OUT_JSON = Path(sys.argv[2])
SEED = int(sys.argv[3])
NUM_SAMPLES = int(sys.argv[4])
NUM_ATOMS_LIST = json.loads(sys.argv[5])
NUM_INFERENCE_STEPS = int(sys.argv[6])

RAW_DIFFCSP_CKPT = Path("~/.paddlemat/weights/matinvent/raw-diffcsp.ckpt").expanduser()

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(RAW_ROOT))
import hydra  # noqa: E402
from models.diffcsp.diffusion import DiffCSPModule  # noqa: E402, F401

hparams_path = (
        RAW_ROOT / "diff_tmp" / "diffcsp_original" / "diffcsp_mp20" / "hparams.yaml"
)
if not hparams_path.exists():
    raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

cfg = OmegaConf.load(hparams_path)
cfg.model._target_ = "models.diffcsp.diffusion.DiffCSPModule"
model = hydra.utils.instantiate(
    cfg.model,
    optim=cfg.optim,
    _recursive_=False,
)
model = model.load_from_checkpoint(
    str(RAW_DIFFCSP_CKPT),
    hparams_file=str(hparams_path),
    strict=False,
)
model.to(device).eval()


def _compute_time_emb(t_value, batch_size):
    half_dim = 128
    freqs = np.exp(np.arange(half_dim) * (-math.log(10000) / (half_dim - 1)))
    emb = np.array([t_value], dtype=np.float32)[:, None] * freqs[None, :]
    te = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    return np.tile(te, (batch_size, 1)).astype(np.float32)


# 从模型中提取 scheduler buffer（已在 checkpoint 中加载）
beta_alphas = model.beta_scheduler.alphas.cpu()
beta_alphas_cumprod = model.beta_scheduler.alphas_cumprod.cpu()
beta_sigmas = model.beta_scheduler.sigmas.cpu()
sigma_sigmas = model.sigma_scheduler.sigmas.cpu()
sigma_sigmas_norm = model.sigma_scheduler.sigmas_norm.cpu()
sigma_begin = float(sigma_sigmas[1])
step_lr = 5e-6

# 构造采样时间步序列：从 1000 到 1 等距抽取 NUM_INFERENCE_STEPS 个时间步
# 原始代码使用 for t in range(1000, 0, -1)；这里用 np.linspace 子采样
timestep_schedule = np.linspace(1000, 1, NUM_INFERENCE_STEPS, dtype=int).tolist()
# 确保 adjacent timestep 正确：每个 t 的 "下一个" t
# 在完整 1000 步中 adjacent_t = t-1，在子采样中 adjacent_t = next_t_in_schedule
timestep_schedule_with_end = timestep_schedule + [0]

results = []
for idx in range(NUM_SAMPLES):
    na = NUM_ATOMS_LIST[idx % len(NUM_ATOMS_LIST)]
    with torch.no_grad():
        try:
            batch_size = 1
            # 使用 numpy 随机数生成初始噪声，保证与 Paddle 侧一致
            x_t = torch.from_numpy(np.random.rand(na, 3).astype(np.float32)).to(device)
            l_t = torch.from_numpy(
                np.random.randn(batch_size, 3, 3).astype(np.float32)
            ).to(device)
            t_t = torch.from_numpy(np.random.randn(na, 100).astype(np.float32)).to(
                device
            )
            num_atoms = torch.LongTensor([na]).to(device)
            batch_idx = torch.zeros(na, dtype=torch.long).to(device)

            for step_i, t in enumerate(timestep_schedule):
                next_t = timestep_schedule_with_end[step_i + 1]
                time_emb = torch.from_numpy(_compute_time_emb(t, batch_size)).to(device)

                alphas = beta_alphas[t]
                alphas_cumprod = beta_alphas_cumprod[t]
                c0 = 1.0 / torch.sqrt(alphas)
                c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
                sigmas = beta_sigmas[t].to(device)
                sigma_x = sigma_sigmas[t].to(device)
                s_norm = sigma_sigmas_norm[t].to(device)

                # Corrector: numpy 随机噪声
                rand_x = (
                    torch.from_numpy(np.random.randn(na, 3).astype(np.float32)).to(
                        device
                    )
                    if t > 1
                    else torch.zeros(na, 3, device=device)
                )

                corr_step_size = step_lr * (sigma_x / sigma_begin) ** 2
                corr_std_x = torch.sqrt(
                    torch.tensor(2.0, device=device) * corr_step_size
                )

                pred_l, pred_x, pred_t = model.decoder(
                    time_emb, t_t, x_t, l_t, num_atoms, batch_idx
                )
                pred_x_corr = pred_x * torch.sqrt(s_norm)
                x_t_minus_05 = x_t - corr_step_size * pred_x_corr + corr_std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t

                # Predictor: numpy 随机噪声
                rand_l = (
                    torch.from_numpy(
                        np.random.randn(batch_size, 3, 3).astype(np.float32)
                    ).to(device)
                    if t > 1
                    else torch.zeros(batch_size, 3, 3, device=device)
                )
                rand_t = (
                    torch.from_numpy(np.random.randn(na, 100).astype(np.float32)).to(
                        device
                    )
                    if t > 1
                    else torch.zeros(na, 100, device=device)
                )
                rand_x = (
                    torch.from_numpy(np.random.randn(na, 3).astype(np.float32)).to(
                        device
                    )
                    if t > 1
                    else torch.zeros(na, 3, device=device)
                )

                # adjacent sigma 使用 next_t（子采样中的下一个时间步）
                adjacent_sigma_x = sigma_sigmas[next_t].to(device)
                pred_step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
                pred_std_x = torch.sqrt(
                    (adjacent_sigma_x ** 2 * pred_step_size) / (sigma_x ** 2)
                )

                pred_l, pred_x, pred_t = model.decoder(
                    time_emb,
                    t_t_minus_05,
                    x_t_minus_05,
                    l_t_minus_05,
                    num_atoms,
                    batch_idx,
                )
                pred_x_pred = pred_x * torch.sqrt(s_norm)

                x_t = x_t_minus_05 - pred_step_size * pred_x_pred + pred_std_x * rand_x
                l_t = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                t_t = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t
                x_t = x_t % 1.0

                x_t_mean = (x_t_minus_05 - pred_step_size * pred_x_pred) % 1.0

            x_t = x_t_mean
            atom_types = torch.argmax(t_t, dim=-1) + 1

            results.append(
                {
                    "frac_coords": x_t.cpu().numpy().tolist(),
                    "lattice": l_t.cpu().numpy().tolist(),
                    "atom_types": atom_types.cpu().numpy().tolist(),
                    "num_atoms": na,
                }
            )
        except Exception as e:
            print(f"[PT] Sampling failed: {e}", flush=True)
            import traceback

            traceback.print_exc()

with open(OUT_JSON, "w") as fp:
    json.dump({"samples": results, "num_samples": len(results)}, fp)
print(f"[PT] Sampling done: {len(results)} samples", flush=True)
