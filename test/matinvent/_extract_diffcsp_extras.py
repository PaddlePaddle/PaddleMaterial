#!/usr/bin/env python3
"""从 DiffCSP PyTorch checkpoint 中提取采样所需的额外数据。

在 matinvent conda 环境下执行（仅需 torch + numpy）。

直接从 checkpoint state_dict 提取，无需 hydra 模型实例化。
state_dict 中包含 register_buffer 注册的 scheduler 参数。

生成两个文件：
  1. diffcsp_type_out_weights.npz -- type_out 层权重
     - weight: shape (100, 512)，PT 格式（PD 需转置为 (512, 100)）
     - bias:   shape (100,)
  2. diffcsp_scheduler_buffers.npz -- BetaScheduler + SigmaScheduler 缓冲区
     - beta_scheduler.alphas:          shape [1001]
     - beta_scheduler.alphas_cumprod:  shape [1001]
     - beta_scheduler.sigmas:          shape [1001]
     - sigma_scheduler.sigmas:         shape [1001]
     - sigma_scheduler.sigmas_norm:    shape [1001]

用法:
    python _extract_diffcsp_extras.py <pt_ckpt_path> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch

PT_CKPT = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(str(PT_CKPT), map_location="cpu")
sd = ckpt["state_dict"]

# --- 1. type_out 权重 ---
# 原始代码: models/diffcsp/cspnet.py:144
#   self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
# state_dict key: decoder.type_out.weight (100, 512), decoder.type_out.bias (100,)
type_out_w = sd.get("decoder.type_out.weight")
type_out_b = sd.get("decoder.type_out.bias")

if type_out_w is None or type_out_b is None:
    print("[EXTRACT] ERROR: type_out weights not found in checkpoint", flush=True)
    sys.exit(1)

np.savez(
    str(OUT_DIR / "diffcsp_type_out_weights.npz"),
    weight=type_out_w.numpy(),
    bias=type_out_b.numpy(),
)
print(
    f"[EXTRACT] type_out saved: weight={tuple(type_out_w.shape)}, "
    f"bias={tuple(type_out_b.shape)}",
    flush=True,
)

# --- 2. scheduler buffers ---
# 这些是 PyTorch register_buffer 注册的参数，直接存在 state_dict 中。
# 原始代码: models/diffcsp/diffusion.py 中 BetaScheduler / SigmaScheduler
SCHED_KEYS = [
    "beta_scheduler.alphas",
    "beta_scheduler.alphas_cumprod",
    "beta_scheduler.sigmas",
    "sigma_scheduler.sigmas",
    "sigma_scheduler.sigmas_norm",
]
sched_data = {}
for k in SCHED_KEYS:
    if k not in sd:
        print(f"[EXTRACT] ERROR: scheduler key '{k}' not found", flush=True)
        sys.exit(1)
    sched_data[k] = sd[k].numpy()

np.savez(str(OUT_DIR / "diffcsp_scheduler_buffers.npz"), **sched_data)
print("[EXTRACT] scheduler buffers saved", flush=True)
