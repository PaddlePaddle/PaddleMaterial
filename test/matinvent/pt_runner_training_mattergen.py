#!/usr/bin/env python3
"""MatterGen PyTorch 训练子脚本 —— 在 matinvent conda 环境下执行。

该脚本使用 monkey-patch 将 torch 随机函数替换为基于 numpy 的实现，
使得 PT 和 PD (Paddle) 两侧产生完全相同的噪声输入，从而实现训练 loss 对齐。

背景：
  MatterGen 的 calc_loss 在每步训练中需要 4 种随机噪声：
    1. timestep  : torch.rand(batch_size)            -- 扩散时间步采样
    2. atom_type : Categorical(logits).sample()       -- D3PM 原子类型加噪
    3. lattice   : torch.randn_like(x) shape=(B,3,3)  -- 晶格高斯噪声
    4. coord     : torch.randn_like(x) shape=(N,3)    -- 坐标高斯噪声

  torch.manual_seed 与 paddle.seed 产生不同的随机序列，导致两侧加噪结果不同，
  loss 对比就没有意义了。

设计：
  - 用 per-purpose 的独立 np.random.RandomState 替代框架 RNG。
  - 每种噪声有固定 seed 偏移 (seed_i*4 + {0,1,2,3})，不受调用顺序影响。
  - 通过 tensor shape 区分用途：len(shape)==3 -> lattice，其余 -> coord。
  - PT 侧 Categorical.sample 在 calc_loss 中被调用 2 次（第 1 次在 d3pm.q_sample
    内部，结果被丢弃；第 2 次在 sample_marginal 中实际使用），PD 侧仅 1 次。
    通过调用计数器在第 2 次调用时重置 _rng_cat，使其与 PD 侧的第 1 次调用对齐。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RAW_ROOT = Path(sys.argv[1])
DATA_PATH = Path(sys.argv[2])
OUT_JSON = Path(sys.argv[3])
LR = float(sys.argv[4])
NUM_EPOCHS = int(sys.argv[5])
BASE_SEED = int(sys.argv[6])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATA_PATH) as f:
    data = json.load(f)
batches = data["batches"]

sys.path.insert(0, str(RAW_ROOT))
from huggingface_hub import hf_hub_download  # noqa: E402
from mattergen.common.data.chemgraph import ChemGraph  # noqa: E402
from mattergen.common.data.collate import collate  # noqa: E402
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo  # noqa: E402
from mattergen.diffusion.lightning_module import DiffusionLightningModule  # noqa: E402

# 替代 from_hf_hub：手动下载将由 matinvent_test.py 预先完成，此处为缓存命中
ckpt_cache = hf_hub_download(
    repo_id="microsoft/mattergen",
    filename="checkpoints/mattergen_base/checkpoints/last.ckpt",
)
config_cache = hf_hub_download(
    repo_id="microsoft/mattergen",
    filename="checkpoints/mattergen_base/config.yaml",
)
ckpt_info = MatterGenCheckpointInfo(
    model_path=str(Path(config_cache).parent),
    load_epoch="last",
)
model, _ = DiffusionLightningModule.load_from_checkpoint_and_config(
    ckpt_info.checkpoint_path,
    config=ckpt_info.config.lightning_module,
    map_location=device,
    strict=False,
)
model.to(device).train()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

losses = []
num_batches = len(batches)

# ===========================================================================
# Monkey-patch: 将 torch 随机函数替换为 numpy 实现，与 Paddle 侧噪声对齐
# ---------------------------------------------------------------------------
# 核心思路：per-purpose 独立 RNG
#   每种噪声使用独立的 np.random.RandomState，seed 互不重叠，
#   因此 PT/PD 两侧即使随机函数调用顺序不同，产生的噪声张量也完全一致。
#
# 为什么不能用单一 numpy RNG？
#   PT 调用顺序: rand -> Categorical x2 -> randn_like(lattice) -> randn_like(coord)
#   PD 调用顺序: rand -> randn(coord)    -> randn(lattice)    -> Categorical x1
#   顺序不同导致同一 RNG stream 中的数值被不同用途消耗，结果不一致。
# ===========================================================================

_orig_rand = torch.rand
_orig_randn_like = torch.randn_like
_OrigCategorical = torch.distributions.Categorical
_orig_cat_sample = _OrigCategorical.sample

# 每步开始时按 seed_i 重新创建，seed 偏移：
#   +0 = timestep (rand)
#   +1 = coord    (randn_like, shape 2D)
#   +2 = lattice  (randn_like, shape 3D)
#   +3 = atom_cat (Categorical.sample)
_rng_timestep = None
_rng_coord = None
_rng_lattice = None
_rng_cat = None
_cat_seed_base = 0  # 保存 cat seed，用于第 2 次调用时重置 RNG
_cat_call_idx = 0  # 每步 Categorical 调用计数器


def _np_rand(*size, device=None, dtype=None, **kwargs):
    """替代 torch.rand -- 用 _rng_timestep 生成 [0,1) 均匀分布。
    仅用于 timestep_samplers.UniformTimestepSampler.__call__ 中的
    torch.rand(batch_size, device=device)。
    """
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])
    arr = _rng_timestep.rand(*size).astype(np.float32)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


def _np_randn_like(input, **kwargs):
    """替代 torch.randn_like -- 通过 tensor shape 区分用途并路由到对应 RNG。
    - shape 为 3D (B,3,3) -> 晶格噪声，使用 _rng_lattice
    - shape 为 2D (N,3)   -> 坐标噪声，使用 _rng_coord
    调用来源分别是：
      sde_lib.py:96  VPSDE/VE.sample_marginal  -> lattice
      sde_lib.py:96  WrappedVESDE.sample_marginal -> coord
    """
    shape = tuple(input.shape)
    if len(shape) == 3:
        rng = _rng_lattice
    else:
        rng = _rng_coord
    arr = rng.randn(*shape).astype(np.float32)
    return torch.from_numpy(arr).to(input.device)


def _np_cat_sample(self, sample_shape=torch.Size()):
    """替代 Categorical.sample -- 用 _rng_cat 执行 numpy 多项式采样。

    PT/PD 调用次数不对称问题：
      PT 的 calc_loss -> MultiCorruption.sample_marginal 对 atomic_numbers 字段
      依次调用：
        call 0: d3pm.q_sample() 内部的 Categorical(logits).sample()
                该 sample 结果仅在 return_logits=False 时使用，但 marginal_prob()
                传入 return_logits=True，因此 sample 结果被丢弃，只取 logits。
        call 1: d3pm_corruption.sample_marginal() 用返回的 logits 再次构造
                Categorical(logits).sample()，此结果才是实际使用的加噪原子类型。
      PD 侧 (scheduling_d3pm.py add_noise) 将 q_sample 的逻辑内联，
      直接计算 logits 后做 1 次 Categorical.sample()，等价于 PT 的 call 1。

    绕过方案：
      用 _cat_call_idx 计数。call 0 时使用初始 _rng_cat（该 RNG 状态会被推进
      但无影响）；call 1 时用 _cat_seed_base 重建 _rng_cat，使其与 PD 侧的
      单次调用完全对齐。
    """
    global _cat_call_idx, _rng_cat
    if _cat_call_idx == 1:
        _rng_cat = np.random.RandomState(_cat_seed_base)
    _cat_call_idx += 1

    # PT Categorical(logits=...) 自动 softmax 得到 self.probs
    probs = self.probs.detach().cpu().numpy()
    probs = probs / probs.sum(axis=-1, keepdims=True)  # 归一化防数值误差
    n = probs.shape[0]
    samples = np.array(
        [_rng_cat.choice(probs.shape[-1], p=probs[i]) for i in range(n)],
        dtype=np.int64,
    )
    return torch.from_numpy(samples).to(self.probs.device)


torch.rand = _np_rand
torch.randn_like = _np_randn_like
_OrigCategorical.sample = _np_cat_sample

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    for step_idx, batch in enumerate(batches):
        seed_i = BASE_SEED + epoch * num_batches + step_idx
        _rng_timestep = np.random.RandomState(seed_i * 4 + 0)
        _rng_coord = np.random.RandomState(seed_i * 4 + 1)
        _rng_lattice = np.random.RandomState(seed_i * 4 + 2)
        _rng_cat = np.random.RandomState(seed_i * 4 + 3)
        _cat_seed_base = seed_i * 4 + 3
        _cat_call_idx = 0

        num_atoms = batch["num_atoms"]
        frac_coords = torch.tensor(batch["frac_coords"], dtype=torch.float32)
        lattices = torch.tensor(batch["lattices"], dtype=torch.float32)
        atoms_int = torch.tensor(batch["atom_types_int"], dtype=torch.long)

        data_list = []
        start = 0
        for i, n in enumerate(num_atoms):
            data_list.append(
                ChemGraph(
                    pos=frac_coords[start: start + n],
                    cell=lattices[i: i + 1],
                    atomic_numbers=atoms_int[start: start + n],
                    num_atoms=torch.tensor([n]),
                )
            )
            start += n

        chem_batch = collate(data_list).to(device)
        optimizer.zero_grad()
        loss, _ = model.diffusion_module.calc_loss(chem_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    losses.extend(epoch_losses)
    print(
        f"[PT] Epoch {epoch + 1}/{NUM_EPOCHS}: mean_loss={np.mean(epoch_losses):.6f}",
        flush=True,
    )

# 恢复原始函数，避免污染后续代码（本脚本即将退出，但保持好习惯）
torch.rand = _orig_rand
torch.randn_like = _orig_randn_like
_OrigCategorical.sample = _orig_cat_sample

result = {
    "losses": losses,
    "mean_loss": float(np.mean(losses)),
    "std_loss": float(np.std(losses)),
}
with open(OUT_JSON, "w") as fp:
    json.dump(result, fp)
print(f"[PT] Completed: mean_loss={result['mean_loss']:.6f}", flush=True)
