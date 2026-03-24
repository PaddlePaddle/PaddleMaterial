#!/usr/bin/env python3
"""MatterGen PyTorch 前向推理子脚本
在 matinvent conda 环境下由 matinvent_test.py 调用。

argv: RAW_ROOT  input_json  PT_CKPT  output_json

PT_CKPT 是 microsoft/mattergen 的 mattergen_base/last.ckpt，
已由 _dl_pt_ckpts.py 预下载至 ~/.paddlemat/weights/matinvent/raw-mattergen.ckpt。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RAW_ROOT = Path(sys.argv[1])
INPUT_JSON = Path(sys.argv[2])
PT_CKPT = Path(sys.argv[3])  # raw-mattergen.ckpt (官方 mattergen_base)
OUTPUT_JSON = Path(sys.argv[4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(INPUT_JSON) as f:
    data = json.load(f)

sys.path.insert(0, str(RAW_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402
from mattergen.common.data.chemgraph import ChemGraph  # noqa: E402
from mattergen.common.data.collate import collate  # noqa: E402
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo  # noqa: E402
from mattergen.diffusion.lightning_module import DiffusionLightningModule  # noqa: E402

# 加载模型（使用 HuggingFace 缓存中的 config.yaml）
config_cache = hf_hub_download(
    repo_id="microsoft/mattergen",
    filename="checkpoints/mattergen_base/config.yaml",
)
ckpt_info = MatterGenCheckpointInfo(
    model_path=str(Path(config_cache).parent),
    load_epoch="last",
)
model, _ = DiffusionLightningModule.load_from_checkpoint_and_config(
    str(PT_CKPT),
    config=ckpt_info.config.lightning_module,
    map_location=device,
    strict=False,
)
model.to(device).eval()

# 构造输入
num_atoms_list = data["num_atoms"]
frac_coords = torch.tensor(data["frac_coords"], dtype=torch.float32)
lattices = torch.tensor(data["lattices"], dtype=torch.float32)
atom_types = torch.tensor(data["atom_types"], dtype=torch.long)
times = torch.tensor(data["times"], dtype=torch.float32).to(device)

data_list = []
start = 0
for i, n in enumerate(num_atoms_list):
    data_list.append(
        ChemGraph(
            pos=frac_coords[start: start + n],
            cell=lattices[i: i + 1],
            atomic_numbers=atom_types[start: start + n],
            num_atoms=torch.tensor([n]),
        )
    )
    start += n

chem_batch = collate(data_list).to(device)

with torch.no_grad():
    output = model.diffusion_module.model(chem_batch, times)

# 提取输出。mattergen denoiser 输出 ChemGraph，字段为 pos/cell/atomic_numbers
# 对应 Paddle 侧的 frac_coords/lattice/atom_types
pred_lattice = output.cell.cpu().numpy()
pred_frac_coords = output.pos.cpu().numpy()
pred_atom_types = output.atomic_numbers.cpu().numpy().astype(np.float32)

result = {
    "pred_lattice": pred_lattice.tolist(),
    "pred_frac_coords": pred_frac_coords.tolist(),
    "pred_atom_types": pred_atom_types.tolist(),
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f)
print(
    f"[PT] MatterGen forward done: lattice shape={list(pred_lattice.shape)}, "
    f"frac_coords shape={list(pred_frac_coords.shape)}, "
    f"atom_types shape={list(pred_atom_types.shape)}",
    flush=True,
)
