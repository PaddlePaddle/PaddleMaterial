#!/usr/bin/env python3
"""DiffCSP PyTorch 前向推理子脚本
在 matinvent conda 环境下由 matinvent_test.py 调用。

argv: RAW_ROOT  input_json  PT_CKPT  output_json
"""

import json
import sys
from pathlib import Path

import torch

RAW_ROOT = Path(sys.argv[1])
INPUT_JSON = Path(sys.argv[2])
PT_CKPT = Path(sys.argv[3])
OUTPUT_JSON = Path(sys.argv[4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(INPUT_JSON) as f:
    data = json.load(f)

sys.path.insert(0, str(RAW_ROOT))
from models.diffcsp.cspnet import CSPNet  # noqa: E402

model = CSPNet(
    hidden_dim=512,
    latent_dim=256,
    num_layers=6,
    act_fn="silu",
    dis_emb="sin",
    num_freqs=128,
    edge_style="fc",
    ln=True,
    ip=True,
    smooth=True,
    pred_type=False,
    pred_scalar=False,
)

ckpt = torch.load(str(PT_CKPT), map_location=device)
sd = ckpt["state_dict"]
decoder_sd = {
    k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")
}
decoder_sd = {k: v for k, v in decoder_sd.items() if not k.startswith("type_out")}
model.load_state_dict(decoder_sd, strict=False)
model.to(device).eval()

time_emb = torch.tensor(data["time_emb"], dtype=torch.float32).to(device)
atom_types = torch.tensor(data["atom_type_probs"], dtype=torch.float32).to(device)
frac_coords = torch.tensor(data["frac_coords"], dtype=torch.float32).to(device)
lattices = torch.tensor(data["lattices"], dtype=torch.float32).to(device)
num_atoms = torch.tensor(data["num_atoms"], dtype=torch.long).to(device)
batch_idx = torch.tensor(data["batch_idx"], dtype=torch.long).to(device)

with torch.no_grad():
    pred_l, pred_x = model(
        time_emb, atom_types, frac_coords, lattices, num_atoms, batch_idx
    )

result = {
    "pred_l": pred_l.cpu().numpy().tolist(),
    "pred_x": pred_x.cpu().numpy().tolist(),
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f)
print(
    f"[PT] DiffCSP forward done: pred_l shape="
    f"{list(pred_l.shape)}, "
    f"pred_x shape={list(pred_x.shape)}",
    flush=True,
)
