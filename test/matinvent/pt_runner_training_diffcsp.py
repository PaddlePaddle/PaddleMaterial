#!/usr/bin/env python3
"""DiffCSP PyTorch 训练子脚本 —— 在 matinvent conda 环境下执行。"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RAW_ROOT = Path(sys.argv[1])
DATA_PATH = Path(sys.argv[2])
OUT_JSON = Path(sys.argv[3])
LR = float(sys.argv[4])
NUM_EPOCHS = int(sys.argv[5])
# sys.argv[6] = seed（保留参数位，diffcsp 当前未按 step 设 seed）

RAW_DIFFCSP_CKPT = Path("~/.paddlemat/weights/matinvent/raw-diffcsp.ckpt").expanduser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(DATA_PATH) as f:
    data = json.load(f)
batches = data["batches"]

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

ckpt = torch.load(str(RAW_DIFFCSP_CKPT), map_location=device)
sd = ckpt["state_dict"]
decoder_sd = {
    k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")
}
decoder_sd = {k: v for k, v in decoder_sd.items() if not k.startswith("type_out")}
model.load_state_dict(decoder_sd, strict=False)
model.to(device).train()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
mse_loss = nn.MSELoss()
losses = []

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    for batch in batches:
        time_emb = torch.tensor(batch["time_emb"], dtype=torch.float32).to(device)
        atom_types = torch.tensor(batch["atom_type_probs"], dtype=torch.float32).to(
            device
        )
        frac_coords = torch.tensor(batch["frac_coords"], dtype=torch.float32).to(device)
        lattices = torch.tensor(batch["lattices"], dtype=torch.float32).to(device)
        num_atoms = torch.tensor(batch["num_atoms"], dtype=torch.long).to(device)
        batch_idx_t = torch.tensor(batch["batch_idx"], dtype=torch.long).to(device)
        target_l = torch.tensor(batch["target_l"], dtype=torch.float32).to(device)
        target_x = torch.tensor(batch["target_x"], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        pred_l, pred_x = model(
            time_emb, atom_types, frac_coords, lattices, num_atoms, batch_idx_t
        )
        loss = mse_loss(pred_l, target_l) + mse_loss(pred_x, target_x)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    losses.extend(epoch_losses)
    print(
        f"[PT] Epoch {epoch + 1}/{NUM_EPOCHS}: mean_loss={np.mean(epoch_losses):.6f}",
        flush=True,
    )

result = {
    "losses": losses,
    "mean_loss": float(np.mean(losses)),
    "std_loss": float(np.std(losses)),
}
with open(OUT_JSON, "w") as fp:
    json.dump(result, fp)
print(f"[PT] Completed: mean_loss={result['mean_loss']:.6f}", flush=True)
