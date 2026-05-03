#!/usr/bin/env python
"""Generate reference I/O arrays for DiffSyn regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical DiffSyn (dim=32, channels=3, seq_length=8,
    # cond_dim=16, dim_mults=(1,2), groups=4, timesteps=100,
    # objective="pred_noise"), create dummy data, run forward pass.
    # np.savez("reference_io.npz", pred=pred, loss=loss)

This script uses the Paddle implementation to create a deterministic
reference snapshot.  Re-run only when the model architecture intentionally
changes.
"""

import importlib.util
import os
import sys

import numpy as np
import paddle

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_mod_path = os.path.join(_root, "ppmat", "models", "diffsyn", "diffsyn.py")
_spec = importlib.util.spec_from_file_location("diffsyn", _mod_path)
_diffsyn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diffsyn)
DiffSyn = _diffsyn.DiffSyn


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = DiffSyn(
        dim=32, channels=3, seq_length=8, cond_dim=16,
        dim_mults=(1, 2), groups=4, timesteps=100,
        objective="pred_noise", cond_drop_prob=0.0,
        loss_type="l1_loss",
    )
    model.eval()

    # Create deterministic input (re-seed for reproducible randn)
    paddle.seed(42)
    np.random.seed(42)
    x = paddle.randn([2, 3, 8])
    cond = paddle.randn([2, 16])
    data = {"x": x, "cond": cond, "synthesis_conditions": x}

    result = model(data)
    pred = result["pred_dict"]["synthesis_conditions"].numpy()
    loss = result["loss_dict"]["loss"].numpy()

    out_path = os.path.join(os.path.dirname(__file__), "reference_io.npz")
    np.savez(out_path, pred=pred, loss=loss)
    print(f"Saved reference to {out_path}")
    print(f"  pred shape={pred.shape}")
    print(f"  loss={loss}")


if __name__ == "__main__":
    main()
