#!/usr/bin/env python
"""Generate reference I/O arrays for TrinityLLM regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical TrinityLLM with PyTorch (n_vocab=100, n_embd=64,
    # n_layers=2, n_heads=4, dropout=0.0), create token_ids input,
    # run forward pass, save prediction output.
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

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_MOD_PATH = os.path.join(
    _REPO_ROOT, "ppmat", "models", "trinityllm", "trinityllm.py"
)
_spec = importlib.util.spec_from_file_location("trinityllm", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrinityLLM = _mod.TrinityLLM


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = TrinityLLM(
        n_vocab=100, n_embd=64, n_layers=2,
        n_heads=4, dropout=0.0,
    )
    model.eval()

    data = {
        "token_ids": paddle.to_tensor([[0, 5, 12, 8, 23, 1, 2, 2]], dtype="int64"),
        "property": paddle.to_tensor([[-1.5]], dtype="float32"),
    }

    result = model(data, return_loss=True, return_prediction=True)
    pred = result["pred_dict"]["property"].numpy()
    loss = result["loss_dict"]["loss"].numpy()

    out_path = os.path.join(os.path.dirname(__file__), "reference_io.npz")
    np.savez(out_path, pred=pred, loss=loss)
    print(f"Saved reference to {out_path}")
    print(f"  pred shape={pred.shape}, values={pred}")
    print(f"  loss={loss}")


if __name__ == "__main__":
    main()
