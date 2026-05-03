#!/usr/bin/env python
"""Generate reference I/O arrays for NewtonNet regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical NewtonNet with PyTorch (cutoff=5.0, n_features=32,
    # n_basis=8, n_interactions=2, activation="swish"), create water
    # molecule input (H2O), run forward pass, save energy output.
    # np.savez("reference_io.npz", energy=energy, loss=loss)

This script uses the Paddle implementation to create a deterministic
reference snapshot.  Re-run only when the model architecture intentionally
changes.
"""

import importlib.util
import os

import numpy as np
import paddle

_module_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "ppmat", "models", "newtonnet", "newtonnet.py",
    )
)
_spec = importlib.util.spec_from_file_location("newtonnet_module", _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
NewtonNet = _mod.NewtonNet


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = NewtonNet(
        cutoff=5.0, n_features=32, n_basis=8,
        n_interactions=2, activation="swish",
        layer_norm=False, property_names="energy",
        data_mean=0.0, data_std=1.0, loss_type="mse_loss",
    )
    model.eval()

    data = {
        "z": paddle.to_tensor([8, 1, 1], dtype="int64"),
        "pos": paddle.to_tensor(
            [[0.0, 0.0, 0.1173],
             [0.0, 0.7572, -0.4692],
             [0.0, -0.7572, -0.4692]],
            dtype="float32",
        ),
        "batch": paddle.zeros([3], dtype="int64"),
        "cell": paddle.eye(3, dtype="float32").unsqueeze(0) * 10.0,
        "energy": paddle.to_tensor([-76.4], dtype="float32"),
    }

    result = model(data, return_loss=True, return_prediction=True)
    energy = result["pred_dict"]["energy"].numpy()
    loss = result["loss_dict"]["loss"].numpy()

    out_path = os.path.join(os.path.dirname(__file__), "reference_io.npz")
    np.savez(out_path, energy=energy, loss=loss)
    print(f"Saved reference to {out_path}")
    print(f"  energy shape={energy.shape}, values={energy}")
    print(f"  loss={loss}")


if __name__ == "__main__":
    main()
