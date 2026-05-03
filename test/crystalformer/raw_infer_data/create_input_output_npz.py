#!/usr/bin/env python
"""Generate reference I/O arrays for CrystalFormer regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical CrystalFormer with PyTorch (model_dim=64, n_heads=4,
    # n_layers=2, n_gaussians=20, cutoff=8.0), create silicon diamond-cubic
    # unit cell input, run forward pass, save outputs.
    # np.savez("reference_io.npz", pred=pred, loss=loss)

This script uses the Paddle implementation to create a deterministic
reference snapshot.  Re-run only when the model architecture intentionally
changes.
"""

import importlib.util
import os

import numpy as np
import paddle

_mod_path = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "ppmat", "models", "crystalformer", "crystalformer.py",
    )
)
_spec = importlib.util.spec_from_file_location(
    "ppmat.models.crystalformer.crystalformer", _mod_path
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CrystalFormer = _mod.CrystalFormer


def _make_si_crystal(B=1):
    N = 2
    x = paddle.zeros([B, N, 98], dtype="float32")
    for b in range(B):
        x[b, 0, 13] = 1.0
        x[b, 1, 13] = 1.0
    pos = paddle.to_tensor(
        [[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]] * B, dtype="float32"
    )
    a = 5.43
    trans_vec = paddle.to_tensor(
        [[[a, 0, 0], [0, a, 0], [0, 0, a]]] * B, dtype="float32"
    )
    mask = paddle.ones([B, N], dtype="bool")
    label = paddle.to_tensor([[-0.5]] * B, dtype="float32")
    return {
        "x": x, "pos": pos, "trans_vec": trans_vec,
        "mask": mask, "sizes": paddle.to_tensor([N] * B, dtype="int64"),
        "formation_energy_per_atom": label,
    }


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = CrystalFormer(
        atom_feat_dim=98, model_dim=64, n_heads=4,
        ff_dim=128, n_layers=2, n_gaussians=20,
        cutoff=8.0, lattice_range=1,
        embedding_dim=[64, 32], dropout=0.0,
        property_names="formation_energy_per_atom",
    )
    model.eval()

    data = _make_si_crystal(B=1)
    result = model(data, return_loss=True)
    pred = result["pred_dict"]["formation_energy_per_atom"].numpy()
    loss = result["loss_dict"]["loss"].numpy()

    out_path = os.path.join(os.path.dirname(__file__), "reference_io.npz")
    np.savez(out_path, pred=pred, loss=loss)
    print(f"Saved reference to {out_path}")
    print(f"  pred shape={pred.shape}, values={pred}")
    print(f"  loss={loss}")


if __name__ == "__main__":
    main()
