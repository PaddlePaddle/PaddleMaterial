#!/usr/bin/env python
"""Generate reference I/O arrays for MOFDiff regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical MOFDiff with PyTorch (node_feat_dim=32,
    # hidden_dim=64, latent_dim=32, num_bb_types=10, max_num_bbs=5,
    # num_diffusion_steps=50), create dummy batch, run forward pass.
    # np.savez("reference_io.npz", loss=loss, loss_coord=..., ...)

This script uses the Paddle implementation to create a deterministic
reference snapshot.  Re-run only when the model architecture intentionally
changes.
"""

import os
import sys

import numpy as np
import paddle

# Add repo root to path so ppmat imports work
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _root)

from ppmat.models.mofdiff.mofdiff import MOFDiff


def _make_batch(batch_size=2, nodes_per_graph=3, node_feat_dim=32, num_bb_types=10):
    total = batch_size * nodes_per_graph
    batch_idx = paddle.repeat_interleave(
        paddle.arange(batch_size),
        paddle.full([batch_size], nodes_per_graph, dtype="int64"),
    )
    return {
        "node_features": paddle.randn([total, node_feat_dim]),
        "frac_coords": paddle.rand([total, 3]),
        "bb_types": paddle.randint(0, num_bb_types, [total]),
        "batch": batch_idx,
        "num_atoms": paddle.full([batch_size], nodes_per_graph, dtype="int64"),
        "lattice_params": paddle.randn([batch_size, 6]),
    }


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = MOFDiff(
        node_feat_dim=32, hidden_dim=64, latent_dim=32,
        num_bb_types=10, max_num_bbs=5,
        num_diffusion_steps=50, fc_num_layers=2,
    )
    model.eval()

    # Re-seed for deterministic data creation
    paddle.seed(42)
    np.random.seed(42)
    batch = _make_batch(node_feat_dim=32, num_bb_types=10)

    result = model(batch)
    loss = result["loss_dict"]["loss"].numpy()
    loss_coord = result["loss_dict"]["loss_coord"].numpy()
    loss_type = result["loss_dict"]["loss_type"].numpy()

    out_path = os.path.join(os.path.dirname(__file__), "reference_io.npz")
    np.savez(out_path, loss=loss, loss_coord=loss_coord, loss_type=loss_type)
    print(f"Saved reference to {out_path}")
    print(f"  loss={loss}")
    print(f"  loss_coord={loss_coord}")
    print(f"  loss_type={loss_type}")


if __name__ == "__main__":
    main()
