#!/usr/bin/env python
"""Generate reference I/O arrays for wD-MPNN regression testing.

Equivalent PyTorch code (for cross-framework comparison):
    import torch, numpy as np
    torch.manual_seed(42); np.random.seed(42)
    # Build identical wD-MPNN with PyTorch chemprop-style architecture,
    # create same MolGraph inputs, run forward pass, save outputs.
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

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

_feat_spec = importlib.util.spec_from_file_location(
    "ppmat.models.wd_mpnn.featurization",
    os.path.join(_ROOT, "ppmat", "models", "wd_mpnn", "featurization.py"),
)
_feat_mod = importlib.util.module_from_spec(_feat_spec)
sys.modules["ppmat.models.wd_mpnn.featurization"] = _feat_mod
_feat_spec.loader.exec_module(_feat_mod)

_model_spec = importlib.util.spec_from_file_location(
    "ppmat.models.wd_mpnn.wd_mpnn",
    os.path.join(_ROOT, "ppmat", "models", "wd_mpnn", "wd_mpnn.py"),
)
_model_mod = importlib.util.module_from_spec(_model_spec)
sys.modules["ppmat.models.wd_mpnn.wd_mpnn"] = _model_mod
_model_spec.loader.exec_module(_model_mod)

WDMPNN = _model_mod.WDMPNN
BatchMolGraph = _feat_mod.BatchMolGraph
MolGraph = _feat_mod.MolGraph


def _create_dummy_mol_graph(n_atoms=5, seed=42):
    rng = np.random.RandomState(seed)
    n_edges = n_atoms - 1
    n_bonds = 2 * n_edges
    f_atoms = rng.randn(n_atoms, 133).astype("float32")
    f_bonds = rng.randn(n_bonds, 14).astype("float32")
    w_atoms = np.ones(n_atoms, dtype="float32")
    w_bonds = np.ones(n_bonds, dtype="float32")
    b2a_list, b2revb_list = [], []
    a2b = [[] for _ in range(n_atoms)]
    for e in range(n_edges):
        fwd, rev = 2 * e, 2 * e + 1
        b2a_list.extend([e, e + 1])
        b2revb_list.extend([rev, fwd])
        a2b[e].append(fwd)
        a2b[e + 1].append(rev)
    b2a = np.array(b2a_list, dtype="int64")
    b2revb = np.array(b2revb_list, dtype="int64")
    return MolGraph(
        f_atoms=f_atoms, f_bonds=f_bonds, a2b=a2b,
        b2a=b2a, b2revb=b2revb, w_atoms=w_atoms,
        w_bonds=w_bonds, degree_of_polym=1.0,
    )


def main():
    paddle.seed(42)
    np.random.seed(42)
    model = WDMPNN(
        hidden_size=64, depth=3, dropout=0.0,
        ffn_hidden_size=64, ffn_num_layers=2,
        atom_fdim=133, bond_fdim=14,
        property_names="property",
        data_mean=0.0, data_std=1.0, loss_type="mse_loss",
    )
    model.eval()

    graphs = [_create_dummy_mol_graph(seed=42)]
    batch = BatchMolGraph(graphs)
    components = batch.get_components()
    f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, a_scope, _b_scope, degree_of_polym = components
    data = {
        "f_atoms": f_atoms, "f_bonds": f_bonds,
        "w_atoms": w_atoms, "w_bonds": w_bonds,
        "a2b": a2b, "b2a": b2a, "b2revb": b2revb,
        "a_scope": a_scope, "degree_of_polym": degree_of_polym,
        "property": paddle.to_tensor([[1.0]], dtype="float32"),
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
