# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate SchNet on QM9 U0 test set.

Computes MAE against ground-truth labels using pretrained checkpoint.
Requires: paddlepaddle, numpy, scipy.

Usage:
    python tools/eval_schnet_qm9.py \
        --checkpoint checkpoints/schnet_qm9_U0/schnet_qm9_U0.pdparams \
        --qm9_dir data/qm9 \
        --split_file path/to/split.npz \
        --n_samples 200
"""
import argparse
import os
import sys
import time
import types
import importlib.util

import numpy as np
import paddle
from scipy.spatial.distance import pdist, squareform


def load_model(checkpoint_path, mean, std):
    """Load SchNet model from checkpoint."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pkg in [
        "ppmat", "ppmat.models", "ppmat.models.schnet",
        "ppmat.utils", "ppmat.utils.scatter", "ppmat.utils.crystal",
        "ppmat.utils.paddle_aux", "ppmat.datasets",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    for mod_name, rel_path in [
        ("ppmat.utils.paddle_aux", "ppmat/utils/paddle_aux.py"),
        ("ppmat.utils.scatter", "ppmat/utils/scatter.py"),
        ("ppmat.utils.crystal", "ppmat/utils/crystal.py"),
        ("ppmat.models.schnet.schnet", "ppmat/models/schnet/schnet.py"),
    ]:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(repo_root, rel_path)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    from ppmat.models.schnet.schnet import SchNet
    state_dict = paddle.load(checkpoint_path)
    model = SchNet(
        n_atom_basis=128, n_interactions=6, n_gaussians=50, cutoff=10.0,
        data_mean=mean, data_std=std, readout="sum", property_names="energy_U0",
    )
    model.set_state_dict(state_dict)
    model.eval()
    return model, state_dict


def evaluate(model, state_dict, qm9_dir, split_file, n_samples, cutoff=10.0):
    """Evaluate model MAE on QM9 test split."""
    from ppmat.models.schnet.schnet import shifted_softplus
    from ppmat.utils.scatter import scatter

    split = np.load(split_file)
    test_idx = split["test_idx"]

    # Load QM9 data
    try:
        from ase.db import connect
        db_path = os.path.join(qm9_dir, "qm9.db") if not qm9_dir.endswith(".db") else qm9_dir
        db = connect(db_path)
    except ImportError:
        print("ase required for QM9 evaluation. Install: pip install ase")
        return

    atomref = state_dict["atomref"].numpy() if "atomref" in state_dict else None
    n_eval = min(n_samples, len(test_idx))
    preds, labels = [], []
    t0 = time.time()

    for count, idx in enumerate(test_idx[:n_eval]):
        row = db.get(int(idx) + 1)
        atoms_np = row.numbers
        positions_np = row.positions
        label = row.data.get("energy_U0", row.get("U0", None))
        if label is None:
            continue
        n_atoms = len(atoms_np)

        dists_mat = squareform(pdist(positions_np))
        src_l, dst_l, dist_l = [], [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j and dists_mat[i, j] < cutoff:
                    src_l.append(i)
                    dst_l.append(j)
                    dist_l.append(dists_mat[i, j])

        atom_types_t = paddle.to_tensor(atoms_np, dtype="int64")
        edge_src_t = paddle.to_tensor(np.array(src_l), dtype="int64")
        edge_dst_t = paddle.to_tensor(np.array(dst_l), dtype="int64")
        dist_t = paddle.to_tensor(np.array(dist_l, dtype=np.float32))
        batch = paddle.zeros([n_atoms], dtype="int64")

        with paddle.no_grad():
            rbf = model.rbf(dist_t)
            x = model.embedding(atom_types_t)
            for interaction in model.interactions:
                x = interaction(x, rbf, edge_src_t, edge_dst_t, n_atoms, dist=dist_t)
            x = shifted_softplus(model.output_network[0](x))
            atom_energy = model.output_network[1](x)
            atom_energy = atom_energy * model.data_std + model.data_mean
            if atomref is not None:
                atomref_t = paddle.to_tensor(atomref)
                atom_energy = atom_energy + atomref_t[atom_types_t]
            energy = scatter(atom_energy, batch, dim=0, reduce="sum")

        preds.append(energy.numpy().flatten()[0])
        labels.append(float(label))

        if (count + 1) % 100 == 0:
            elapsed = time.time() - t0
            mae_so_far = np.mean(np.abs(np.array(preds) - np.array(labels)))
            print(f"  {count+1}/{n_eval} done ({elapsed:.1f}s), running MAE: {mae_so_far*1000:.3f} meV")

    preds = np.array(preds)
    labels = np.array(labels)
    mae = np.mean(np.abs(preds - labels))

    print(f"\nResults ({len(preds)} samples):")
    print(f"  MAE: {mae*1000:.3f} meV/molecule")
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Paper reference: ~14 meV for QM9 U0")
    return mae


def main():
    parser = argparse.ArgumentParser(description="Evaluate SchNet on QM9 U0")
    parser.add_argument("--checkpoint", required=True, help="Path to .pdparams")
    parser.add_argument("--qm9_dir", required=True, help="Path to QM9 data dir")
    parser.add_argument("--split_file", required=True, help="Path to split.npz")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--mean", type=float, default=-76.1160, help="Data mean")
    parser.add_argument("--std", type=float, default=10.3238, help="Data std")
    args = parser.parse_args()

    model, state_dict = load_model(args.checkpoint, args.mean, args.std)
    evaluate(model, state_dict, args.qm9_dir, args.split_file, args.n_samples)


if __name__ == "__main__":
    main()
