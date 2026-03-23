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

"""Forward alignment test between SchNetPack (PyTorch) and our Paddle implementation.

This script:
1. Loads a pretrained SchNetPack v0.3 model (PyTorch)
2. Creates a Paddle SchNet model with matching architecture
3. Transfers weights from PyTorch → Paddle
4. Constructs identical inputs for both
5. Compares forward outputs (energy predictions)

Usage:
    python tools/test_schnet_alignment.py \
        --torch_model pretrained_weights/trained_schnet_models/qm9_energy_U0/best_model \
        --paddle_params checkpoints/schnet_qm9_U0/schnet_qm9_U0.pdparams

Requirements:
    - torch, schnetpack==0.3, paddle, pgl, numpy, ase
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import paddle
import torch

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Pre-register ppmat packages as stubs to avoid the full import chain
# (which requires pymatgen, pgl, etc.)
for _pkg in [
    "ppmat",
    "ppmat.datasets",
    "ppmat.losses",
    "ppmat.metrics",
    "ppmat.models",
    "ppmat.optimizer",
    "ppmat.schedulers",
    "ppmat.trainer",
    "ppmat.utils",
    "ppmat.sampler",
]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        sys.modules[_pkg] = _m


def _load_mod(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_mod("ppmat.utils.paddle_aux", "ppmat/utils/paddle_aux.py")
_load_mod("ppmat.utils.crystal", "ppmat/utils/crystal.py")
_load_mod("ppmat.utils.scatter", "ppmat/utils/scatter.py")
_schnet_mod = _load_mod("ppmat.models.schnet.schnet", "ppmat/models/schnet/schnet.py")

SchNet = _schnet_mod.SchNet
GaussianRBF = _schnet_mod.GaussianRBF
shifted_softplus = _schnet_mod.shifted_softplus
from ppmat.utils.scatter import scatter


def create_test_molecule():
    """Create a simple water-like molecule for testing.

    Returns dict of numpy arrays representing the molecule.
    """
    # 3 atoms: O(8), H(1), H(1) in a non-periodic box
    atom_types = np.array([8, 1, 1], dtype=np.int64)
    positions = np.array(
        [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
        dtype=np.float32,
    )
    # Use a large box (non-periodic effectively)
    lattice = np.eye(3, dtype=np.float32) * 50.0
    return {
        "atom_types": atom_types,
        "positions": positions,
        "lattice": lattice,
    }


def build_neighbor_list(positions, lattice, cutoff=10.0):
    """Build neighbor list using distance matrix (no PBC needed for molecules).

    Returns edge_index [num_edges, 2] and pbc_offsets [num_edges, 3].
    """
    n = len(positions)
    edge_src = []
    edge_dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edge_src.append(j)
                    edge_dst.append(i)
    edge_index = np.stack([np.array(edge_src), np.array(edge_dst)], axis=1).astype(
        np.int64
    )
    pbc_offsets = np.zeros((len(edge_src), 3), dtype=np.float32)
    return edge_index, pbc_offsets


def run_pytorch_forward(model_path, mol):
    """Run SchNetPack v0.3 forward pass and return energy."""

    # Load model (weights_only=False needed for SchNetPack v0.3 pickle format)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if hasattr(model, "module"):
        model = model.module
    model.eval()

    # Build SchNetPack input manually (AtomsConverter broken with numpy 1.24+)
    positions = mol["positions"].astype(np.float32)
    atom_types = mol["atom_types"].astype(np.int64)
    n_atoms = len(atom_types)
    max_nbh = n_atoms - 1  # each atom sees all others

    # Build per-atom neighbor list [n_atoms, max_nbh]
    nbh_list = np.full((n_atoms, max_nbh), -1, dtype=np.int64)
    nbh_mask = np.zeros((n_atoms, max_nbh), dtype=np.float32)
    for i in range(n_atoms):
        k = 0
        for j in range(n_atoms):
            if j != i and np.linalg.norm(positions[i] - positions[j]) < 10.0:
                nbh_list[i, k] = j
                nbh_mask[i, k] = 1.0
                k += 1

    # cell_offset shape: [n_atoms, max_nbh, 3] (no PBC → all zeros)
    cell_offset = np.zeros((n_atoms, max_nbh, 3), dtype=np.float32)

    inputs = {
        "_atomic_numbers": torch.LongTensor(atom_types).unsqueeze(0),
        "_positions": torch.FloatTensor(positions).unsqueeze(0),
        "_cell": torch.FloatTensor(mol["lattice"]).unsqueeze(0),
        "_cell_offset": torch.FloatTensor(cell_offset).unsqueeze(0),
        "_neighbors": torch.LongTensor(nbh_list).unsqueeze(0),
        "_neighbor_mask": torch.FloatTensor(nbh_mask).unsqueeze(0),
        "_atom_mask": torch.ones(1, n_atoms),
    }

    with torch.no_grad():
        result = model(inputs)

    for key in result:
        if isinstance(result[key], torch.Tensor):
            energy = result[key].detach().numpy()
            print(f"  PyTorch output['{key}'] = {energy}")

    return result


def run_paddle_forward(
    params_path, mol, n_atom_basis=128, n_interactions=6, n_gaussians=50, cutoff=10.0
):
    """Run our Paddle SchNet forward pass and return energy."""
    # Load params to get data_mean and data_std
    state_dict = paddle.load(params_path)
    data_mean = float(state_dict.get("data_mean", 0.0))
    data_std = float(state_dict.get("data_std", 1.0))

    # Create model
    model = SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        n_gaussians=n_gaussians,
        cutoff=cutoff,
        data_mean=data_mean,
        data_std=data_std,
        readout="sum",
        property_names="energy_U0",
    )

    # Load weights
    missing, unexpected = model.set_state_dict(state_dict)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model.eval()

    # Build graph manually (no PGL dependency needed)
    positions = mol["positions"]
    atom_types = mol["atom_types"]
    n_atoms = len(atom_types)

    edge_index, pbc_offsets = build_neighbor_list(positions, mol["lattice"], cutoff)
    edge_src = edge_index[:, 0]
    edge_dst = edge_index[:, 1]

    # Compute distances directly
    dist = np.linalg.norm(
        positions[edge_dst] - positions[edge_src], axis=-1
    ).astype(np.float32)

    # Convert to Paddle tensors
    atom_types_t = paddle.to_tensor(atom_types, dtype="int64")
    edge_src_t = paddle.to_tensor(edge_src, dtype="int64")
    edge_dst_t = paddle.to_tensor(edge_dst, dtype="int64")
    dist_t = paddle.to_tensor(dist)
    batch = paddle.zeros([n_atoms], dtype="int64")

    with paddle.no_grad():
        rbf = model.rbf(dist_t)
        x = model.embedding(atom_types_t)
        for interaction in model.interactions:
            x = interaction(x, rbf, edge_src_t, edge_dst_t, n_atoms, dist=dist_t)
        x = shifted_softplus(model.output_network[0](x))
        atom_energy = model.output_network[1](x)  # [n_atoms, 1]

        # SchNetPack standardizes per atom, then adds atomref, then sums
        atom_energy = atom_energy * model.data_std + model.data_mean

        # Apply atomref if available
        if "atomref" in state_dict:
            atomref = paddle.to_tensor(state_dict["atomref"])  # [100, 1]
            atom_energy = atom_energy + atomref[atom_types_t]

        energy = scatter(atom_energy, batch, dim=0, reduce="sum")

    energy_val = energy.numpy().flatten()[0]
    print(f"  Paddle energy = {energy_val:.8f}")
    return {"energy": energy, "energy_val": energy_val}


def test_component_alignment(
    model_path, params_path, n_atom_basis, n_gaussians, cutoff
):
    """Test individual components match between PyTorch and Paddle."""
    print("\n=== Component-Level Alignment ===")

    # Load PyTorch model
    pt_model = torch.load(model_path, map_location="cpu", weights_only=False)
    if hasattr(pt_model, "module"):
        pt_model = pt_model.module
    pt_state = pt_model.state_dict()

    # Load Paddle params
    pd_state = paddle.load(params_path)

    # Test 1: Embedding
    print("\n1. Embedding weight alignment:")
    pt_emb = pt_state["representation.embedding.weight"].numpy()
    pd_emb = pd_state["embedding.weight"].numpy()
    max_diff = np.max(np.abs(pt_emb - pd_emb))
    print(f"   Max diff: {max_diff:.2e} (should be < 1e-6)")

    # Test 2: RBF offsets
    print("\n2. RBF offsets alignment:")
    pd_rbf = GaussianRBF(n_gaussians=n_gaussians, cutoff=cutoff)
    pd_rbf_offsets = pd_rbf.offsets.numpy()
    pt_rbf_offsets = np.linspace(0, cutoff, n_gaussians).astype(np.float32)
    max_diff = np.max(np.abs(pd_rbf_offsets - pt_rbf_offsets))
    print(f"   Max diff: {max_diff:.2e} (should be < 1e-6)")

    # Test 3: shifted_softplus
    print("\n3. shifted_softplus alignment:")
    test_input = paddle.to_tensor(np.array([-1.0, 0.0, 1.0, 5.0], dtype=np.float32))
    pd_ssp = shifted_softplus(test_input).numpy()
    import torch.nn.functional as F

    pt_ssp = (F.softplus(torch.tensor([-1.0, 0.0, 1.0, 5.0])) - np.log(2.0)).numpy()
    max_diff = np.max(np.abs(pd_ssp - pt_ssp))
    print(f"   Max diff: {max_diff:.2e} (should be < 1e-6)")

    # Test 4: Filter network weight alignment (interaction 0)
    print("\n4. Filter network weights (interaction 0):")
    pt_fn0_w = (
        pt_state["representation.interactions.0.filter_network.0.weight"].numpy().T
    )
    pd_fn0_w = pd_state["interactions.0.cfconv.filter_net.0.weight"].numpy()
    max_diff = np.max(np.abs(pt_fn0_w - pd_fn0_w))
    print(f"   Max diff: {max_diff:.2e} (should be < 1e-6)")

    # Test 5: Output network weights
    print("\n5. Output network weights:")
    pt_out0_w = pt_state["output_modules.0.out_net.1.out_net.0.weight"].numpy().T
    pd_out0_w = pd_state["output_network.0.weight"].numpy()
    max_diff = np.max(np.abs(pt_out0_w - pd_out0_w))
    print(f"   Max diff: {max_diff:.2e} (should be < 1e-6)")

    print("\n=== Component tests done ===")


def main():
    parser = argparse.ArgumentParser(description="SchNet forward alignment test")
    parser.add_argument(
        "--torch_model",
        type=str,
        default="pretrained_weights/trained_schnet_models/qm9_energy_U0/best_model",
        help="Path to SchNetPack v0.3 pretrained model",
    )
    parser.add_argument(
        "--paddle_params",
        type=str,
        default="checkpoints/schnet_qm9_U0/schnet_qm9_U0.pdparams",
        help="Path to converted Paddle parameters",
    )
    parser.add_argument(
        "--n_atom_basis", type=int, default=128, help="Atom feature dimension"
    )
    parser.add_argument(
        "--n_interactions", type=int, default=6, help="Number of interaction blocks"
    )
    parser.add_argument(
        "--n_gaussians", type=int, default=50, help="Number of Gaussian RBF basis"
    )
    parser.add_argument("--cutoff", type=float, default=10.0, help="Cutoff distance")
    args = parser.parse_args()

    print("=" * 60)
    print("SchNet Forward Alignment Test")
    print("=" * 60)

    mol = create_test_molecule()
    print(f"\nTest molecule: {len(mol['atom_types'])} atoms (water)")
    print(f"  Atoms: {mol['atom_types']}")
    print(f"  Positions:\n{mol['positions']}")

    # Component alignment
    test_component_alignment(
        args.torch_model,
        args.paddle_params,
        args.n_atom_basis,
        args.n_gaussians,
        args.cutoff,
    )

    # PyTorch forward
    print("\n=== PyTorch (SchNetPack v0.3) Forward ===")
    pt_result = run_pytorch_forward(args.torch_model, mol)

    # Paddle forward
    print("\n=== Paddle (PaddleMaterials) Forward ===")
    pd_result = run_paddle_forward(
        args.paddle_params,
        mol,
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        n_gaussians=args.n_gaussians,
        cutoff=args.cutoff,
    )

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    pd_energy = pd_result["energy"].numpy().flatten()
    print(f"  Paddle energy:  {pd_energy}")

    # Extract PyTorch energy
    for key, val in pt_result.items():
        if isinstance(val, torch.Tensor):
            pt_energy = val.detach().numpy().flatten()
            diff = np.abs(pt_energy - pd_energy)
            print(f"  PyTorch energy ({key}): {pt_energy}")
            print(f"  Abs diff: {diff}")
            if np.max(diff) < 1e-4:
                print("  ✅ PASS (max diff < 1e-4)")
            elif np.max(diff) < 1e-2:
                print("  ⚠️  WARN (max diff < 1e-2, acceptable for float32)")
            else:
                print("  ❌ FAIL (max diff >= 1e-2)")


if __name__ == "__main__":
    main()
