#!/usr/bin/env python3
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

"""Convert SchNetPack v0.3 pretrained weights to PaddleMaterials SchNet format.

Usage:
    python tools/convert_schnet_weights.py \
        --input trained_schnet_models/qm9_energy_U0/best_model \
        --output checkpoints/schnet_qm9_U0.pdparams

The pretrained models are from:
    https://www.quantum-machine.org/datasets/trained_schnet_models.zip
"""

import argparse
import os

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def load_torch_state_dict(path: str) -> dict:
    """Load PyTorch state dict from SchNetPack model file."""
    if torch is None:
        raise RuntimeError("PyTorch is required for conversion: pip install torch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # SchNetPack v0.3 saves the full model object, not a state dict
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def convert_schnet_weights(torch_state: dict, n_interactions: int = 6) -> dict:
    """Convert SchNetPack v0.3 state dict to PaddleMaterials SchNet format.

    Key mapping:
        SchNetPack v0.3                          → PaddleMaterials SchNet
        representation.embedding.weight          → embedding.weight
        representation.distance_expansion.*      → rbf.*
        representation.interactions.{i}.*        → interactions.{i}.*
        output_modules.0.*                       → output_network.* / data_mean / data_std

    Linear weight convention:
        PyTorch: [out_features, in_features]
        Paddle:  [in_features, out_features]   → transpose needed
    """
    paddle_state = {}

    for key, value in torch_state.items():
        arr = value.numpy() if hasattr(value, "numpy") else np.array(value)

        # === Embedding ===
        if key == "representation.embedding.weight":
            paddle_state["embedding.weight"] = arr
            continue

        # === RBF distance expansion (buffers) ===
        if key == "representation.distance_expansion.offset":
            paddle_state["rbf.offsets"] = arr.flatten()
            continue
        if key == "representation.distance_expansion.offsets":
            paddle_state["rbf.offsets"] = arr.flatten()
            continue
        if key == "representation.distance_expansion.width":
            # SchNetPack stores width per gaussian; we store a scalar
            paddle_state["rbf.widths"] = arr.flatten()[0]
            continue

        # === Atom reference energies (optional) ===
        if key == "output_modules.0.atomref.weight":
            # Per-atom type reference energies, not used in our model
            # but stored for completeness
            paddle_state["atomref"] = arr
            continue

        # === Interaction blocks ===
        matched = False
        for i in range(n_interactions):
            prefix = f"representation.interactions.{i}"

            # Skip duplicate filter_network at top level (same as cfconv.filter_network)
            if key.startswith(f"{prefix}.filter_network."):
                matched = True
                break
            # Skip cutoff_network buffers (we use analytic cutoff)
            if key.startswith(f"{prefix}.cutoff_network.") or key.startswith(
                f"{prefix}.cfconv.cutoff_network."
            ):
                matched = True
                break

            if not key.startswith(f"{prefix}."):
                continue

            matched = True
            suffix = key[len(prefix) + 1 :]  # e.g., "cfconv.in2f.weight"

            # Map cfconv layers
            if suffix == "cfconv.in2f.weight":
                paddle_state[f"interactions.{i}.cfconv.in2f.weight"] = arr.T
            elif suffix == "cfconv.f2out.weight":
                paddle_state[f"interactions.{i}.cfconv.f2out.weight"] = arr.T
            elif suffix == "cfconv.f2out.bias":
                paddle_state[f"interactions.{i}.cfconv.f2out.bias"] = arr
            elif suffix == "cfconv.filter_network.0.weight":
                paddle_state[f"interactions.{i}.cfconv.filter_net.0.weight"] = arr.T
            elif suffix == "cfconv.filter_network.0.bias":
                paddle_state[f"interactions.{i}.cfconv.filter_net.0.bias"] = arr
            elif suffix == "cfconv.filter_network.1.weight":
                paddle_state[f"interactions.{i}.cfconv.filter_net.2.weight"] = arr.T
            elif suffix == "cfconv.filter_network.1.bias":
                paddle_state[f"interactions.{i}.cfconv.filter_net.2.bias"] = arr
            # Map dense layer
            elif suffix == "dense.weight":
                paddle_state[f"interactions.{i}.dense.weight"] = arr.T
            elif suffix == "dense.bias":
                paddle_state[f"interactions.{i}.dense.bias"] = arr
            else:
                print(f"  [WARN] Unmapped interaction key: {key}")
            break

        if matched:
            continue

        # === Output network ===
        if key == "output_modules.0.out_net.1.out_net.0.weight":
            paddle_state["output_network.0.weight"] = arr.T
        elif key == "output_modules.0.out_net.1.out_net.0.bias":
            paddle_state["output_network.0.bias"] = arr
        elif key == "output_modules.0.out_net.1.out_net.1.weight":
            paddle_state["output_network.1.weight"] = arr.T
        elif key == "output_modules.0.out_net.1.out_net.1.bias":
            paddle_state["output_network.1.bias"] = arr
        elif key == "output_modules.0.standardize.mean":
            paddle_state["data_mean"] = arr.flatten()[0]
        elif key == "output_modules.0.standardize.stddev":
            paddle_state["data_std"] = arr.flatten()[0]
        else:
            print(f"  [SKIP] Unrecognized key: {key} (shape={arr.shape})")

    return paddle_state


def save_paddle_params(state: dict, output_path: str):
    """Save converted state dict as .pdparams (numpy-based)."""
    import paddle

    paddle_state = {}
    for k, v in state.items():
        paddle_state[k] = v if isinstance(v, np.ndarray) else np.array(v)
    paddle.save(paddle_state, output_path)
    print(f"Saved {len(paddle_state)} parameters to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SchNetPack v0.3 weights to PaddleMaterials format"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to SchNetPack model file (e.g., best_model)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for .pdparams file"
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=6,
        help="Number of interaction blocks (default: 6)",
    )
    args = parser.parse_args()

    print(f"Loading PyTorch state dict from: {args.input}")
    torch_state = load_torch_state_dict(args.input)
    print(f"  Found {len(torch_state)} entries")

    print("Converting weights...")
    paddle_state = convert_schnet_weights(torch_state, args.n_interactions)
    print(f"  Converted {len(paddle_state)} entries")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_paddle_params(paddle_state, args.output)

    # Print summary
    print("\n=== Conversion Summary ===")
    for k, v in sorted(paddle_state.items()):
        print(f"  {k:55s} {str(v.shape):>20s}")


if __name__ == "__main__":
    main()
