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

"""
Convert CrystalLLM PyTorch checkpoint to Paddle format.

PyTorch checkpoints available at: https://zenodo.org/records/10642388

Usage:
    python convert_weights.py --input ckpt.pt --output model.pdparams

Notes:
    - PyTorch nn.Linear stores weight as [out_features, in_features]
    - Paddle nn.Linear stores weight as [in_features, out_features]
    - Therefore Linear weights must be transposed during conversion
    - Embedding weights are NOT transposed (same layout in both frameworks)
    - LayerNorm weights (1D) are NOT transposed
"""

import argparse

import paddle


def convert_pytorch_to_paddle(input_path, output_path):
    """Convert a CrystalLLM PyTorch checkpoint to Paddle format.

    Args:
        input_path: Path to PyTorch .pt checkpoint file.
        output_path: Path to save Paddle .pdparams file.
    """
    # Import torch only when needed (not a runtime dependency)
    import torch

    checkpoint = torch.load(input_path, map_location="cpu")
    pt_state = checkpoint["model"]

    paddle_state = {}
    for key, tensor in pt_state.items():
        # Strip torch.compile prefix: _orig_mod.transformer. or _orig_mod.
        clean_key = key
        if clean_key.startswith("_orig_mod.transformer."):
            clean_key = clean_key[len("_orig_mod.transformer."):]
        elif clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]

        # Skip lm_head.weight — our model uses weight tying via matmul with wte.weight
        if clean_key == "lm_head.weight":
            continue

        np_array = tensor.numpy()

        # Transpose Linear weight matrices (2D, not embedding/layernorm)
        # Linear weights in pytorch: [out, in], paddle: [in, out]
        # Skip embedding weights (wte.weight, wpe.weight) and 1D params
        needs_transpose = (
            np_array.ndim == 2 and "wte.weight" not in clean_key and "wpe.weight" not in clean_key
        )

        if needs_transpose:
            np_array = np_array.T

        # Map PyTorch key names to Paddle conventions
        # The model structure is identical, just framework prefix differences
        paddle_state[clean_key] = np_array

    paddle.save(paddle_state, output_path)
    print(f"Converted {len(paddle_state)} parameters")
    print(f"Saved to {output_path}")

    # Print model config from checkpoint
    if "model_args" in checkpoint:
        print(f"Model config: {checkpoint['model_args']}")
    if "iter_num" in checkpoint:
        print(f"Training iteration: {checkpoint['iter_num']}")
    if "best_val_loss" in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CrystalLLM PyTorch checkpoint to Paddle"
    )
    parser.add_argument("--input", required=True, help="Path to PyTorch .pt file")
    parser.add_argument(
        "--output", required=True, help="Path for output .pdparams file"
    )
    args = parser.parse_args()

    convert_pytorch_to_paddle(args.input, args.output)
