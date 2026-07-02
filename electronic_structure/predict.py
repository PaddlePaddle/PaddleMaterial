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

import argparse

from ppmat.predictor import FieldPredictor
from ppmat.predictor.field_predictor import apply_predict_config

__all__ = ["FieldPredictor", "apply_predict_config", "build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(description="Electron density field inference")
    parser.add_argument(
        "--model_name",
        default=None,
        help=(
            "Registered model name from MODEL_REGISTRY. When set, config and "
            "checkpoint are resolved automatically."
        ),
    )
    parser.add_argument(
        "--weights_name",
        default=None,
        help="Optional weight filename inside a registered model package.",
    )
    parser.add_argument(
        "--config",
        default="electronic_structure/configs/infgcn/infgcn_qm9.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/infgcn_qm9_best/infgcn_qm9.pdparams",
        help="Checkpoint (.pdparams) to load",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test"],
        help="Dataset split to sample from",
    )
    parser.add_argument(
        "--index",
        default=None,
        type=int,
        help="Index within the chosen split",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override dataset root; defaults to value in config",
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="Override split file path; defaults to value in config",
    )
    parser.add_argument(
        "--atom_file",
        default=None,
        help="Override atom info file; defaults to value in config",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store predictions/visualizations",
    )
    parser.add_argument(
        "--grid_batch_size",
        default=None,
        type=int,
        help="Number of grid points per forward pass",
    )
    parser.add_argument(
        "--skip_vis",
        action="store_true",
        default=None,
        help="Skip writing/visualizing density plots",
    )
    parser.add_argument(
        "--save_true_cube",
        action="store_true",
        default=None,
        help="Save reference (DFT) electron density as a cube file",
    )
    parser.add_argument(
        "--save_pred_cube",
        action="store_true",
        default=None,
        help="Save predicted electron density as a cube file",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        default=None,
        help="Save Plotly figures as interactive HTML (in addition to PNG)",
    )
    parser.add_argument(
        "--cube_dir",
        default=None,
        help="Directory to store cube files (defaults to output_dir)",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        default=None,
        help="Display plotly figures inline (requires kaleido)",
    )
    parser.add_argument(
        "--mol_input",
        default=None,
        help=(
            "Path to a .mol file or a directory of .mol files for direct "
            "structure inference"
        ),
    )
    parser.add_argument(
        "--mol_pattern",
        default=None,
        help="Glob pattern when --mol_input is a directory",
    )
    parser.add_argument(
        "--mol_grid_shape",
        default=None,
        help="Grid shape for MOL inference, e.g. '80' or '80,80,80'",
    )
    parser.add_argument(
        "--mol_grid_padding",
        default=None,
        type=float,
        help="Padding (Angstrom) around molecular coordinates for MOL grid generation",
    )
    parser.add_argument(
        "--mol_true_cube_dir",
        default=None,
        help=(
            "Optional directory containing reference/true CUBE files for MOL inputs. "
            "Expected names: <mol_basename>.cube or <mol_basename>_true.cube"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    predictor = FieldPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    predictor.predict(args)


if __name__ == "__main__":
    main()
