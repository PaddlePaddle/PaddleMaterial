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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Electron-density field prediction")
    # Select exactly one model source: a registered package or a local config.
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_name", help="Registered model name.")
    model_source.add_argument("--config_path", help="Path to a local config file.")
    parser.add_argument("--weights_name", help="Weight filename in a model package.")
    parser.add_argument(
        "--checkpoint_path",
        help="Checkpoint path or URL; defaults to Predict.checkpoint_path in config.",
    )
    parser.add_argument("--device", help="Paddle device, for example cpu or gpu:0.")

    parser.add_argument(
        "--mol_file_path",
        help="Path to one MOL file or a directory containing MOL files.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "validation", "test"],
        help="Configured dataset split used when --mol_file_path is omitted.",
    )
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="Sample index in the configured dataset split.",
    )
    parser.add_argument("--data_root", help="Override the configured dataset root.")
    parser.add_argument(
        "--split_file_path",
        help="Override the configured dataset split-file path.",
    )

    parser.add_argument(
        "--save_path",
        default="./results",
        help="Directory used for predicted CUBE files and visualizations.",
    )
    parser.add_argument(
        "--grid_batch_size",
        type=int,
        help="Maximum grid points per forward pass; defaults to Predict config.",
    )
    parser.add_argument(
        "--grid_shape",
        default="80,80,80",
        help="MOL grid shape as N or Nx,Ny,Nz.",
    )
    parser.add_argument(
        "--grid_padding",
        default=6.0,
        type=float,
        help="Padding around MOL coordinates in Angstrom.",
    )
    parser.add_argument(
        "--reference_cube_dir",
        help="Directory containing optional reference CUBE files for MOL inputs.",
    )
    parser.add_argument(
        "--save_true_cube",
        action="store_true",
        help="Also save the reference density when it is available.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write density visualizations as PNG files.",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="Also write interactive HTML visualizations.",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display generated Plotly figures.",
    )
    args, config_overrides = parser.parse_known_args(argv)
    if args.model_name is not None and args.checkpoint_path is not None:
        parser.error("--checkpoint_path cannot be combined with --model_name")
    if args.config_path is not None and args.weights_name is not None:
        parser.error("--weights_name can only be used with --model_name")
    if any(value.startswith("-") or "=" not in value for value in config_overrides):
        parser.error("unrecognized arguments: " + " ".join(config_overrides))
    return args, config_overrides


def main():
    args, config_overrides = parse_args()
    predictor = FieldPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        config_overrides=config_overrides,
    )
    output_options = {
        "save_path": args.save_path,
        "grid_batch_size": args.grid_batch_size,
        "save_true_cube": args.save_true_cube,
        "visualize": args.visualize,
        "save_html": args.save_html,
        "show_plot": args.show_plot,
    }
    if args.mol_file_path is not None:
        predictor.from_mol_file(
            mol_file_path=args.mol_file_path,
            grid_shape=args.grid_shape,
            grid_padding=args.grid_padding,
            reference_cube_dir=args.reference_cube_dir,
            **output_options,
        )
    else:
        predictor.from_dataset(
            split=args.split,
            index=args.index,
            data_root=args.data_root,
            split_file_path=args.split_file_path,
            **output_options,
        )


if __name__ == "__main__":
    main()
