# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from ppmat.sampler.base_sampler import MolecularSampler
from ppmat.utils import logger

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    argparse.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    argparse.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    argparse.add_argument("--save_path", type=str, default="results")
    argparse.add_argument(
        "--mode",
        type=str,
        choices=[
            "by_dataloader",
            "compute_metric",
        ],
        default="by_dataloader",
    )

    args = argparse.parse_args()

    sampler = MolecularSampler(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    if args.mode == "compute_metric":
        metric_result = sampler.compute_metric(
            save_path=args.save_path,
        )
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    elif args.mode == "by_dataloader":
        result = sampler.sample_by_dataloader(
            save_path=args.save_path,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
