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

"""
Reinforcement learning training script for material generation.

This code is adapted from:
"""

import argparse
import os

from omegaconf import OmegaConf

from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl.mat_invent import MatInvent
from ppmat.models.matinvent.rl.models.diffcsp_suite import DiffCSPSuite
from ppmat.models.matinvent.rl.models.mattergen_suite import MatterGenSuite
from ppmat.utils import logger as ppmat_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reinforcement learning training for material generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="structure_generation/configs/rl/mattergen_rl.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mattergen",
        choices=["mattergen", "diffcsp"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (gpu/cpu)",
    )
    return parser.parse_args()


def setup_logging(output_dir: str):
    log_file = os.path.join(output_dir, "training.log")
    ppmat_logger.init_logger(name="matinvent_rl", log_file=log_file, log_level=20)
    return ppmat_logger._logger


def build_model_suite(cfg, model_name: str, device: str):
    """Build model suite based on configuration."""
    if model_name == "mattergen":
        return MatterGenSuite(
            model_name=model_name,
            sample_cfg=cfg.RL.sample_cfg,
            finetune_cfg=cfg.RL.finetune_cfg,
            model_path=cfg.get("model_path", None),
            device=device,
        )
    elif model_name == "diffcsp":
        return DiffCSPSuite(
            model_name=model_name,
            sample_cfg=cfg.RL.sample_cfg,
            finetune_cfg=cfg.RL.finetune_cfg,
            model_path=cfg.get("model_path", None),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_reward(cfg, output_dir: str):
    """Build reward function based on configuration."""
    reward_cfg = cfg.RL.reward_cfg
    return Reward(
        root_dir=os.path.join(output_dir, "rewards"),
        prop_cfg=reward_cfg.prop_cfg,
        reward_threshold=reward_cfg.reward_threshold,
        reduce=reward_cfg.reduce,
    )


def main():
    """Main training function."""
    args = parse_args()

    cfg = OmegaConf.load(args.config)

    if args.output_dir is not None:
        cfg.Global.output_dir = args.output_dir

    os.makedirs(cfg.Global.output_dir, exist_ok=True)

    logger = setup_logging(cfg.Global.output_dir)
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Output directory: {cfg.Global.output_dir}")

    logger.info("Building model suite...")
    model_suite = build_model_suite(cfg, args.model, args.device)

    logger.info("Building reward function...")
    reward = build_reward(cfg, cfg.Global.output_dir)

    logger.info("Initializing MatInvent pipeline...")
    mat_invent = MatInvent(
        rl_epoch=cfg.RL.rl_epoch,
        model_suite=model_suite,
        reward=reward,
        sample_cfg={},
        finetune_cfg={},
        topk_ratio=cfg.RL.topk_ratio,
        save_dir=cfg.Global.output_dir,
        save_freq=cfg.RL.save_freq,
        device=args.device,
        logger=logger,
        replay=cfg.RL.get("replay_cfg", None) is not None,
        replay_args=cfg.RL.get("replay_cfg", {}),
        div_filter=cfg.RL.div_filter_cfg.enabled,
        df_args=cfg.RL.div_filter_cfg,
    )

    logger.info("Starting reinforcement learning training...")
    mat_invent.run_rl()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
