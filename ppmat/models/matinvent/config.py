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
MatInvent module configuration registration.

This module registers MatInvent models with ppmat's model registry,
enabling reuse of ppmat's unified model loading and training infrastructure.
"""

import os
from typing import Optional

from omegaconf import OmegaConf

from ppmat.utils import logger as ppmat_logger


def register_matinvent_models():
    """Register MatInvent models with ppmat's model registry.

    This allows MatInvent models to be loaded using the same patterns
    as other ppmat models (DiffCSP, MatterGen, etc.).
    """
    # Check if models are already registered
    if hasattr(register_matinvent_models, "_registered"):
        return

    register_matinvent_models._registered = True


def load_matinvent_config(
    model_type: str,
    config_path: Optional[str] = None,
) -> OmegaConf:
    """Load MatInvent model configuration.

    This follows the same pattern as ppmat.models.load_model_from_config.

    Args:
        model_type: Type of matinvent model ("mattergen" or "diffcsp")
        config_path: Optional path to config file

    Returns:
        OmegaConf configuration object
    """
    if config_path is None:
        # Use default config locations
        config_candidates = [
            f"structure_generation/configs/matinvent/matinvent_{model_type}.yaml",
            f"structure_generation/configs/rl/{model_type}_rl.yaml",
        ]
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path is None:
        raise FileNotFoundError(
            f"MatInvent config not found for {model_type}. "
            f"Checked: {config_candidates}"
        )

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    ppmat_logger.info(f"MatInvent config loaded from {config_path}")
    return config


def build_matinvent_suite(
    model_type: str,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """Build MatInvent model suite using ppmat patterns.

    Args:
        model_type: Type of matinvent model ("mattergen" or "diffcsp")
        config_path: Optional path to config file
        model_path: Optional path to model weights
        device: Device to run the model on

    Returns:
        Model suite object
    """
    from ppmat.models.matinvent.rl.models.diffcsp_suite import DiffCSPSuite
    from ppmat.models.matinvent.rl.models.mattergen_suite import MatterGenSuite

    # Load configuration
    config = load_matinvent_config(model_type, config_path)

    # Extract model suite config
    suite_config = config.get("ModelSuite", config)
    model_config = config.get("Model", {})

    # Build model suite
    if model_type == "mattergen":
        model_suite = MatterGenSuite(
            model_name=model_config.get("name", "mattergen"),
            sample_cfg=suite_config.get("sample_cfg", {}),
            finetune_cfg=suite_config.get("finetune_cfg", {}),
            model_path=model_path or model_config.get("checkpoint_path"),
            device=device or model_config.get("device"),
        )
    elif model_type == "diffcsp":
        model_suite = DiffCSPSuite(
            model_name=model_config.get("name", "diffcsp"),
            sample_cfg=suite_config.get("sample_cfg", {}),
            finetune_cfg=suite_config.get("finetune_cfg", {}),
            model_path=model_path or model_config.get("checkpoint_path"),
            device=device or model_config.get("device"),
        )
    else:
        raise ValueError(f"Unknown MatInvent model type: {model_type}")

    ppmat_logger.info(f"MatInvent {model_type} suite built successfully")
    return model_suite


def setup_matinvent_training(
    config_path: str,
    model_type: str,
    output_dir: str,
):
    """Setup MatInvent training using ppmat infrastructure.

    This function replaces rl_train.py's main() function with a simpler
    interface that reuses ppmat's configuration and model loading patterns.

    Args:
        config_path: Path to training configuration file
        model_type: Type of matinvent model
        output_dir: Output directory for training results

    Returns:
        Dictionary containing configured components
    """
    # Load config
    config = load_matinvent_config(model_type, config_path)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["models", "samples", "rewards", "logs"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Setup logging
    from ppmat.models.matinvent.rl.utils import setup_rl_logger

    log_file = os.path.join(output_dir, "logs", "training.log")
    setup_rl_logger(log_file=log_file)

    # Build model suite
    model_suite = build_matinvent_suite(
        model_type=model_type,
        config_path=config_path,
        device=config.get("Global", {}).get("device", None),
    )

    # Build reward function
    from ppmat.models.matinvent.rewards.reward import Reward

    reward_cfg = config.get("RL", {}).get("reward_cfg", {})
    reward = Reward(
        root_dir=os.path.join(output_dir, "rewards"),
        prop_cfg=reward_cfg.get("prop_cfg", []),
        reward_threshold=reward_cfg.get("reward_threshold", 0.5),
        reduce=reward_cfg.get("reduce", "mean"),
    )

    # Create MatInvent instance
    from ppmat.models.matinvent.rl.mat_invent import MatInvent

    rl_config = config.get("RL", {})
    mat_invent = MatInvent(
        rl_epoch=rl_config.get("rl_epoch", 100),
        model_suite=model_suite,
        reward=reward,
        sample_cfg=rl_config.get("sample_cfg", {}),
        finetune_cfg=rl_config.get("finetune_cfg", {}),
        topk_ratio=rl_config.get("topk_ratio", 0.5),
        save_dir=output_dir,
        save_freq=rl_config.get("save_freq", 50),
        device=config.get("Global", {}).get("device", None),
        logger=ppmat_logger._logger,
        replay=rl_config.get("replay_cfg") is not None,
        replay_args=rl_config.get("replay_cfg", {}),
        div_filter=rl_config.get("div_filter_cfg", {}).get("enabled", False),
        df_args=rl_config.get("div_filter_cfg", {}),
    )

    return {
        "config": config,
        "model_suite": model_suite,
        "reward": reward,
        "mat_invent": mat_invent,
    }


# Prevent re-registration
register_matinvent_models._registered = False
