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
Common utilities for MatInvent module.

This module provides common utilities that reuse ppmat built-in functionality.
"""

import os

from omegaconf import OmegaConf

from ppmat.utils import logger as ppmat_logger


def setup_matinvent_logging(
    output_dir: str,
    log_name: str = "matinvent_rl",
    log_level: int = 20,
):
    """Setup logging for MatInvent training using ppmat logger.

    This replaces the custom setup_logging in rl_train.py
    with ppmat's standard logging infrastructure.

    Args:
        output_dir: Directory to save log file
        log_name: Name of the logger
        log_level: Logging level (default: INFO=20)
    """
    log_file = os.path.join(output_dir, "training.log")
    ppmat_logger.init_logger(name=log_name, log_file=log_file, log_level=log_level)
    return ppmat_logger._logger


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file.

    This follows the same pattern as ppmat.models.load_model_from_config
    and ppmat.sampler.base_sampler.MolecularSampler.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    ppmat_logger.info(f"Configuration loaded from {config_path}")
    return config


def build_model_from_config(cfg, model_type: str):
    """Build model from configuration.

    This follows the same pattern as ppmat.models.build_model.

    Args:
        cfg: Configuration object
        model_type: Type of model to build ("mattergen" or "diffcsp")

    Returns:
        Model suite object
    """
    if model_type == "mattergen":
        from ppmat.models.matinvent.rl.models.mattergen_suite import MatterGenSuite

        return MatterGenSuite(
            model_name=model_type,
            sample_cfg=cfg.get("sample_cfg", {}),
            finetune_cfg=cfg.get("finetune_cfg", {}),
            model_path=cfg.get("model_path", None),
            device=cfg.get("device", None),
        )
    elif model_type == "diffcsp":
        from ppmat.models.matinvent.rl.models.diffcsp_suite import DiffCSPSuite

        return DiffCSPSuite(
            model_name=model_type,
            sample_cfg=cfg.get("sample_cfg", {}),
            finetune_cfg=cfg.get("finetune_cfg", {}),
            model_path=cfg.get("model_path", None),
            device=cfg.get("device", None),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def prepare_output_dir(output_dir: str) -> str:
    """Prepare output directory, creating if needed.

    Args:
        output_dir: Path to output directory

    Returns:
        Absolute path to output directory
    """
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ["models", "samples", "rewards"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    ppmat_logger.info(f"Output directory prepared: {output_dir}")
    return output_dir
