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
Utility functions for MatInvent RL module.

This module provides common utilities that reuse ppmat built-in functionality.
"""

from typing import Optional

import paddle
import paddle.nn as nn

from ppmat.utils import logger as ppmat_logger


def get_device(device: Optional[str] = None) -> str:
    """Get device for PaddlePaddle.

    Args:
        device: Device string ('gpu', 'cpu', etc.). If None, auto-detect.

    Returns:
        Device string ('gpu' or 'cpu')
    """
    if device is None:
        if paddle.is_compiled_with_cuda():
            device = "gpu"
        else:
            device = "cpu"
    # Set the device and return the string
    paddle.set_device(device)
    return device


def create_optimizer(
    model: nn.Layer, lr: float = 5e-4, opt_type: str = "Adam", **kwargs
) -> paddle.optimizer.Optimizer:
    """Create optimizer using ppmat optimizer factory pattern.
    Args:
        model: Model to optimize
        lr: Learning rate
        opt_type: Optimizer type ('Adam', 'AdamW', etc.)
        **kwargs: Additional optimizer parameters (weight_decay, grad_clip, etc.)

    Returns:
        PaddlePaddle optimizer
    """
    from ppmat.optimizer.optimizer import Adam as PpmatAdam
    from ppmat.optimizer.optimizer import AdamW as PpmatAdamW

    if opt_type == "Adam":
        optimizer = PpmatAdam(learning_rate=lr, **kwargs)(model)
    elif opt_type == "AdamW":
        optimizer = PpmatAdamW(learning_rate=lr, **kwargs)(model)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer


def create_scheduler(
    optimizer: paddle.optimizer.Optimizer, scheduler_type: str = "LinearLR", **kwargs
) -> Optional[paddle.optimizer.lr.LRScheduler]:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Scheduler type
        **kwargs: Scheduler parameters (e.g., start_factor, total_iters)

    Returns:
        LRScheduler or None
    """
    if scheduler_type == "LinearLR":
        start_factor = kwargs.get("start_factor", 0.1)
        total_iters = kwargs.get("total_iters", 10)
        scheduler = paddle.optimizer.lr.LinearLR(
            learning_rate=optimizer.get_lr(),
            start_factor=start_factor,
            total_iters=total_iters,
        )
    elif scheduler_type is None or scheduler_type == "None":
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


def setup_rl_logger(log_file: Optional[str] = None, log_level: int = 20):
    """Setup logger for RL training using ppmat logger.

    Args:
        log_file: Optional log file path
        log_level: Logging level (default: INFO=20)
    """
    ppmat_logger.init_logger(
        name="matinvent_rl", log_file=log_file, log_level=log_level
    )


def log_training_stats(
    epoch: int,
    step: int,
    loss: float,
    reward_mean: float,
    reward_std: float,
    prefix: str = "RL",
):
    """Log training statistics using ppmat logger.

    Args:
        epoch: Current epoch
        step: Current step
        loss: Loss value
        reward_mean: Mean reward
        reward_std: Reward standard deviation
        prefix: Log prefix
    """
    ppmat_logger.info(
        f"[{prefix}] Epoch {epoch}, Step {step}: "
        f"loss={loss:.4f}, reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}"
    )
