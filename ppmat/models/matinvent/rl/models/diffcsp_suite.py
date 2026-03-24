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
DiffCSP model suite for reinforcement learning.


"""

import os
from pathlib import Path
from typing import List

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rl.datasets import create_rl_dataloader
from ppmat.models.matinvent.rl.models.base import ModelSuite
from ppmat.models.matinvent.rl.samplers import DiffCSPSampler as Sampler

# DiffCSP mp-20 standard model configuration
# (参照 structure_generation/configs/diffcsp/diffcsp_mp20.yaml)
_DIFFCSP_DEFAULT_CFG = dict(
    decoder_cfg=dict(
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=128,
        edge_style="fc",
        ln=True,
        ip=True,
        smooth=False,
        pred_type=False,
        prop_dim=512,
        pred_scalar=False,
        num_classes=100,
    ),
    lattice_noise_scheduler_cfg={
        "__class_name__": "DDPMScheduler",
        "__init_params__": {
            "beta_schedule": "squaredcos_cap_v2",
            "num_train_timesteps": 1000,
            "clip_sample": False,
        },
    },
    coord_noise_scheduler_cfg={
        "__class_name__": "ScoreSdeVeSchedulerWrapped",
        "__init_params__": {
            "num_train_timesteps": 1000,
            "sigma_min": 0.005,
            "sigma_max": 0.5,
            "snr": 1e-5,
        },
    },
    num_train_timesteps=1000,
    time_dim=256,
    lattice_loss_weight=1.0,
    coord_loss_weight=1.0,
)


class DiffCSPSuite(ModelSuite):
    """DiffCSP model suite for RL training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        """Load DiffCSP model from checkpoint.

        The checkpoint (.pdparams) stores only CSPNet (decoder) weights
        (keys have no 'decoder.' prefix). This method adds the prefix before
        loading into the full DiffCSP model so the schedulers are also
        initialised correctly.

        Uses self.model_path as the .pdparams checkpoint file.
        Supports both absolute and workspace-relative paths.

        Returns:
            Loaded DiffCSP model in eval mode.

        Original reference:
        """
        from ppmat.models.diffcsp.diffcsp import DiffCSP

        if self.model_path is None:
            raise ValueError(
                "model_path must be specified for DiffCSPSuite.load_model(). "
                "Pass the path to the .pdparams checkpoint file."
            )

        ckpt_path = Path(self.model_path).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = DiffCSP(**_DIFFCSP_DEFAULT_CFG)

        # The checkpoint contains bare CSPNet weights; DiffCSP wraps it as
        # self.decoder, so we prepend 'decoder.' to each key.
        raw_sd = paddle.load(str(ckpt_path))
        state_dict = {"decoder." + k: v for k, v in raw_sd.items()}
        model.set_state_dict(state_dict)
        model.eval()
        return model

    def get_sampler(self):
        """Get sampler for DiffCSP generation."""
        return Sampler(
            batch_size=self.sample_cfg.get("batch_size", 16),
            num_batches=self.sample_cfg.get("num_batches", 4),
            num_inference_steps=self.sample_cfg.get("num_inference_steps", 1000),
        )

    def get_dataloader(
        self,
        samples: List[Structure],
        rewards: np.ndarray,
        batch_size: int = 8,
    ):
        """Create dataloader from samples and rewards."""
        return create_rl_dataloader(
            structures=samples,
            rewards=rewards,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    def save_model(self, model, ckpt_dir: str):
        """Save model checkpoint."""
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))
