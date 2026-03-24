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
MatterGen model suite for reinforcement learning.


"""

import os
from pathlib import Path
from typing import List

import numpy as np
import paddle
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rl.datasets import create_rl_dataloader
from ppmat.models.matinvent.rl.models.base import ModelSuite
from ppmat.models.matinvent.rl.models.mattergen_adapter import create_matinvent_adapter
from ppmat.models.matinvent.rl.samplers import MatterGenSampler as Sampler

# Standard MatterGen mp-20 model configuration
_MATTERGEN_DEFAULT_CFG = dict(
    decoder_cfg={
        "gemnet_cfg": {
            "num_targets": 1,
            "latent_dim": 512,
            "atom_embedding_cfg": {
                "emb_size": 512,
                "with_mask_type": True,
            },
            "max_neighbors": 50,
            "max_cell_images_per_dim": 5,
            "cutoff": 7.0,
            "num_blocks": 4,
            "otf_graph": True,
        }
    },
    lattice_noise_scheduler_cfg={
        "__class_name__": "LatticeVPSDEScheduler",
        "limit_density": 0.05771451654022283,
        "__init_params__": {},
    },
    coord_noise_scheduler_cfg={
        "__class_name__": "NumAtomsVarianceAdjustedWrappedVESDE",
        "__init_params__": {},
    },
    atom_noise_scheduler_cfg={"__class_name__": "D3PMScheduler", "__init_params__": {}},
    num_train_timesteps=1000,
    time_dim=256,
    lattice_loss_weight=1,
    coord_loss_weight=0.1,
    atom_loss_weight=1,
)


class MatterGenSuite(ModelSuite):
    """MatterGen model suite for RL training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        """Load MatterGen model from checkpoint.

        Uses self.model_path as the .pdparams checkpoint file.
        Supports both absolute paths and paths relative to the workspace root.

        Returns:
            RL-adapted MatterGen model (wrapped in MatterGenRLAdapter).
        """
        from ppmat.models.matinvent.mattergen_compat import MatinventMatterGen

        if self.model_path is None:
            raise ValueError(
                "model_path must be specified for MatterGenSuite.load_model(). "
                "Pass the path to the .pdparams checkpoint file."
            )

        ckpt_path = Path(self.model_path).expanduser()
        if not ckpt_path.is_absolute():
            # resolve relative to current working dir
            ckpt_path = Path.cwd() / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = MatinventMatterGen(**_MATTERGEN_DEFAULT_CFG)
        model.set_state_dict(paddle.load(str(ckpt_path)))
        model.eval()

        # Wrap the model with RL adapter for compatibility
        # This replaces the monkey patching approach
        return create_matinvent_adapter(model)

    def get_sampler(self):
        """Get sampler for MatterGen generation."""
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
