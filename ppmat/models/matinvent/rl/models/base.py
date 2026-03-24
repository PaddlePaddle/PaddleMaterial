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

from typing import Literal

from omegaconf import DictConfig
from omegaconf import OmegaConf

from ppmat.models.matinvent.rl.utils import get_device

AVA_MODEL_NAME = Literal[
    "diffcsp",
    "mattergen_base",
    "mattergen_chemical_system",
    "mattergen_space_group",
    "mattergen_dft_mag_density",
    "mattergen_dft_band_gap",
    "mattergen_ml_bulk_modulus",
    "mattergen_dft_mag_density_hhi_score",
    "mattergen_chemical_system_energy_above_hull",
]


class ModelSuite:
    """Base class for model suite in reinforcement learning."""

    def __init__(
        self,
        model_name: str,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.model_path = model_path
        self.config_overrides = config_overrides
        self.device = get_device(device)
        self.cfg = OmegaConf.create(kwargs)

    def load_model(self):
        """Load model from checkpoint."""
        raise NotImplementedError

    def get_sampler(self):
        """Get sampler for generation."""
        raise NotImplementedError

    def get_dataloader(self):
        """Get dataloader for training."""
        raise NotImplementedError

    def save_model(self):
        """Save model checkpoint."""
        raise NotImplementedError
