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
Reinforcement learning module for material generation.

"""

from ppmat.models.matinvent.rl.base import ReinL
from ppmat.models.matinvent.rl.datasets import RLDataset
from ppmat.models.matinvent.rl.datasets import collate_fn
from ppmat.models.matinvent.rl.datasets import create_rl_dataloader
from ppmat.models.matinvent.rl.mat_invent import MatInvent
from ppmat.models.matinvent.rl.models.mattergen_adapter import MatterGenRLAdapter
from ppmat.models.matinvent.rl.models.mattergen_adapter import create_matinvent_adapter
from ppmat.models.matinvent.rl.samplers import BaseSampler
from ppmat.models.matinvent.rl.samplers import DiffCSPSampler
from ppmat.models.matinvent.rl.samplers import MatterGenSampler
from ppmat.models.matinvent.rl.training_utils import is_valid_structure
from ppmat.models.matinvent.rl.training_utils import save_structures
from ppmat.models.matinvent.rl.utils import create_optimizer
from ppmat.models.matinvent.rl.utils import get_device
from ppmat.models.matinvent.rl.utils import log_training_stats
from ppmat.models.matinvent.rl.utils import setup_rl_logger

__all__ = [
    "ReinL",
    "MatInvent",
    "BaseSampler",
    "MatterGenSampler",
    "DiffCSPSampler",
    "RLDataset",
    "collate_fn",
    "create_rl_dataloader",
    "get_device",
    "create_optimizer",
    "setup_rl_logger",
    "log_training_stats",
    "is_valid_structure",
    "save_structures",
    "MatterGenRLAdapter",
    "create_matinvent_adapter",
]
