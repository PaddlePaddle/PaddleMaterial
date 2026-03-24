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
RL dataset for fine-tuning diffusion models during reinforcement learning.



Provides dataset classes for loading samples and rewards for RL fine-tuning.
Now reuses data_utils for common functionality.
"""

# Import from data_utils for code reuse
from ppmat.models.matinvent.rl.data_utils import RLDataset
from ppmat.models.matinvent.rl.data_utils import collate_fn
from ppmat.models.matinvent.rl.data_utils import create_rl_dataloader
from ppmat.models.matinvent.rl.data_utils import filter_by_reward

# Re-export for backward compatibility
__all__ = [
    "RLDataset",
    "collate_fn",
    "create_rl_dataloader",
    "filter_by_reward",
]
