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
MatInvent model module.

This module contains the MatInvent material generation model and its associated
components for reinforcement learning based material discovery.

Components:
- memory: Long-term memory and replay buffer for RL
- rewards: Reward system and property calculators
- common: Shared utilities that reuse ppmat built-in functionality

Note: RL compatibility for MatterGen is now provided through the
MatterGenRLAdapter class instead of monkey patching.
"""

from ppmat.models.matinvent.common import load_config
from ppmat.models.matinvent.memory import LongTimeMem
from ppmat.models.matinvent.memory import ReplayBuffer
from ppmat.models.matinvent.rewards import Calculator
from ppmat.models.matinvent.rewards import Reward
from ppmat.models.matinvent.rl import MatInvent

__all__ = [
    "MatInvent",
    "load_config",
    "ReplayBuffer",
    "LongTimeMem",
    "Reward",
    "Calculator",
]
