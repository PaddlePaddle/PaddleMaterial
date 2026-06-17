# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local compatibility shims for UMA without fairchem dependency."""

from . import gp_utils
from .base import HeadInterface
from .graph import generate_graph
from .inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    OutputSpec,
    Task,
    UMATask,
)
from .irreps import cg_change_mat, irreps_sum
from .registry import registry
from .utils import conditional_grad

__all__ = [
    "gp_utils",
    "HeadInterface",
    "generate_graph",
    "CHARGE_RANGE",
    "DEFAULT_CHARGE",
    "DEFAULT_SPIN",
    "DEFAULT_SPIN_OMOL",
    "SPIN_RANGE",
    "InferenceSettings",
    "OutputSpec",
    "Task",
    "UMATask",
    "cg_change_mat",
    "irreps_sum",
    "registry",
    "conditional_grad",
]
