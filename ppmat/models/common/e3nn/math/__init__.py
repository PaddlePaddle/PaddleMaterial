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

from ._linalg import complete_basis
from ._linalg import direct_sum
from ._linalg import orthonormalize
from ._normalize_activation import moment
from ._normalize_activation import normalize2mom
from ._reduce import germinate_formulas
from ._reduce import reduce_permutation
from ._soft_one_hot_linspace import soft_one_hot_linspace
from ._soft_unit_step import soft_unit_step

__all__ = [
    "complete_basis",
    "direct_sum",
    "orthonormalize",
    "moment",
    "normalize2mom",
    "soft_unit_step",
    "soft_one_hot_linspace",
    "germinate_formulas",
    "reduce_permutation",
]
