# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMatG private constants.

A single :class:`OMatGConstants` dataclass collects every OMatG-specific
magic number so callers do not need to import a separate symbol for each
one. This is the single source of truth: ``OMATGCSPNet`` /
``OMATGCSPNetFull`` / ``IndependentSampler`` / ``DiscreteFlowMatchingMask``
all read from the same struct, and overriding the default width is a
single-attribute change here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OMatGConstants:
    """OMatG private constants.

    Attributes:
        default_max_atoms: Species cardinality of the released PP-OMatG
            checkpoints (5 datasets x 11 variants, all trained at width
            100). The ``OMATGCSPNet`` / ``OMATGCSPNetFull`` constructors
            and ``IndependentSampler`` / ``DiscreteFlowMatchingMask`` all
            read from this single value, so changing the released width
            is a one-line edit.
        small_time: Lower bound of the SI integration time grid
            ``[SMALL_TIME, BIG_TIME]``. Matches upstream OMatG.
        big_time: Upper bound of the SI integration time grid, equal to
            ``1 - SMALL_TIME``.
    """

    default_max_atoms: int = 100
    small_time: float = 1.0e-3
    big_time: float = 1.0 - 1.0e-3


OMatG = OMatGConstants()

__all__ = ["OMatG"]
