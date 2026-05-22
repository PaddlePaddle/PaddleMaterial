# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
OMatG Dataset package initialization.

DefaultCollator uses paddle.stack, which requires equal-length tensors.
Crystal structures have variable atom counts per sample, so OMatG must
concat along dim=0 instead — a dedicated dataset + collator is unavoidable.
"""

from .structure_dataset import StructureDataset as OMATGStructureDataset

__all__ = ["OMATGStructureDataset"]
