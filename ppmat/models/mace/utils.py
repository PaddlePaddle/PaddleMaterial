# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

def atomic_number_to_index(atomic_numbers):
    """Convert supported MACE atomic numbers to fixed embedding indices."""
    supported = list(range(1, 84)) + list(range(89, 96))
    atom_to_idx = {atomic_number: index for index, atomic_number in enumerate(supported)}
    return [atom_to_idx[int(atomic_number)] for atomic_number in atomic_numbers]
