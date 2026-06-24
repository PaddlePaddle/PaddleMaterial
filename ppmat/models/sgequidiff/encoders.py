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

"""Feature encoders for SGEquiDiff."""
import paddle.nn as nn

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.non_equivariant_drift_modules import Swish


class SpaceGroupEncoder(nn.Layer):
    def __init__(self, hidden_channels: int = 128, space_group_embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_vars.embedding_tools.space_group_embedding_length, hidden_channels),
            Swish(),
            nn.Linear(hidden_channels, space_group_embedding_dim),
            Swish(),
        )

    def forward(self, space_group_indices):
        return self.net(global_vars.embedding_tools.get_space_group_embedding(space_group_indices))
