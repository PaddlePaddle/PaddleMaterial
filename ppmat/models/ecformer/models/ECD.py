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

import paddle.nn as nn

from .base_ecformer import ECFormerBase


class ECFormerECD(ECFormerBase):
    """ECFormer for ECD spectrum prediction - peak attribute decoupling version"""
    
    def __init__(
        self,
        num_position_classes = 20,
        height_classes       = 2,       
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Peak number prediction head
        self.pred_number_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 2, self.max_peaks)
        )
        
        # Peak position prediction head
        self.pred_position_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, num_position_classes)
        )
        
        # Peak sign prediction head
        self.pred_height_layer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 4, height_classes)
        )