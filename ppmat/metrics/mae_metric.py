# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import paddle.nn.functional as F


class MAEMetirc:
    r"""Mean absolute error."""

    def __init__(self):

        self.reset()

    def __call__(self, pred, label):
        mae = F.l1_loss(pred, label, "none")
        self.results.append(mae)
        self.all_num += mae.numel()
        mae = mae.mean().item()
        return mae

    def get_metric(self):
        mae_sum = 0
        for i in range(len(self.results)):
            mae_sum += self.results[i].sum()
        mae = mae_sum / self.all_num

        return mae.item()

    def reset(self):
        self.all_num = 0
        self.results = []
