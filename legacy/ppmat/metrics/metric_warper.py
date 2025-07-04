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

from typing import Callable
from typing import Dict


class MetricWarper:
    r"""Class for metric warp."""

    def __init__(
        self,
        metric_fn: Dict[str, Callable],
        main_indicator: str,
        min_better: bool,
    ):
        self.metric_fn = metric_fn
        self.main_indicator = main_indicator
        self.min_better = min_better
        self.reset()

    def __call__(self, pred_dict, label_dict):
        metric_dict = {}
        for key in self.metric_fn:
            metric_dict[key] = self.metric_fn[key](pred_dict[key], label_dict[key])
        return metric_dict

    def get_metric(self):
        return {key: self.metric_fn[key].get_metric() for key in self.metric_fn}

    def reset(self):
        for key in self.metric_fn:
            self.metric_fn[key].reset()
