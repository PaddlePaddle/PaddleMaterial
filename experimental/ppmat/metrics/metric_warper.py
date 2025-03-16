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

import copy
from typing import Callable
from typing import Dict
from typing import List


class MetricWarper:
    """This class is used to wrap the metric function, supporting multiple metrics.

    Args:
        metric_fns (Dict[str, Callable]): The metric functions.
        apply_keys (List[str]): The keys that need to be applied.
    """

    def __init__(
        self,
        metric_fns: Dict[str, Callable],
        apply_keys: List[str],
    ):
        if not isinstance(metric_fns, list):
            metric_fns = [metric_fns]
        if len(metric_fns) == 1 and len(apply_keys) > 1:
            metric_fns = [copy.deepcopy(metric_fns[0]) for _ in range(len(apply_keys))]

        assert len(metric_fns) == len(
            apply_keys
        ), "The length of metric_fns and apply_keys must be equal."

        self.metric_fns = metric_fns
        self.apply_keys = apply_keys

    def __call__(self, pred_dict, label_dict):
        metric_dict = {}
        for i, key in enumerate(self.apply_keys):
            metric_dict[key] = self.metric_fns[i](pred_dict[key], label_dict[key])
        return metric_dict
