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

import copy

import paddle  # noqa

__all__ = ["build_metric"]


def build_metric(cfg):
    """Build metric.

    Args:
        cfg (DictConfig): Metric config.

    Returns:
        Metric: Callable Metric object.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    if "__class_name__" not in cfg:
        assert isinstance(cfg, dict)
        metric_dict = {}
        for key, sub_cfg in cfg.items():
            metric_dict[key] = build_metric(sub_cfg)
        return metric_dict

    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")

    metric = eval(class_name)(**init_params)
    return metric
