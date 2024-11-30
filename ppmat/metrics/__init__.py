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

from ppmat.metrics.csp_metric import CSPMetric
from ppmat.metrics.mae_metric import MAEMetirc
from ppmat.metrics.metric_warper import MetricWarper

__all__ = ["MAEMetirc", "MetricWarper", "CSPMetric"]


def build_metric(cfg):
    """Build metric.

    Args:
        cfg (DictConfig): Metric config.

    Returns:
        Metric: Callable Metric object.
    """
    cfg = copy.deepcopy(cfg)

    metric_cls = cfg.pop("__name__")
    if metric_cls == "MetricWarper":
        metric_cfg = cfg.pop("metric_fn")
        metric_fn = {}
        for key in metric_cfg.keys():
            metric_fn[key] = build_metric(metric_cfg[key])
        cfg["metric_fn"] = metric_fn

    loss = eval(metric_cls)(**cfg)
    return loss
