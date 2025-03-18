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

from ppmat.metrics.mae_metric import MAEMetric
from ppmat.metrics.metric_warper import MetricWarper

__all__ = ["MAEMetric", "MetricWarper", "CSPMetric", "GenMetric"]


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

    metric_cls = cfg.pop("__name__")
    if metric_cls == "MetricWarper":
        metric_cfg = cfg.pop("metric_fns")

        if isinstance(metric_cfg, list):
            metric_fns = []
            for i in range(len(metric_cfg)):
                metric_fns.append(build_metric(metric_cfg[i]))
        else:
            metric_fns = build_metric(metric_cfg)
        cfg["metric_fns"] = metric_fns

    loss = eval(metric_cls)(**cfg)
    return loss
