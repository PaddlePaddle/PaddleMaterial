# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
import paddle

from ppmat.metrics.liflow_metric import LiFlowMSE

__all__ = ["build_metric", "LiFlowMSE"]


class IgnoreNanMetricWrapper:
    def __init__(self, **metric_cfg):
        self._metric = build_metric(metric_cfg)

    def __call__(self, pred, label):
        valid = ~paddle.isnan(label)
        return self._metric(pred[valid], label[valid]) if valid.any() else paddle.nan


def build_metric(cfg):
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    if "__class_name__" not in cfg:
        return {key: build_metric(value) for key, value in cfg.items()}
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    return eval(class_name)(**init_params)
