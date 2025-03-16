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

from typing import Dict
from typing import Optional

import pgl
from ppmat.modules.base_module import BaseModule

from ppmat.datasets.transform import build_post_process
from ppmat.metrics import build_metric


class RegressionModule(BaseModule):
    """Regression module for regression task

    Args:
        model_cfg (Dict): Model config.
        loss_cfg (Optional[Dict], optional): Loss config. Defaults to None.
        post_process_cfg (Optional[Dict], optional): Post process config. Defaults to
            None.
        metric_cfg (Optional[Dict], optional): Metric config. Defaults to None.
    """

    def __init__(
        self,
        model_cfg: Dict,
        loss_cfg: Optional[Dict] = None,
        post_process_cfg: Optional[Dict] = None,
        metric_cfg: Optional[Dict] = None,
        **kwargs,  # for compatibility
    ):

        super().__init__(
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
        )
        self.post_process_cfg = post_process_cfg
        self.metric_cfg = metric_cfg

        self.post_process_fn = build_post_process(post_process_cfg)
        self.metric_fn = build_metric(metric_cfg)

    def __convert_to_tensor(self, data):
        # convert pgl.graph.Graph to tensor
        if isinstance(data, dict):
            for _, value in data.items():
                self.__convert_to_tensor(value)
        elif isinstance(data, pgl.graph.Graph):
            data.tensor()

    def forward_step(
        self, batch_data, calc_loss=True, post_process=True, calc_metric=True
    ):

        results = {}
        pred_data = self.model(batch_data)
        results["pred_data"] = pred_data

        if calc_loss and self.loss_fn is not None:
            loss_dict = self.loss_fn(pred_data, batch_data)
            results["loss_dict"] = loss_dict
        if post_process and self.post_process_fn is not None:
            pred_data, batch_data = self.post_process_fn(pred_data, batch_data)
            results["pred_data"] = pred_data
        if calc_metric and self.metric_fn is not None:
            metric_dict = self.metric_fn(pred_data, batch_data)
            results["metric_dict"] = metric_dict

        return results

    def _forward_train(self, batch_data, **kwargs):
        return self.forward_step(batch_data, **kwargs)

    def _forward_validation(self, batch_data, **kwargs):
        return self.forward_step(batch_data, **kwargs)

    def _forward_test(self, batch_data, **kwargs):
        return self.forward_step(batch_data, **kwargs)

    def _forward_predict(self, batch_data, **kwargs):
        return self.forward_step(batch_data, **kwargs)
