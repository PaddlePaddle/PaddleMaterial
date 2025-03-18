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

import paddle.nn as nn

from ppmat.losses import build_loss
from ppmat.models import build_model


class BaseModule(nn.Layer):
    """Base model class.

    Args:
        model_cfg (Dict): Model config.
        loss_cfg (Optional[Dict], optional): Loss config. Defaults to None.
    """

    def __init__(
        self,
        model_cfg: Dict,
        loss_cfg: Optional[Dict] = None,
        **kwargs,  # for compatibility
    ):

        super().__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg

        self.model = build_model(model_cfg)
        self.loss_fn = build_loss(loss_cfg)

    def _forward_train(self, batch_data, **kwargs):
        raise NotImplementedError()

    def _forward_validation(self, batch_data, **kwargs):
        raise NotImplementedError()

    def _forward_test(self, batch_data, **kwargs):
        raise NotImplementedError()

    def _forward_predict(self, batch_data, **kwargs):
        raise NotImplementedError()

    def forward(self, batch_data, mode="train", **kwargs):
        assert mode in ["train", "eval", "test", "predict"]
        if mode == "train":
            return self._forward_train(batch_data, **kwargs)
        elif mode == "eval":
            return self._forward_validation(batch_data, **kwargs)
        elif mode == "test":
            return self._forward_test(batch_data, **kwargs)
        elif mode == "predict":
            return self._forward_predict(batch_data, **kwargs)
        else:
            raise NotImplementedError()
