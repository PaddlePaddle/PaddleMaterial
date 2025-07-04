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

from ppmat.losses.l1_loss import L1Loss
from ppmat.losses.l1_loss import SmoothL1Loss
from ppmat.losses.loss_warper import LossWarper
from ppmat.losses.mse_loss import MSELoss

__all__ = ["MSELoss", "L1Loss", "SmoothL1Loss", "LossWarper", "build_loss"]


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (DictConfig): Loss config.

    Returns:
        Loss: Callable loss object.
    """
    cfg = copy.deepcopy(cfg)

    loss_cls = cfg.pop("__name__")
    if loss_cls == "LossWarper":
        losses_cfg = cfg.pop("loss_fn")
        loss_fn = {}
        for key in losses_cfg.keys():
            loss_fn[key] = build_loss(losses_cfg[key])
        cfg["loss_fn"] = loss_fn

    loss = eval(loss_cls)(**cfg)
    return loss
