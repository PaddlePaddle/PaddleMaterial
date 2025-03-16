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
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    loss_cls = cfg.pop("__name__")
    if loss_cls == "LossWarper":
        losses_cfg = cfg.pop("loss_fns")
        if isinstance(losses_cfg, list):
            loss_fns = []
            for i in range(len(losses_cfg)):
                loss_fns.append(build_loss(losses_cfg[i]))
        else:
            loss_fns = build_loss(losses_cfg)
        cfg["loss_fns"] = loss_fns

    loss = eval(loss_cls)(**cfg)
    return loss
