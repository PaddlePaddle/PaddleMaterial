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

from ppmat.models.comformer.comformer import iComformer
from ppmat.models.comformer.comformer_graph_converter import ComformerGraphConverter
from ppmat.utils import logger

__all__ = ["iComformer", "ComformerGraphConverter"]


MODEL_NAMES = {
    "comformer_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_e_form.zip",
}


def build_graph_converter(cfg):
    """Build Model.

    Args:
        cfg (DictConfig): Model config.

    Returns:
        nn.Layer: Model.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    model = eval(class_name)(**init_params)
    logger.debug(str(model))

    return model


def build_model(cfg):
    """Build Model.

    Args:
        cfg (DictConfig): Model config.

    Returns:
        nn.Layer: Model.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    model = eval(class_name)(**init_params)
    logger.debug(str(model))

    return model
