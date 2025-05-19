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
import os.path as osp
from typing import Dict
from typing import Optional

from omegaconf import OmegaConf

from ppmat.models.comformer.comformer import iComformer
from ppmat.models.comformer.comformer_graph_converter import ComformerGraphConverter
from ppmat.models.common.graph_converter import FindPointsInSpheres
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.megnet.megnet import MEGNetPlus
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load

__all__ = [
    "iComformer",
    "ComformerGraphConverter",
    "DiffCSP",
    "FindPointsInSpheres",
    "MEGNetPlus",
    "MatterGen",
]


MODEL_REGISTRY = {
    "comformer_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_e_form.zip",
    "comformer_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_band_gap.zip",
    "comformer_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_G.zip",
    "comformer_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_K.zip",
    "megnet_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_e_form.zip",
    "diffcsp_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip",
}


def build_graph_converter(cfg: Dict):
    """Build graph converter.

    Args:
        cfg (Dict): Graph converter config.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    graph_converter = eval(class_name)(**init_params)
    logger.debug(str(graph_converter))

    return graph_converter


def build_model(cfg: Dict):
    """Build Model.

    Args:
        cfg (Dict): Model config.

    Returns:
        nn.Layer: Model object.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    model = eval(class_name)(**init_params)
    logger.debug(str(model))

    return model


def build_model_from_name(model_name: str, weights_name: Optional[str] = None):
    path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
    path = osp.join(path, model_name)

    config_path = osp.join(path, f"{model_name}.yaml")

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    model_config = config.get("Model", None)
    assert model_config is not None, "Model config must be provided."
    model = build_model(model_config)

    save_load.load_pretrain(model, path, weights_name)

    return model, config
