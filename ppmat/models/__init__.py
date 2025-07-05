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
import os
import os.path as osp
from typing import Dict
from typing import Optional

from omegaconf import OmegaConf

from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter
from ppmat.models.comformer.comformer import iComformer
from ppmat.models.comformer.comformer_graph_converter import ComformerGraphConverter
from ppmat.models.common.graph_converter import CrystalNN
from ppmat.models.common.graph_converter import FindPointsInSpheres
from ppmat.models.diffcsp.diffcsp import DiffCSP
from ppmat.models.dimenetpp.dimenetpp import DimeNetPlusPlus
from ppmat.models.mattergen.mattergen import MatterGen
from ppmat.models.mattergen.mattergen import MatterGenWithCondition
from ppmat.models.mattersim.m3gnet import M3GNet
from ppmat.models.mattersim.m3gnet_graph_converter import M3GNetGraphConvertor
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
    "MatterGenWithCondition",
    "DimeNetPlusPlus",
    "CrystalNN",
    "CHGNetGraphConverter",
    "CHGNet",
    "M3GNetGraphConvertor",
    "M3GNet",
]

# Warning: The key of the dictionary must be consistent with the file name of the value
MODEL_REGISTRY = {
    "comformer_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_e_form.zip",
    "comformer_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_band_gap.zip",
    "comformer_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_G.zip",
    "comformer_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_K.zip",
    "comformer_mp2024_train_130k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2024_train_130k_e_form.zip",
    "comformer_jarvis_dft_2d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_dft_2d_e_form.zip",
    "comformer_jarvis_dft_3d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_dft_3d_e_form.zip",
    "comformer_jarvis_alex_pbe_2d_all_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_alex_pbe_2d_all_e_form.zip",
    "megnet_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_e_form.zip",
    "megnet_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_band_gap.zip",
    "megnet_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_G.zip",
    "megnet_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2018_train_60k_K.zip",
    "megnet_mp2024_train_130k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_mp2024_train_130k_e_form.zip",
    "megnet_jarvis_dft_2d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_dft_2d_e_form.zip",
    "megnet_jarvis_dft_3d_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_dft_3d_e_form.zip",
    "megnet_jarvis_alex_pbe_2d_all_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/megnet/megnet_jarvis_alex_pbe_2d_all_e_form.zip",
    "diffcsp_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip",
    "mattergen_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20.zip",
    "mattergen_mp20_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_chemical_system.zip",
    "mattergen_mp20_dft_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_band_gap.zip",
    "mattergen_mp20_dft_bulk_modulus": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_bulk_modulus.zip",
    "mattergen_mp20_dft_mag_density": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_mag_density.zip",
    "mattergen_alex_mp20": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20.zip",
    "mattergen_alex_mp20_dft_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_band_gap.zip",
    "mattergen_alex_mp20_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system.zip",
    "mattergen_alex_mp20_dft_mag_density": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density.zip",
    "mattergen_alex_mp20_ml_bulk_modulus": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_ml_bulk_modulus.zip",
    "mattergen_alex_mp20_space_group": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_space_group.zip",
    "mattergen_alex_mp20_chemical_system_energy_above_hull": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_chemical_system_energy_above_hull.zip",
    "mattergen_alex_mp20_dft_mag_density_hhi_score": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20_dft_mag_density_hhi_score.zip",
    "chgnet_mptrj": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/chgnet/chgnet_mptrj.zip",
    "dimenetpp_mp2018_train_60k_e_form": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/dimenet%2B%2B/dimenetpp_mp2018_train_60k_e_form.zip",
    "dimenetpp_mp2018_train_60k_band_gap": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/dimenet%2B%2B/dimenetpp_mp2018_train_60k_band_gap.zip",
    "dimenetpp_mp2018_train_60k_G": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/dimenet%2B%2B/dimenetpp_mp2018_train_60k_G.zip",
    "dimenetpp_mp2018_train_60k_K": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/dimenet%2B%2B/dimenetpp_mp2018_train_60k_K.zip",
    "mattersim_1M": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/mattersim/mattersim_1M.zip",
    "mattersim_5M": "https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/mattersim/mattersim_5M.zip",
    "mattergen_ml2ddb": "https://paddle-org.bj.bcebos.com/paddlematerial/workflow/ml2ddb/mattergen_ml2ddb.zip",
    "mattergen_ml2ddb_chemical_system": "https://paddle-org.bj.bcebos.com/paddlematerial/workflow/ml2ddb/mattergen_ml2ddb_chemical_system.zip",
    "mattergen_ml2ddb_space_group": "https://paddle-org.bj.bcebos.com/paddlematerial/workflow/ml2ddb/mattergen_ml2ddb_space_group.zip",
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
    if not osp.exists(config_path):
        logger.warning(
            f"Config file not found: {config_path}, try find other yaml files."
        )
        file_list = os.listdir(path)
        find_list = []
        for file in file_list:
            if file.endswith(".yaml") or file.endswith(".yml"):
                find_list.append(osp.join(path, file))
        if len(find_list) == 1:
            config_path = find_list[0]
        else:
            raise ValueError(
                f"Multiple yaml files found: {find_list}, must be only one"
            )
        logger.warning(f"Find config file: {config_path}, using this file.")

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    model_config = config.get("Model", None)
    assert model_config is not None, "Model config must be provided."
    model = build_model(model_config)

    save_load.load_pretrain(model, path, weights_name)

    return model, config
