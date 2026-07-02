# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from typing import List
from typing import Optional

from omegaconf import OmegaConf

SAMPLER_REGISTRY = {
    "diffnmr": "ppmat.sampler.diffnmr_sampler:DiffNMRSampler",
}

MODEL_CLASS_TO_SAMPLER = {
    "DiffNMR": "diffnmr",
}

MODEL_NAME_TO_SAMPLER = {
    "diffnmr_msdnmr_nless15": "diffnmr",
}


def _get_sampler_name_from_overrides(config_overrides: Optional[List[str]]):
    if not config_overrides:
        return None
    config = OmegaConf.from_dotlist(config_overrides)
    sampler_config = config.get("Sampler", None)
    if sampler_config is None:
        return None
    return sampler_config.get("name", None)


def _load_config(config_path: Optional[str], config_overrides: Optional[List[str]]):
    assert (
        config_path is not None
    ), "config_path must be provided when model_name is None."

    config = OmegaConf.load(config_path)
    if config_overrides:
        cli_config = OmegaConf.from_dotlist(config_overrides)
        config = OmegaConf.merge(config, cli_config)
    return OmegaConf.to_container(config, resolve=True)


def _infer_sampler_name(
    model_name: Optional[str],
    config_path: Optional[str],
    config_overrides: Optional[List[str]],
):
    sampler_name = _get_sampler_name_from_overrides(config_overrides)
    if sampler_name is not None:
        return sampler_name

    if model_name is not None:
        if model_name in MODEL_NAME_TO_SAMPLER:
            return MODEL_NAME_TO_SAMPLER[model_name]
        raise ValueError(
            f"Unable to infer sampler type from model_name '{model_name}'. "
            "Please add it to MODEL_NAME_TO_SAMPLER or set `Sampler.name`."
        )

    config = _load_config(config_path=config_path, config_overrides=config_overrides)
    sampler_config = config.get("Sampler", None)
    if sampler_config is not None:
        sampler_name = sampler_config.get("name", None)
        if sampler_name is not None:
            return sampler_name

    model_config = config.get("Model", None)
    if model_config is not None:
        model_class_name = model_config.get("__class_name__", None)
        if model_class_name in MODEL_CLASS_TO_SAMPLER:
            return MODEL_CLASS_TO_SAMPLER[model_class_name]

    raise ValueError(
        "Unable to infer sampler type from config. Please set `Sampler.name`."
    )


def _build_sampler_from_name(sampler_name: str):
    if sampler_name not in SAMPLER_REGISTRY:
        supported_names = ", ".join(sorted(SAMPLER_REGISTRY))
        raise KeyError(
            f"No sampler registered for '{sampler_name}'. "
            f"Supported samplers: {supported_names}."
        )

    module_name, class_name = SAMPLER_REGISTRY[sampler_name].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class MolecularSampler:
    """Build a molecular sampler from a registered or explicit configuration."""

    def __new__(
        cls,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        config_overrides: Optional[List[str]] = None,
    ):
        sampler_name = _infer_sampler_name(
            model_name=model_name,
            config_path=config_path,
            config_overrides=config_overrides,
        )
        sampler_cls = _build_sampler_from_name(sampler_name)
        return sampler_cls(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            config_overrides=config_overrides,
        )
