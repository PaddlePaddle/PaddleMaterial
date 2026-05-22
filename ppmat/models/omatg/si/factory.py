# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Factory functions to build StochasticInterpolants and samplers from config.

Supports PaddleMaterials style (__class_name__ / __init_params__) configs so that
SI training can be driven purely by yaml configuration.
"""

import importlib

from ppmat.models.omatg.si.stochastic_interpolants import StochasticInterpolants
from ppmat.models.omatg.sampler.independent_sampler import IndependentSampler

_SI_MODULE = "ppmat.models.omatg.si"
_SAMPLER_MODULE = "ppmat.models.omatg.sampler"


def _resolve_class(class_name: str, default_module: str):
    """Resolve a class from a dotted path or a short name in default_module."""
    if "." in class_name:
        module_path, cls_name = class_name.rsplit(".", 1)
    else:
        module_path, cls_name = default_module, class_name
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_object(cfg: dict, default_module: str):
    """Build a single object from a {__class_name__, __init_params__} config.

    Recursively builds nested objects when an __init_params__ entry is itself a
    config dict (i.e. contains __class_name__).
    """
    cls = _resolve_class(cfg["__class_name__"], default_module)
    params = cfg.get("__init_params__", {})
    built_params = {}
    for key, val in params.items():
        if isinstance(val, dict) and "__class_name__" in val:
            built_params[key] = _build_object(val, default_module)
        elif isinstance(val, list):
            built_params[key] = [
                _build_object(item, default_module)
                if isinstance(item, dict) and "__class_name__" in item
                else item
                for item in val
            ]
        else:
            built_params[key] = val
    return cls(**built_params)


def build_si_from_cfg(si_cfg: dict) -> StochasticInterpolants:
    """Build StochasticInterpolants from a config dict.

    Expected schema:
        stochastic_interpolants: list of {__class_name__, __init_params__}
        data_fields: list[str]
        integration_time_steps: int
        relative_si_costs: dict[str, float]  # optional
    """
    interpolants = [
        _build_object(cfg_item, _SI_MODULE)
        for cfg_item in si_cfg["stochastic_interpolants"]
    ]
    return StochasticInterpolants(
        stochastic_interpolants=interpolants,
        data_fields=si_cfg["data_fields"],
        integration_time_steps=si_cfg.get("integration_time_steps", 210),
    )


def build_sampler_from_cfg(sampler_cfg: dict) -> IndependentSampler:
    """Build IndependentSampler from a config dict.

    Expected schema (all keys optional):
        position_distribution: {__class_name__, __init_params__}
        cell_distribution: {__class_name__, __init_params__}
        species_distribution: {__class_name__, __init_params__}
    """
    kwargs = {}
    for key in ("position_distribution", "cell_distribution", "species_distribution"):
        if key in sampler_cfg and sampler_cfg[key] is not None:
            kwargs[key] = _build_object(sampler_cfg[key], _SAMPLER_MODULE)
    return IndependentSampler(**kwargs)
