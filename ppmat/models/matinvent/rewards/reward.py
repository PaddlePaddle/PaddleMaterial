# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward class for reinforcement learning.

This code is adapted from:
"""

import os
from typing import List
from typing import Tuple

import numpy as np
from pymatgen.core.structure import Structure


def linear_scaling(values, minv=0.0, maxv=6.0):
    """Linear scaling of values to [0, 1] range."""
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


def average_props(prop_dict):
    """Average multiple properties."""
    prop_num = len(prop_dict)
    prop_list = list(prop_dict.values())
    prop_sum = prop_list[0]
    for _prop in prop_list[1:]:
        prop_sum += _prop
    prop_mean = prop_sum / prop_num

    return prop_mean


def min_props(prop_dict):
    """Take minimum of multiple properties."""
    prop_list = list(prop_dict.values())
    prop_arr = np.array(prop_list)
    prop_min = prop_arr.min(axis=0)
    return prop_min


class Reward:
    """Reward class for multi-property optimization."""

    def __init__(
        self,
        root_dir: str,
        prop_cfg: List,
        reward_threshold: float,
        reduce: str = "mean",
        **kwargs,
    ) -> None:
        assert reduce in ["mean", "min", "weight"]
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.reduce = reduce
        self._calculator_cache = {}
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def calc_props(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        """Calculate properties for samples."""
        prop_dict, prop_list = {}, []
        for idx, _cfg in enumerate(self.prop_cfg):
            calculator = self._calculator_cache.get(idx, _cfg.calculator)
            if not hasattr(calculator, "calc"):
                cls_expr = calculator.get("__class_name__")
                if cls_expr is None:
                    raise ValueError("calculator config must contain '__class_name__'")
                init_kwargs = {
                    k: v for k, v in calculator.items() if k != "__class_name__"
                }
                calculator = eval(cls_expr)(**init_kwargs)
                self._calculator_cache[idx] = calculator

            _prop1 = calculator.calc(samples, label)
            prop_list.append(_prop1)
            _prop2 = np.nan_to_num(_prop1, nan=0.0)
            prop_dict[_cfg.name] = _prop2.astype(float)

        prop_list = np.array(prop_list)
        none_ids = np.isnan(prop_list).any(axis=0)

        return prop_dict, none_ids

    def scoring(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        """Calculate rewards for samples."""

        prop_dict, failed_mask = self.calc_props(samples, label)

        scaled_prop_dict = {}
        for _cfg in self.prop_cfg:
            if _cfg.target == "ascending":
                _sprop = linear_scaling(
                    values=prop_dict[_cfg.name],
                    minv=_cfg.minv,
                    maxv=_cfg.maxv,
                )
            elif _cfg.target == "descending":
                _sprop = linear_scaling(
                    values=-prop_dict[_cfg.name],
                    minv=-_cfg.maxv,
                    maxv=-_cfg.minv,
                )
            elif isinstance(_cfg.target, float):
                diff = np.abs(prop_dict[_cfg.name] - _cfg.target)
                _sprop = linear_scaling(
                    values=-diff,
                    minv=-_cfg.maxv,
                    maxv=-_cfg.minv,
                )
            else:
                raise TypeError(
                    "prop cfg.target must be a float" " or descending or ascending"
                )

            scaled_prop_dict[_cfg.name] = _sprop

        if self.reduce == "mean":
            rewards = average_props(scaled_prop_dict)
        elif self.reduce == "min":
            rewards = min_props(scaled_prop_dict)
        elif self.reduce == "weight":
            for _cfg in self.prop_cfg:
                w = _cfg.weight
                scaled_prop_dict[_cfg.name] = scaled_prop_dict[_cfg.name] * w
            sprop_list = list(scaled_prop_dict.values())
            sprop_arr = np.array(sprop_list)
            rewards = sprop_arr.sum(axis=0)

        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask
