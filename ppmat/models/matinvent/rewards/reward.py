# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List
from typing import Tuple

import numpy as np
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen

CALCULATOR_REGISTRY = {"PyMatGen": PyMatGen}


def _linear_scale(values, minv=0.0, maxv=6.0):
    ss = (values - minv) / (maxv - minv)
    ss[ss > 1.0] = 1.0
    ss[ss < 0.0] = 0.0
    return ss


class Reward:
    def __init__(self, root_dir: str, prop_cfg: List, reward_threshold: float,
                 reduce: str = "mean"):
        assert reduce in ("mean", "min", "weight")
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = reward_threshold
        self.reduce = reduce
        self._cache = {}
        os.makedirs(root_dir, exist_ok=True)

    def calc_props(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        values = {}
        for idx, cfg in enumerate(self.prop_cfg):
            calc = self._cache.get(idx)
            if calc is None:
                cls_name = cfg.calculator.get("__class_name__")
                if cls_name not in CALCULATOR_REGISTRY:
                    raise ValueError(
                        f"Unknown calculator: {cls_name}, available: {list(CALCULATOR_REGISTRY)}")
                calc = CALCULATOR_REGISTRY[cls_name](**{
                    k: v for k, v in cfg.calculator.items() if k != "__class_name__"})
                self._cache[idx] = calc
            raw = calc.calc(samples, label)
            values[cfg.name] = np.nan_to_num(raw, nan=0.0).astype(float)
        failed = np.isnan(np.array(list(values.values()))).any(axis=0)
        return values, failed

    def scoring(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        prop_dict, failed_mask = self.calc_props(samples, label)
        scaled = {}
        for cfg in self.prop_cfg:
            v = prop_dict[cfg.name]
            if cfg.target == "ascending":
                sv = _linear_scale(v, cfg.minv, cfg.maxv)
            elif cfg.target == "descending":
                sv = _linear_scale(-v, -cfg.maxv, -cfg.minv)
            elif isinstance(cfg.target, float):
                sv = _linear_scale(-np.abs(v - cfg.target), -cfg.maxv, -cfg.minv)
            else:
                raise TypeError(
                    "prop cfg.target must be a float, 'ascending', or 'descending'")
            scaled[cfg.name] = sv
        if self.reduce == "mean":
            rewards = np.mean(list(scaled.values()), axis=0)
        elif self.reduce == "min":
            rewards = np.min(list(scaled.values()), axis=0)
        elif self.reduce == "weight":
            for cfg in self.prop_cfg:
                scaled[cfg.name] *= cfg.weight
            rewards = np.sum(list(scaled.values()), axis=0)
        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask
