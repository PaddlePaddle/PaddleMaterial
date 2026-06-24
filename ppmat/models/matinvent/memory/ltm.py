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
Long-term memory for reinforcement learning.

This code is adapted from:
"""

from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure


class LongTimeMem:
    """
    Long-term memory class which stores all generated crystals so far.
        1. Crystals (data, pymatgen.Structure, compositions)
        2. Reward
    """

    def __init__(
        self,
    ) -> None:

        self.memory = pd.DataFrame(
            columns=["struc", "comp", "ele_comb", "reward", "RL_step"]
        )
        self.unique_comps = []

    def extend(self, strucs: List[Structure], rewards: np.ndarray, step: int) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        steps = [step for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)


        df_sample = pd.DataFrame.from_dict(
            {
                "struc": strucs,
                "comp": comps,
                "ele_comb": ele_comb,
                "reward": rewards,
                "RL_step": steps,
            }
        )
        if len(self.memory) > 0:
            self.memory = pd.concat([self.memory, df_sample])
        else:
            self.memory = df_sample
        self.unique_comps = self.memory["comp"].unique()

    def div_filter(
        self,
        strucs: List[Structure],
        rewards: np.ndarray,
        tol: int = 10,
        buff: int = 20,
        method: str = "composition",
        **kwargs,
    ) -> Tuple[np.ndarray, List[int], int, int]:
        assert tol < buff
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)

        if method == "composition":
            key = "comp"
            values = comps
        elif method == "element_comb":
            key = "ele_comb"
            values = ele_comb

        new_rewards = []
        penalty_idx = []
        tol_n = 0
        buff_n = 0
        for i, v in enumerate(values):
            occ = self.memory[key].value_counts().get(v, 0)
            if occ <= tol:
                new_rewards.append(rewards[i])
            elif occ > tol and occ < buff:
                new_rewards.append(rewards[i] * (buff - occ) / (buff - tol))
                tol_n += 1
            else:
                new_rewards.append(0.0)
                penalty_idx.append(i)
                buff_n += 1

        return np.array(new_rewards), penalty_idx, tol_n, buff_n

    def calc_metrics(
        self,
        thred: float,
        budget: int = 3000,
        num_candidate: int = 100,
    ) -> Tuple[float, float]:
        _df = self.memory.sort_values("reward", ascending=False)
        unique_df = _df.drop_duplicates(subset=["comp"])
        candidates = (unique_df["reward"] > thred).sum()
        calc_cost = len(self.memory)
        if candidates >= num_candidate:
            burden = calc_cost / candidates
        else:
            burden = None

        num_uni_comp = len(self.unique_comps)
        if calc_cost <= budget:
            div_ratio = num_uni_comp / calc_cost
        else:
            div_ratio = None

        return burden, div_ratio

    def get_baseline(self, step: int, prev: int = 3):
        baseline = self.memory[self.memory["RL_step"] > step - prev]["reward"].mean()
        return baseline

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        """
        Removes duplicate crystals based on different methods like composition,
        StructureMatcher, symmetry (crystal system, space group, etc.)
        Keep only non-zero rewards crystals.
        """
        if method == "composition":
            _df = df.sort_values("reward", ascending=False)
            unique_df = _df.drop_duplicates(subset=["comp"])
            return unique_df

    def save(self, save_path: str):
        df = self.memory.copy()
        strucs = df["struc"].values
        cif_list = [s.to(fmt="cif") for s in strucs]
        df["cif"] = cif_list
        df.to_csv(save_path, index=False, quoting=1)

    def __len__(self) -> int:
        return len(self.memory)
