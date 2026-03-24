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
Replay buffer for reinforcement learning.

This code is adapted from:

Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""

from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure


class ReplayBuffer:
    """
    Replay buffer class which stores the top K highest reward crystals generated so far.
        1. Crystals (data, pymatgen.Structure, compositions)
        2. Reward
    """

    def __init__(
        self,
        buffer_size: int = 100,
        sample_size: int = 8,
        reward_cutoff: float = 0.0,
        capacity: int = None,
    ) -> None:
        self.buffer_size = capacity if capacity is not None else buffer_size
        self.sample_size = sample_size
        self.reward_cutoff = reward_cutoff
        # Stores the top N highest reward crystal generated so far
        self.buffer = pd.DataFrame(
            columns=["data", "struc", "comp", "ele_comb", "reward"]
        )

    def extend(
        self,
        data: list,
        strucs: List[Structure],
        rewards: np.ndarray,
    ) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = []
        for s in strucs:
            elements = set(str(e) for e in s.species)
            comb = tuple(sorted(elements))
            ele_comb.append(comb)

        # cs_list, pg_list, sg_list = [], [], []
        # for struc in strucs:
        #     analyzer = SpacegroupAnalyzer(struc)
        #     # Get the crystal system
        #     crystal_system = analyzer.get_crystal_system()
        #     # Get the point group
        #     point_group = analyzer.get_point_group_symbol()
        #     # Get the space group symbol and number
        #     space_group = analyzer.get_space_group_symbol()
        #     cs_list.append(crystal_system)
        #     pg_list.append(point_group)
        #     sg_list.append(space_group)

        df_sam = pd.DataFrame.from_dict(
            {
                "data": data,
                "struc": strucs,
                "comp": comps,
                "ele_comb": ele_comb,
                "reward": rewards,
            }
        )
        if len(self.buffer) > 0:
            df_all = pd.concat([self.buffer, df_sam])
        else:
            df_all = df_sam
        unique_df = self.deduplicate(df_all)
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.buffer = sorted_df.head(self.buffer_size)
        # reward cutoff
        self.buffer = self.buffer.loc[self.buffer["reward"] > self.reward_cutoff]

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        """
        Removes duplicate crystals based on different methods like composition,
        StructureMatcher, symmetry (crystal system, space group, etc.)
        Keep only non-zero rewards crystals.
        """
        _df = df.sort_values("reward", ascending=False)
        if method == "composition":
            unique_df = _df.drop_duplicates(subset=["comp"])
        elif method == "element_comb":
            unique_df = _df.drop_duplicates(subset=["ele_comb"])

        return unique_df

    def sample(self) -> Tuple[list, np.ndarray]:
        sample_size = min(len(self.buffer), self.sample_size)
        if sample_size > 0:
            sampled = self.buffer.sample(sample_size)
            data = sampled["data"].values.tolist()
            rewards = sampled["reward"].values
            return data, rewards
        else:
            return [], []

    def memory_purge(self, strucs: List[Structure]) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps)]

    def __len__(self) -> int:
        return len(self.buffer)
