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

from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure


class LongTimeMem:
    def __init__(self) -> None:
        self.memory = pd.DataFrame(columns=["struc", "comp", "ele_comb", "reward", "RL_step"])
        self.unique_comps = []

    def extend(self, strucs: List[Structure], rewards: np.ndarray, step: int) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = [tuple(sorted(set(str(e) for e in s.species))) for s in strucs]
        df = pd.DataFrame.from_dict({"struc": strucs, "comp": comps, "ele_comb": ele_comb,
                                      "reward": rewards, "RL_step": [step] * len(strucs)})
        self.memory = pd.concat([self.memory, df]) if len(self.memory) > 0 else df
        self.unique_comps = list(self.memory["comp"].unique())

    def div_filter(self, strucs: List[Structure], rewards: np.ndarray,
                   tol: int = 10, buff: int = 20, method: str = "composition",
                   **kwargs) -> Tuple[np.ndarray, List[int], int, int]:
        assert tol < buff
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = [tuple(sorted(set(str(e) for e in s.species))) for s in strucs]
        key = "comp" if method == "composition" else "ele_comb"
        values = comps if method == "composition" else ele_comb
        new_rewards, penalty_idx, tol_n, buff_n = [], [], 0, 0
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

    def calc_metrics(self, thred: float, budget: int = 3000,
                     num_candidate: int = 100) -> Tuple[float, float]:
        _df = self.memory.sort_values("reward", ascending=False).drop_duplicates(subset=["comp"])
        candidates = (_df["reward"] > thred).sum()
        burden = len(self.memory) / candidates if candidates >= num_candidate else None
        div_ratio = len(self.unique_comps) / len(self.memory) if len(self.memory) <= budget else None
        return burden, div_ratio

    def get_baseline(self, step: int, prev: int = 3):
        return self.memory[self.memory["RL_step"] > step - prev]["reward"].mean()

    def deduplicate(self, df: pd.DataFrame, method="composition") -> pd.DataFrame:
        return df.sort_values("reward", ascending=False).drop_duplicates(subset=["comp"])

    def save(self, save_path: str):
        df = self.memory.copy()
        df["cif"] = [s.to(fmt="cif") for s in df["struc"].values]
        df.to_csv(save_path, index=False, quoting=1)

    def __len__(self) -> int:
        return len(self.memory)


class ReplayBuffer:
    def __init__(self, buffer_size: int = 100, sample_size: int = 8,
                 reward_cutoff: float = 0.0, capacity: int = None) -> None:
        self.buffer_size = capacity if capacity is not None else buffer_size
        self.sample_size = sample_size
        self.reward_cutoff = reward_cutoff
        self.buffer = pd.DataFrame(columns=["data", "struc", "comp", "ele_comb", "reward"])

    def extend(self, data: list, strucs: List[Structure], rewards: np.ndarray) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        ele_comb = [tuple(sorted(set(str(e) for e in s.species))) for s in strucs]
        df = pd.DataFrame.from_dict({"data": data, "struc": strucs, "comp": comps,
                                      "ele_comb": ele_comb, "reward": rewards})
        df_all = pd.concat([self.buffer, df]) if len(self.buffer) > 0 else df
        self.buffer = df_all.sort_values("reward", ascending=False).drop_duplicates(
            subset=["comp"]).head(self.buffer_size)
        self.buffer = self.buffer[self.buffer["reward"] > self.reward_cutoff]

    def sample(self) -> Tuple[list, np.ndarray]:
        n = min(len(self.buffer), self.sample_size)
        if n > 0:
            sampled = self.buffer.sample(n)
            return sampled["data"].values.tolist(), sampled["reward"].values
        return [], []

    def memory_purge(self, strucs: List[Structure]) -> None:
        comps = [s.composition.reduced_formula for s in strucs]
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps)]

    def __len__(self) -> int:
        return len(self.buffer)
