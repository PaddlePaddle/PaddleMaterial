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

from typing import List
from typing import Tuple

import numpy as np
import paddle
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure


def _process_result_dict(r: dict):
    na = r["num_atoms"]
    frac = np.array(r["frac_coords"], dtype=np.float32)
    at = np.array(r["atom_types"], dtype=np.int32)
    lat = np.array(r["lattice"], dtype=np.float32)
    sd = {"structure_array": {"num_atoms": paddle.to_tensor([na], dtype="int64"),
          "frac_coords": paddle.to_tensor(frac), "atom_types": paddle.to_tensor(at),
          "lattice": paddle.to_tensor(lat).unsqueeze(0)}}
    try:
        pmg = Structure(lattice=Lattice(lat), species=at.astype(int),
                        coords=frac, coords_are_cartesian=False)
    except Exception:
        pmg = None
    return sd, pmg


class BaseSampler:
    def __init__(self, batch_size: int = 16, num_batches: int = 4,
                 num_inference_steps: int = 1000):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_inference_steps = num_inference_steps

    def generate(self, model, **kwargs):
        raise NotImplementedError


class MatterGenSampler(BaseSampler):
    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        all_data, all_structs = [], []
        for idx in range(self.num_batches * self.batch_size):
            na = self._resolve_num_atoms(kwargs, idx, min_atoms=2)
            if int(na.shape[0]) == 1:
                na = paddle.concat([na, na], axis=0)
            try:
                with paddle.no_grad():
                    out = model.sample({"structure_array": {"num_atoms": na}},
                                       num_inference_steps=self.num_inference_steps)
            except Exception:
                continue
            for r in out["result"]:
                sd, s = _process_result_dict(r)
                all_data.append(sd)
                all_structs.append(s)
        return all_data, all_structs

    @staticmethod
    def _resolve_num_atoms(kwargs, idx, min_atoms=1):
        if "num_atoms" not in kwargs:
            return paddle.randint(low=min_atoms, high=50, shape=[1])
        na = kwargs["num_atoms"]
        if isinstance(na, int):
            na = [max(min_atoms, na)]
        elif isinstance(na, (list, tuple, np.ndarray)):
            na = [max(min_atoms, int(na[idx % len(na)]))]
        else:
            na = [max(min_atoms, int(na))]
        return paddle.to_tensor(na, dtype="int64")


class DiffCSPSampler(BaseSampler):
    def generate(self, model, **kwargs) -> Tuple[List, List[Structure]]:
        all_data, all_structs = [], []
        for _ in range(self.num_batches):
            if "num_atoms" in kwargs:
                na = kwargs["num_atoms"]
                na = paddle.to_tensor([na] * self.batch_size if isinstance(na, int) else na, dtype="int64")
            else:
                na = paddle.randint(low=1, high=50, shape=[self.batch_size])
            at = paddle.randint(low=1, high=101, shape=[int(na.sum().item())], dtype="int64")
            with paddle.no_grad():
                out = model.sample({"structure_array": {"num_atoms": na, "atom_types": at}},
                                   num_inference_steps=self.num_inference_steps)
            for r in out.get("result", []):
                sd, s = _process_result_dict(r)
                all_data.append(sd)
                all_structs.append(s)
        return all_data, all_structs
