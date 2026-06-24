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

import os
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import paddle.io as io
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure

from ppmat.utils import logger as ppmat_logger


def is_valid_structure(struc: Structure, min_volume: float = 0.1,
                       max_lattice_param: float = 25.0,
                       min_interatomic_dist: float = 0.5) -> bool:
    try:
        if struc is None or struc.num_sites == 0 or struc.volume <= min_volume:
            return False
        if max(struc.lattice.abc) > max_lattice_param:
            return False
        dmat = struc.distance_matrix.copy()
        np.fill_diagonal(dmat, np.inf)
        if dmat.min() < min_interatomic_dist:
            return False
        return True
    except Exception:
        return False


def save_structures(structures: List[Structure], save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    with open(out_path, "w"):
        for struc in structures:
            write(out_path, adaptor.get_atoms(struc), append=True)
    ppmat_logger.info(f"Saved {len(structures)} structures to {out_path}")
    return out_path


class RLDataset(io.Dataset):
    def __init__(self, structures: List[Structure], rewards: np.ndarray,
                 transform: Optional[callable] = None):
        assert len(structures) == len(rewards)
        self.structures = structures
        self.rewards = rewards
        self.transform = transform
        ppmat_logger.info(f"Created RLDataset with {len(structures)} samples")

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        s = self.structures[idx]
        frac = np.array([site.frac_coords for site in s.sites], dtype=np.float32)
        return {"frac_coords": frac, "lattice": s.lattice.matrix.astype(np.float32),
                "atom_types": np.array([site.specie.Z for site in s.sites], dtype=np.int64),
                "num_atoms": len(s), "reward": np.float32(self.rewards[idx])}


def collate_fn(batch):
    fc = np.concatenate([b["frac_coords"] for b in batch], axis=0)
    at = np.concatenate([b["atom_types"] for b in batch], axis=0)
    lt = np.stack([b["lattice"] for b in batch], axis=0)
    na = np.array([b["num_atoms"] for b in batch], dtype=np.int64)
    rw = np.array([b["reward"] for b in batch], dtype=np.float32)
    bi = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(na)])
    return {"structure_array": {"frac_coords": paddle.to_tensor(fc),
            "lattice": paddle.to_tensor(lt), "atom_types": paddle.to_tensor(at),
            "num_atoms": paddle.to_tensor(na), "batch": paddle.to_tensor(bi),
            "reward": paddle.to_tensor(rw)}}


def create_rl_dataloader(structures: List[Structure], rewards: np.ndarray,
                         batch_size: int = 8, shuffle: bool = True,
                         num_workers: int = 0) -> io.DataLoader:
    dl = io.DataLoader(dataset=RLDataset(structures, rewards), batch_size=batch_size,
                       shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    ppmat_logger.info(f"DataLoader: {len(dl.dataset)} samples, batch_size={batch_size}")
    return dl


def _process_result_dict(r: dict):
    na = r["num_atoms"]
    frac = np.array(r["frac_coords"], dtype=np.float32)
    at = np.array(r["atom_types"], dtype=np.int32)
    lat = np.array(r["lattice"], dtype=np.float32)
    sd = {"structure_array": {"num_atoms": paddle.to_tensor([na], dtype="int64"),
          "frac_coords": paddle.to_tensor(frac), "atom_types": paddle.to_tensor(at),
          "lattice": paddle.to_tensor(lat).unsqueeze(0)}}
    try:
        pmg = _to_pmg(frac, at, lat[np.newaxis], [na])[0]
    except Exception:
        pmg = None
    return sd, pmg


def _to_pmg(frac_coords, atom_types, lattice, num_atoms):
    structs, start = [], 0
    for i in range(len(num_atoms)):
        n = num_atoms[i]
        end = start + n
        structs.append(Structure(lattice=Lattice(lattice[i]),
                        species=atom_types[start:end].astype(int),
                        coords=frac_coords[start:end], coords_are_cartesian=False))
        start = end
    return structs


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
