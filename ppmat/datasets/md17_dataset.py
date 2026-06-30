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
"""MD17 molecular dynamics dataset for energy and force prediction."""

import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle
import paddle.distributed as dist
import shutil
from paddle.io import Dataset
from tqdm import tqdm

from ppmat.datasets.graph_utils.spherenet_graph_utils import build_md17_graph
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


_BUNDLE_NPZ_MAP = {
    "aspirin": "md17_aspirin.npz",
    "benzene_old": "md17_benzene2017.npz",
    "ethanol": "md17_ethanol.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "naphthalene": "md17_naphthalene.npz",
    "salicylic": "md17_salicylic.npz",
    "toluene": "md17_toluene.npz",
    "uracil": "md17_uracil.npz",
}


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset for energy and force prediction.

    **STATS:**
    +----------------+----------+--------+-------+----------+-------+
    | Molecule       | #samples | #atoms | #tasks| #targets | Split |
    +================+==========+========+=======+==========+=======+
    | Aspirin        | 211,762  | 21     | 2     | E + F    | 1k/1k/R |
    | Benzene (old)  | 627,983  | 12     | 2     | E + F    | 1k/1k/R |
    | Ethanol        | 555,092  | 9      | 2     | E + F    | 1k/1k/R |
    | Malonaldehyde  | 993,237  | 9      | 2     | E + F    | 1k/1k/R |
    | Naphthalene    | 326,250  | 10     | 2     | E + F    | 1k/1k/R |
    | Salicylic      | 320,231  | 16     | 2     | E + F    | 1k/1k/R |
    | Toluene        | 442,790  | 15     | 2     | E + F    | 1k/1k/R |
    | Uracil         | 133,770  | 12     | 2     | E + F    | 1k/1k/R |
    +----------------+----------+--------+-------+----------+-------+

    Contains ab-initio molecular dynamics trajectories for eight small
    organic molecules.  Each frame provides atomic numbers, 3D positions,
    total energy, and per-atom forces.

    Data source: https://www.quantum-machine.org/datasets/

    Args:
        path (str): Root directory for storing raw and cached data.
        name (str): Molecule name from the supported list. Defaults to ``'benzene_old'``.
        split (Optional[str]): Split identifier ``'train'``, ``'val'``,
            ``'test'``, or ``None`` (all). Defaults to ``None``.
        force_key (Optional[str]): Key name for forces in the output dict.
            Defaults to ``'force'``.
        energy_key (Optional[str]): Key name for energy in the output dict.
            Defaults to ``'energy'``.
        build_graph_cfg (Optional[Dict]): Configuration dict for graph
            converter. Defaults to ``None``.
        transforms (Optional[Callable]): Per-sample transform callable.
            Defaults to ``None``.
        cache_path (Optional[str]): Explicit cache path. Auto-generated
            when ``None``. Defaults to ``None``.
        overwrite (bool): Whether to overwrite existing cached graphs.
            Defaults to ``False``.
        filter_unvalid (bool): Whether to filter out invalid samples.
            Defaults to ``True``.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
    md5 = "634cc25cc8a3fb0d99bd14245eb8dabd"
    name = "md17"

    def __init__(
        self,
        path: str,
        name: str = "benzene_old",
        split: str = None,
        *,
        force_key="force",
        energy_key="energy",
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.mol_name = name
        self.force_key = force_key
        self.energy_key = energy_key
        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        os.makedirs(path, exist_ok=True)

        # ---- 1. Download + split data ----
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        npz_path = osp.join(root_path, self.name, f"{name}_dft.npz")
        self._ensure_splits(npz_path, name)
        self.row_data, total = self.read_data(path, name)
        self._indices = self._load_split_indices(path, name, split)
        self.num_samples = len(self._indices)

        # ---- 2. Cache path ----
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            base = path.rstrip("/").rstrip("\\")
            split_suffix = split if split is not None else "all"
            self.cache_path = osp.join(f"{base}_cache", f"{name}_{split_suffix}")
        logger.info(f"Cache path: {self.cache_path}")

        # ---- 3. Pre‑build edge_index + triplet indices (MP20 pattern) ----
        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.graphs = None
        if build_graph_cfg is not None:
            graph_cache_path = osp.join(self.cache_path, "graphs")
            cfg_pkl = osp.join(graph_cache_path, "build_graph_cfg.pkl")
            if self.cache_exists and not overwrite:
                try:
                    cfg_cached = self.load_from_cache(cfg_pkl)
                    if not is_equal(cfg_cached, build_graph_cfg):
                        logger.warning(
                            "build_graph_cfg differs from cache. Rebuilding."
                        )
                        overwrite = True
                except Exception as e:
                    logger.warning(f"Cache check failed ({e}). Rebuilding.")
                    overwrite = True

            if overwrite or not self.cache_exists:
                if dist.get_rank() == 0:
                    os.makedirs(graph_cache_path, exist_ok=True)
                    self.save_to_cache(cfg_pkl, build_graph_cfg)
                    converter = build_graph_converter(build_graph_cfg)
                    logger.info(
                        f"Pre‑building graphs for {self.mol_name} ({total} frames) "
                        f"with 24 threads ..."
                    )
                    with ThreadPoolExecutor(max_workers=24) as executor:
                        futures = {
                            executor.submit(
                                build_md17_graph,
                                i, self.row_data["z"], self.row_data["pos"][i],
                                converter, graph_cache_path,
                            ): i for i in range(total)
                        }
                        for _ in tqdm(as_completed(futures), total=total, desc="Build graphs"):
                            pass
                if dist.is_initialized():
                    dist.barrier()
            self.graphs = [osp.join(graph_cache_path, f"{i:010d}.pkl") for i in range(total)]

        assert (
            self.graphs is None or len(self.graphs) == total
        ), "The number of graphs must be equal to the number of samples."

        logger.info(f"Load {self.num_samples} samples, split={split}")

    def _ensure_splits(self, raw_path, name):
        """Create pre-split npz files and index arrays (seed 42, 1000/1000 train/val)."""
        split_dir = osp.join(osp.dirname(raw_path), "splits")
        full_npz = osp.join(split_dir, f"{name}_all.npz")
        if not osp.exists(full_npz):
            if dist.get_rank() == 0:
                os.makedirs(split_dir, exist_ok=True)
                raw = np.load(raw_path)
                total = raw["R"].shape[0]
                rng = np.random.RandomState(42)
                perm = rng.permutation(total)
                ts, vs = 1000, 1000
                np.savez(osp.join(split_dir, f"{name}_train.npz"),
                         z=raw["z"], R=raw["R"][perm[:ts]],
                         E=raw["E"][perm[:ts]], F=raw["F"][perm[:ts]])
                np.savez(osp.join(split_dir, f"{name}_val.npz"),
                         z=raw["z"], R=raw["R"][perm[ts:ts + vs]],
                         E=raw["E"][perm[ts:ts + vs]], F=raw["F"][perm[ts:ts + vs]])
                np.savez(osp.join(split_dir, f"{name}_test.npz"),
                         z=raw["z"], R=raw["R"][perm[ts + vs:]],
                         E=raw["E"][perm[ts + vs:]], F=raw["F"][perm[ts + vs:]])
                np.savez(full_npz,
                         z=raw["z"], R=raw["R"], E=raw["E"], F=raw["F"])
                # Save split indices for MP20-style index-based access
                np.save(osp.join(split_dir, f"{name}_train_idx.npy"), perm[:ts])
                np.save(osp.join(split_dir, f"{name}_val_idx.npy"),
                        perm[ts:ts + vs])
                np.save(osp.join(split_dir, f"{name}_test_idx.npy"),
                        perm[ts + vs:])
                np.save(osp.join(split_dir, f"{name}_all_idx.npy"),
                        np.arange(total, dtype=np.int64))
            if dist.is_initialized():
                dist.barrier()

    def read_data(self, path, name):
        """Load all trajectory frames from the merged npz file."""
        split_dir = osp.join(path, "splits")
        data = np.load(osp.join(split_dir, f"{name}_all.npz"))
        row_data = {
            "z": data["z"],
            "pos": data["R"],
            "energy": data["E"],
            "force": data["F"],
        }
        return row_data, data["R"].shape[0]

    def _load_split_indices(self, path, name, split):
        """Load frame indices for the requested split."""
        split_dir = osp.join(path, "splits")
        key = split if split is not None else "all"
        return np.load(osp.join(split_dir, f"{name}_{key}_idx.npy"))

    def save_to_cache(self, cache_path: str, obj):
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, idx):
        frame = self._indices[idx]
        sample = {
            "z": self.row_data["z"],
            "pos": self.row_data["pos"][frame],
            self.energy_key: np.array(
                [float(self.row_data["energy"][frame])], dtype=np.float32
            ),
            self.force_key: self.row_data["force"][frame],
        }
        if self.graphs is not None:
            gpath = self.graphs[frame]
            graph = self.load_from_cache(gpath) if isinstance(gpath, str) else gpath
            if isinstance(graph, dict):
                sample["edge_index"] = graph["edge_index"]
                sample["triplet_indices"] = {
                    "i": graph["ti_i"],
                    "j": graph["ti_j"],
                    "idx_kj": graph["ti_idx_kj"],
                    "idx_ji": graph["ti_idx_ji"],
                    "idx_lk": graph["ti_idx_lk"],
                    "idx_triplet": graph["ti_idx_triplet"],
                }
            else:
                sample["edge_index"] = graph
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.num_samples
