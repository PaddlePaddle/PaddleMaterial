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
"""

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
from paddle.io import Dataset

from ppmat.models import build_graph_converter
from ppmat.models.common.xyz_utils import compute_triplet_indices
from ppmat.utils import logger
from ppmat.utils.download import get_datasets_path_from_url
from ppmat.utils.misc import is_equal

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


_MOLECULE_URLS = {
    "aspirin": "http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz",
    "benzene_old": "http://quantum-machine.org/gdml/data/npz/benzene_old_dft.npz",
    "ethanol": "http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz",
    "malonaldehyde": "http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz",
    "naphthalene": "http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz",
    "salicylic": "http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz",
    "toluene": "http://quantum-machine.org/gdml/data/npz/toluene_dft.npz",
    "uracil": "http://quantum-machine.org/gdml/data/npz/uracil_dft.npz",
}

_DEFAULT_SPLITS = {name: (1000, 1000) for name in _MOLECULE_URLS}

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


def _build_md17_graph_thread(idx, all_z, all_pos, build_graph_cfg, cache_dir):
    """Thread worker: build graph + triplet indices for one MD17 frame."""
    converter = build_graph_converter(build_graph_cfg)
    pos_i = all_pos[idx]
    batch_t = np.zeros(all_z.shape[0], dtype=np.int64)
    ei = converter(paddle.to_tensor(pos_i), paddle.to_tensor(batch_t))
    ti = compute_triplet_indices(ei, all_z.shape[0])
    cache_data = {
        "edge_index": ei.numpy(),
        "ti_i": ti["i"].numpy(),
        "ti_j": ti["j"].numpy(),
        "ti_idx_kj": ti["idx_kj"].numpy(),
        "ti_idx_ji": ti["idx_ji"].numpy(),
        "ti_idx_lk": ti["idx_lk"].numpy(),
        "ti_idx_triplet": ti["idx_triplet"].numpy(),
    }
    save_path = osp.join(cache_dir, f"{idx:010d}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(cache_data, f)
    return idx


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset for energy and force prediction.

    Single `.npz` file downloaded via bcebos bundle or individual URL.
    Pre‑computed ``edge_index`` when ``build_graph_cfg`` is provided,
    with pickle cache (rank‑0 build, barrier sync).

    Args:
        path: Root directory for storing raw and cached data.
        name: Molecule name from the supported list.
        split: ``'train'``, ``'val'``, ``'test'``, or ``None`` (all).
        train_size: Number of training samples (default 1000).
        val_size: Number of validation samples (default 1000).
        force_key: Key name for forces in the output dict (default ``'force'``).
        build_graph_cfg: Configuration dict for graph converter. Defaults to None.
        transforms: Optional transform callable.
        **kwargs: Compatibility.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
    md5 = "634cc25cc8a3fb0d99bd14245eb8dabd"
    name = "md17"

    def __init__(
        self,
        path: str,
        name: str = "benzene_old",
        split=None,
        train_size=None,
        val_size=None,
        *,
        force_key="force",
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        super().__init__()

        if name not in _MOLECULE_URLS:
            raise ValueError(
                f"Unknown MD17 molecule '{name}'. "
                f"Supported: {list(_MOLECULE_URLS.keys())}"
            )
        self.mol_name = name
        self.force_key = force_key
        self.transforms = transforms

        os.makedirs(path, exist_ok=True)
        raw_dir = osp.join(path, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # ---- 1. Inline download (MP20 pattern) ----
        individual_path = osp.join(raw_dir, f"{name}_dft.npz")
        if osp.exists(individual_path):
            raw_path = individual_path
        else:
            raw_path = None
            try:
                extract_dir = get_datasets_path_from_url(self.url, self.md5)
                bundle_rel = _BUNDLE_NPZ_MAP[name]
                for sub in ["", "md17/"]:
                    candidate = osp.join(extract_dir, sub, bundle_rel)
                    if osp.exists(candidate):
                        raw_path = candidate
                        break
            except Exception as e:
                logger.warning(f"bcebos download failed: {e}")
            if raw_path is None:
                import urllib.request

                url = _MOLECULE_URLS[name]
                logger.info(f"Downloading MD17/{name} from {url} ...")
                urllib.request.urlretrieve(url, individual_path)
                raw_path = individual_path

        # ---- 2. Load npz data ----
        data = np.load(raw_path)
        all_z = data["z"]
        all_pos = data["R"]
        all_energy = data["E"]
        all_forces = data["F"]
        total = all_pos.shape[0]

        # ---- 3. Split indices (DIG convention) ----
        ts = train_size or _DEFAULT_SPLITS[name][0]
        vs = val_size or _DEFAULT_SPLITS[name][1]
        rng = np.random.RandomState(42)
        indices = rng.permutation(total)
        if split == "train":
            self._indices = indices[:ts]
        elif split == "val":
            self._indices = indices[ts : ts + vs]
        elif split == "test":
            self._indices = indices[ts + vs :]
        else:
            self._indices = indices

        self._z_tensor = paddle.to_tensor(all_z, dtype=paddle.int64)
        self._pos_np = all_pos
        self._energy_np = all_energy
        self._forces_np = all_forces

        # ---- 4. Cache path (MP20 pattern) ----
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            base = path.rstrip("/").rstrip("\\")
            split_suffix = split if split is not None else "all"
            self.cache_path = osp.join(f"{base}_cache", f"{name}_{split_suffix}")
        logger.info(f"Cache path: {self.cache_path}")

        # ---- 5. Pre‑build edge_index + triplet indices (MP20 pattern) ----
        self.graph_cache = None
        if build_graph_cfg is not None:
            graph_cache_path = osp.join(self.cache_path, "graphs")
            cfg_pkl = osp.join(self.cache_path, "build_graph_cfg.pkl")
            cache_exists = osp.exists(graph_cache_path) and osp.exists(cfg_pkl)

            need_rebuild = overwrite or not cache_exists
            if cache_exists and not overwrite:
                try:
                    cfg_cached = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(cfg_cached, build_graph_cfg):
                        logger.info("build_graph_cfg matches cache. Reusing.")
                    else:
                        logger.warning(
                            "build_graph_cfg differs from cache. Rebuilding."
                        )
                        need_rebuild = True
                except Exception as e:
                    logger.warning(f"Cache check failed ({e}). Rebuilding.")
                    need_rebuild = True

            if need_rebuild:
                if dist.get_rank() == 0:
                    os.makedirs(graph_cache_path, exist_ok=True)
                    self.save_to_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl"),
                        build_graph_cfg,
                    )
                    logger.info(
                        f"Pre‑building graphs for MD17/{name} ({total} frames) "
                        f"with 24 threads ..."
                    )
                    with ThreadPoolExecutor(max_workers=24) as executor:
                        futures = {
                            executor.submit(
                                _build_md17_graph_thread,
                                i,
                                all_z,
                                all_pos,
                                build_graph_cfg,
                                graph_cache_path,
                            ): i
                            for i in range(total)
                        }
                        for _ in tqdm(
                            as_completed(futures), total=total, desc="Build graphs"
                        ):
                            pass
                if dist.is_initialized():
                    dist.barrier()
            self.graph_cache = [
                osp.join(graph_cache_path, f"{i:010d}.pkl") for i in range(total)
            ]

        self.num_samples = len(self._indices)
        logger.info(f"Load {self.num_samples} samples, split={split}")

    def save_to_cache(self, cache_path: str, obj):
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        sample = {
            "z": self._z_tensor.numpy(),
            "pos": self._pos_np[real_idx],
            "energy": np.array([float(self._energy_np[real_idx])], dtype=np.float32),
            self.force_key: self._forces_np[real_idx],
        }
        if self.graph_cache is not None:
            gpath = self.graph_cache[real_idx]
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
