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
"""QM9 (GDB-9) molecular quantum-chemical dataset.

**Key Properties:**
    mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
"""

from __future__ import annotations

import math
import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.models import build_graph_converter
from ppmat.models.common.xyz_utils import compute_triplet_indices
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal
from ppmat.utils.download import get_datasets_path_from_url
from tqdm import tqdm

# Symbol-to-atomic-number mapping (elements present in QM9)
_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
}

_QM9_PROP_NAMES = [
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2",
    "zpve", "U0", "U", "H", "G", "Cv",
]


def _parse_qm9_xyz(lines):
    """Parse a single QM9 .xyz block."""
    n_atoms = int(lines[0].strip())
    prop_line = lines[1].strip().replace("*^", "e")
    parts = prop_line.split()
    raw_vals = []
    for p in parts:
        try:
            raw_vals.append(float(p))
        except ValueError:
            raw_vals.append(0.0)
    props = {}
    for k, name in enumerate(_QM9_PROP_NAMES):
        idx = k + 2
        props[name] = raw_vals[idx] if idx < len(raw_vals) else 0.0
    z_list, pos_list = [], []
    for i in range(n_atoms):
        atom_line = lines[2 + i].strip().replace("*^", "e")
        atom_parts = atom_line.split()
        symbol = atom_parts[0]
        z_list.append(_SYMBOL_TO_Z.get(symbol, 0))
        x, y, z = float(atom_parts[1]), float(atom_parts[2]), float(atom_parts[3])
        pos_list.append([x, y, z])
    return np.array(z_list, dtype=np.int64), np.array(pos_list, dtype=np.float32), props


def _build_qm9_graph_thread(idx, merged_xyz, offsets, build_graph_cfg, cache_dir):
    """Thread worker: build graph + triplet indices for one QM9 molecule.
    
    No pickling needed (threads share memory).  Each thread creates its
    own converter instance for thread-safety.
    """
    converter = build_graph_converter(build_graph_cfg)
    z, pos = QM9Dataset._read_one_molecule(merged_xyz, offsets, idx)
    num_nodes = z.shape[0]
    batch_t = np.zeros(num_nodes, dtype=np.int64)
    ei = converter(paddle.to_tensor(pos), paddle.to_tensor(batch_t))
    ti = compute_triplet_indices(ei, num_nodes)
    cache_data = {
        'edge_index': ei.numpy(),
        'ti_i': ti['i'].numpy(),
        'ti_j': ti['j'].numpy(),
        'ti_idx_kj': ti['idx_kj'].numpy(),
        'ti_idx_ji': ti['idx_ji'].numpy(),
        'ti_idx_lk': ti['idx_lk'].numpy(),
        'ti_idx_triplet': ti['idx_triplet'].numpy(),
    }
    save_path = osp.join(cache_dir, f"{idx:010d}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(cache_data, f)
    return idx


class QM9Dataset(Dataset):
    """QM9 (GDB-9) dataset for quantum-chemical property prediction.

    Single merged ``.xyz`` file downloaded via ``get_datasets_path_from_url``.
    Pre‑computed ``edge_index`` when ``build_graph_cfg`` is provided,
    with pickle cache (rank‑0 build, barrier sync).

    Args:
        path: Root directory for storing raw and cached data.
        property_names: Target property name(s) from
            ``['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']``.
        build_graph_cfg: Configuration dict for graph converter
            (e.g. ``RadiusGraph(cutoff=5.0)``). Defaults to None.
        transforms: Optional transform callable applied in ``__getitem__``.
        cache_path: Explicit cache path. Auto‑generated when None.
        overwrite: Force cache rebuild.
        filter_unvalid: Remove samples with NaN/None properties.
        split: One of ``'train'``, ``'val'``, ``'test'``, or ``None`` (all).
        **kwargs: Compatibility.
    """

    name = "qm9"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2"
    md5 = "ad1ebd51ee7f5b3a6e32e974e5d54012"

    # DIG/TFDS split: seed=42 shuffle → 110000 / 10000 / rest
    _TRAIN_SIZE = 110000
    _VAL_SIZE = 10000

    def __init__(
        self,
        path: str,
        property_names: Union[str, List[str]] = None,
        *,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        split: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if split is not None and split not in ("train", "val", "test"):
            raise ValueError(f"split must be None/'train'/'val'/'test', got '{split}'")
        self.split = split

        if property_names is None:
            raise ValueError("property_names must be provided")
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = list(property_names)
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        os.makedirs(path, exist_ok=True)
        raw_dir = osp.join(path, "raw_qm9")
        os.makedirs(raw_dir, exist_ok=True)

        # ---- 1. Inline download (MP20 pattern) ----
        merged_xyz = osp.join(raw_dir, "dsgdb9nsd.xyz")
        if not osp.exists(merged_xyz):
            logger.info(f"Downloading QM9 from {self.url} ...")
            extract_dir = get_datasets_path_from_url(self.url, self.md5)
            candidate = osp.join(extract_dir, "dsgdb9nsd.xyz")
            if not osp.exists(candidate):
                # Try subdirectory
                candidate = osp.join(extract_dir, "qm9", "dsgdb9nsd.xyz")
            if not osp.exists(candidate):
                # Merge individual .xyz files
                xyz_files = sorted([
                    f for f in os.listdir(extract_dir) if f.endswith(".xyz")
                ])
                if not xyz_files:
                    raise RuntimeError(f"No .xyz files found in {extract_dir}")
                logger.info(f"Merging {len(xyz_files)} xyz files into {merged_xyz} ...")
                with open(merged_xyz, "w") as fout:
                    for fname in tqdm(xyz_files, desc="Merging"):
                        fpath = osp.join(extract_dir, fname)
                        with open(fpath, "r") as fin:
                            fout.write(fin.read())
            else:
                if osp.exists(candidate) and not osp.exists(merged_xyz):
                    import shutil
                    shutil.copy2(candidate, merged_xyz)

        # ---- 2. Build offset index + parse properties ----
        self._offsets = self._build_merged_index(merged_xyz)
        total = len(self._offsets)

        raw_props = self._parse_all_properties(merged_xyz, self._offsets, self.property_names)

        # ---- 3. Pre-computed split indices (saved once on rank 0) ----
        split_dir = osp.join(raw_dir, "splits")
        indices_file = osp.join(split_dir, f"split_{split}.npy")
        indices_all_file = osp.join(split_dir, "split_all.npy")

        if not osp.exists(indices_file) and not osp.exists(indices_all_file):
            if dist.get_rank() == 0:
                os.makedirs(split_dir, exist_ok=True)
                rng = np.random.default_rng(42)
                perm = rng.permutation(total)
                ts, vs = self._TRAIN_SIZE, self._VAL_SIZE
                np.save(osp.join(split_dir, "split_train.npy"), np.sort(perm[:ts]))
                np.save(osp.join(split_dir, "split_val.npy"),
                        np.sort(perm[ts:ts + vs]))
                np.save(osp.join(split_dir, "split_test.npy"),
                        np.sort(perm[ts + vs:]))
                np.save(osp.join(split_dir, "split_all.npy"), np.sort(perm))
            if dist.is_initialized():
                dist.barrier()

        if split is not None:
            self._indices = np.load(indices_file)
        else:
            self._indices = np.load(indices_all_file)

        # ---- 4. Cache graph cache path ----
        if build_graph_cfg is not None:
            gc_name = build_graph_cfg.get("__class_name__", "custom")
            cutoff = build_graph_cfg.get("__init_params__", {}).get("cutoff", 5)
            base_cache = cache_path if cache_path is not None else path
            graph_cache_dir = osp.join(base_cache, f"qm9_graphs_{gc_name}_cutoff{cutoff}")
        else:
            graph_cache_dir = None

        # ---- 5. Pre‑build edge_index + triplet indices (rank 0 + barrier, parallel) ----
        if build_graph_cfg is not None:
            self.graph_paths = [None] * total  # placeholder for all samples
            if not osp.exists(graph_cache_dir):
                os.makedirs(graph_cache_dir, exist_ok=True)
            cfg_pkl = osp.join(graph_cache_dir, "build_graph_cfg.pkl")
            cache_ready = osp.exists(cfg_pkl) and not overwrite

            need_rebuild = not cache_ready
            if cache_ready:
                try:
                    cfg_cached = self.load_from_cache(cfg_pkl)
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
                    self.save_to_cache(cfg_pkl, build_graph_cfg)
                    logger.info(
                        f"Pre‑building graphs for QM9 ({total} molecules) "
                        f"with 24 threads ..."
                    )
                    with ThreadPoolExecutor(max_workers=24) as executor:
                        futures = {
                            executor.submit(
                                _build_qm9_graph_thread,
                                i, merged_xyz, self._offsets,
                                build_graph_cfg, graph_cache_dir,
                            ): i for i in range(total)
                        }
                        for _ in tqdm(
                            as_completed(futures), total=total, desc="Build graphs"
                        ):
                            pass
                if dist.is_initialized():
                    dist.barrier()
            for i in range(total):
                self.graph_paths[i] = osp.join(graph_cache_dir, f"{i:010d}.pkl")
        else:
            self.graph_paths = None

        # ---- 6. Filter invalid ----
        self._raw_properties = raw_props
        self._merged_xyz_path = merged_xyz
        self._valid_indices = self._filter_invalid(self._indices)
        self.num_samples = len(self._valid_indices)
        logger.info(f"Load {self.num_samples} samples, targets={self.property_names}, split={split}")

    # --- index helpers ---

    @staticmethod
    def _build_merged_index(merged_path):
        index = []
        with open(merged_path, "r") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    n_atoms = int(line.strip())
                except ValueError:
                    # Some BOS .xyz files pack multiple molecules per file
                    # with tab separators instead of newlines. Scan forward
                    # to the next atom-count line.
                    continue
                for _ in range(2 + n_atoms):
                    f.readline()
                index.append(offset)
        return index

    @staticmethod
    def _read_one_molecule(path, offsets, idx):
        offset = offsets[idx]
        with open(path, "r") as f:
            f.seek(offset)
            n_atoms = int(f.readline().strip())
            prop_line = f.readline()
            atom_lines = [f.readline() for _ in range(n_atoms)]
        lines = [f"{n_atoms}\n", prop_line] + atom_lines
        z, pos, _ = _parse_qm9_xyz(lines)
        return z, pos

    @staticmethod
    def _parse_all_properties(merged_path, offsets, property_names):
        for name in property_names:
            if name not in _QM9_PROP_NAMES:
                raise ValueError(
                    f"Unknown QM9 property '{name}'. Available: {_QM9_PROP_NAMES}"
                )
        name_to_idx = {name: k + 2 for k, name in enumerate(_QM9_PROP_NAMES)}
        props = {name: np.zeros(len(offsets), dtype=np.float32) for name in property_names}
        with open(merged_path, "r") as f:
            for i, offset in enumerate(offsets):
                f.seek(offset)
                f.readline()
                prop_line = f.readline().strip().replace("*^", "e")
                parts = prop_line.split()
                raw_vals = []
                for p in parts:
                    try:
                        raw_vals.append(float(p))
                    except ValueError:
                        raw_vals.append(0.0)
                for name in property_names:
                    idx = name_to_idx[name]
                    props[name][i] = raw_vals[idx] if idx < len(raw_vals) else 0.0
        return props

    def _filter_invalid(self, indices):
        keep = []
        for idx in indices:
            valid = True
            for name in self.property_names:
                val = self._raw_properties[name][idx]
                if val is None or (isinstance(val, (float, np.floating)) and
                                    (math.isnan(val) or math.isinf(val))):
                    valid = False
                    break
            if valid:
                keep.append(idx)
        return np.array(keep, dtype=np.int64)

    def save_to_cache(self, path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_from_cache(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # --- public API ---

    def __getitem__(self, idx):
        real_idx = self._valid_indices[idx]
        z, pos = self._read_one_molecule(self._merged_xyz_path, self._offsets, real_idx)
        data = {"z": z, "pos": pos}
        if self.graph_paths is not None:
            gpath = self.graph_paths[real_idx]
            graph = self.load_from_cache(gpath) if isinstance(gpath, str) else gpath
            if isinstance(graph, dict):
                # New format: dict with edge_index + precomputed triplet indices
                data["edge_index"] = graph["edge_index"]
                data["triplet_indices"] = {
                    'i': graph['ti_i'],
                    'j': graph['ti_j'],
                    'idx_kj': graph['ti_idx_kj'],
                    'idx_ji': graph['ti_idx_ji'],
                    'idx_lk': graph['ti_idx_lk'],
                    'idx_triplet': graph['ti_idx_triplet'],
                }
            else:
                # Old format: plain numpy array (edge_index only, no triplet cache)
                data["edge_index"] = graph
        for name in self.property_names:
            data[name] = np.array([self._raw_properties[name][real_idx]], dtype=np.float32)
        data["id"] = int(real_idx)
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return self.num_samples
