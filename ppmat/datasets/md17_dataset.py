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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.custom_data_type import ConcatData
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
        split_file (Optional[str]): Preprocessed numpy index file for the
            selected split. Defaults to ``None``.
        build_molecule_cfg (Optional[Dict]): Configuration dict for molecule
            converter. Defaults to ``None``.
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
        split_file: Optional[str] = None,
        build_molecule_cfg: Optional[Dict] = None,
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
        if build_molecule_cfg is None:
            build_molecule_cfg = {
                "format": "dict",
                "sanitize": False,
                "add_hs": False,
                "remove_hs": False,
                "kekulize": False,
                "num_cpus": 1,
            }
            logger.message(
                "The build_molecule_cfg is not set, will use the default "
                f"configs: {build_molecule_cfg}"
            )
        self.build_molecule_cfg = build_molecule_cfg

        # ---- 1. Read full trajectory and selected frame indices ----
        npz_path = self._resolve_data_path(path, name)
        self.path = npz_path
        self.row_data, total = self.read_data(npz_path)
        self._indices = self._load_split_indices(
            npz_path, name, split, split_file, total
        )
        self.num_samples = len(self._indices)

        # ---- 2. Cache path ----
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            base_dir = osp.split(npz_path)[0]
            base_name = osp.splitext(osp.basename(npz_path))[0]
            split_suffix = split if split is not None else "all"
            self.cache_path = osp.join(
                f"{base_dir}_cache", f"{base_name}_{split_suffix}"
            )
        logger.info(f"Cache path: {self.cache_path}")

        # ---- 3. Pre-build molecules and graphs (MPtrj-style cache) ----
        self.cache_exists = True if osp.exists(self.cache_path) else False
        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_molecule_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl")
                )
                if not is_equal(build_molecule_cfg_cache, build_molecule_cfg):
                    logger.warning(
                        "build_molecule_cfg differs from cache. Rebuilding."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_molecule_cfg.pkl from cache. "
                    "Will rebuild the molecules and graphs(if need)."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if not is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.warning(
                            "build_graph_cfg differs from cache. Rebuilding."
                        )
                        overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load build_graph_cfg.pkl from cache. "
                        "Will rebuild the graphs."
                    )
                    overwrite = True

        molecule_cache_path = osp.join(self.cache_path, "molecules")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not self.cache_exists:
            if dist.get_rank() == 0:
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl"),
                    build_molecule_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )

                molecule_data = [
                    {
                        "atomic_numbers": self.row_data["z"],
                        "positions": self.row_data["pos"][frame],
                    }
                    for frame in self._indices
                ]
                molecules = BuildMolecule(**build_molecule_cfg)(molecule_data)
                os.makedirs(molecule_cache_path, exist_ok=True)
                for i, mol in enumerate(molecules):
                    self.save_to_cache(
                        osp.join(molecule_cache_path, f"{i:010d}.pkl"), mol
                    )
                logger.info(
                    f"Save {self.num_samples} molecules to {molecule_cache_path}"
                )

                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(molecules)
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")
            if dist.is_initialized():
                dist.barrier()

        self.molecules = [
            osp.join(molecule_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None
        assert (
            len(self.molecules) == self.num_samples
        ), "The number of molecules must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        logger.info(f"Load {self.num_samples} samples, split={split}")

    def _resolve_data_path(self, path, name):
        if osp.isfile(path):
            return path

        candidates = [
            osp.join(path, f"{name}_dft.npz"),
            osp.join(path, self.name, f"{name}_dft.npz"),
        ]
        if name in _BUNDLE_NPZ_MAP:
            candidates.extend(
                [
                    osp.join(path, _BUNDLE_NPZ_MAP[name]),
                    osp.join(path, self.name, _BUNDLE_NPZ_MAP[name]),
                ]
            )
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate

        logger.message("The dataset is not found. Will download it now.")
        root_path = download.get_datasets_path_from_url(self.url, self.md5)
        root_paths = [root_path]
        if not osp.exists(root_path):
            root_paths.append(osp.dirname(root_path))

        candidates = []
        for candidate_root in root_paths:
            candidates.extend(
                [
                    osp.join(candidate_root, self.name, f"{name}_dft.npz"),
                    osp.join(candidate_root, f"{name}_dft.npz"),
                ]
            )
            if name in _BUNDLE_NPZ_MAP:
                candidates.extend(
                    [
                        osp.join(
                            candidate_root, self.name, _BUNDLE_NPZ_MAP[name]
                        ),
                        osp.join(candidate_root, _BUNDLE_NPZ_MAP[name]),
                    ]
                )
        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Cannot find MD17 npz file for molecule: {name}")

    def read_data(self, path):
        """Load all trajectory frames from the npz file."""
        data = np.load(path)
        row_data = {
            "z": data["z"],
            "pos": data["R"],
            "energy": data["E"],
            "force": data["F"],
        }
        return row_data, data["R"].shape[0]

    def _load_split_indices(self, path, name, split, split_file, total):
        """Load frame indices for the requested split."""
        if split is None and split_file is None:
            return np.arange(total, dtype=np.int64)
        if split_file is None:
            split_dir = osp.join(osp.dirname(path), "splits")
            key = split if split is not None else "all"
            split_file = osp.join(split_dir, f"{name}_{key}_idx.npy")
            if not osp.exists(split_file):
                split_file = osp.join(split_dir, f"split_{key}.npy")
        if not osp.exists(split_file):
            raise FileNotFoundError(f"No such split file: {split_file}")
        return np.load(split_file).astype(np.int64)

    def get_molecule_array(self, molecule):
        conf = molecule.GetConformer()
        z = np.array(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int64
        )
        pos = np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(molecule.GetNumAtoms())
            ],
            dtype=np.float32,
        )
        return {
            "z": ConcatData(z),
            "pos": ConcatData(pos),
            "num_atoms": ConcatData(np.array([z.shape[0]], dtype=np.int64)),
        }

    def save_to_cache(self, cache_path: str, obj: Any):
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
            self.energy_key: np.array(
                [float(self.row_data["energy"][frame])], dtype=np.float32
            ),
            self.force_key: ConcatData(self.row_data["force"][frame]),
        }
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            sample["graph"] = graph
        else:
            mol = self.load_from_cache(self.molecules[idx])
            sample.update(self.get_molecule_array(mol))
        sample["id"] = int(frame)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.num_samples
