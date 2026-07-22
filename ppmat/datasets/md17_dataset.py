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


class MD17Dataset(Dataset):
    """MD17 molecular dynamics trajectories for eight organic molecules.

    +----------------+----------+--------+-------+----------+-------+
    | Molecule       | #samples | #atoms | #tasks| #targets | Split |
    +================+==========+========+=======+==========+=======+
    | Aspirin        | 211,762  | 21     | 2     | E + F    | 1k/1k/R |
    | Benzene (old)  | 627,983  | 12     | 2     | E + F    | 1k/1k/R |
    | Ethanol        | 555,092  | 9      | 2     | E + F    | 1k/1k/R |
    | Malonaldehyde  | 993,237  | 9      | 2     | E + F    | 1k/1k/R |
    | Naphthalene    | 326,250  | 18     | 2     | E + F    | 1k/1k/R |
    | Salicylic      | 320,231  | 16     | 2     | E + F    | 1k/1k/R |
    | Toluene        | 442,790  | 15     | 2     | E + F    | 1k/1k/R |
    | Uracil         | 133,770  | 12     | 2     | E + F    | 1k/1k/R |
    +----------------+----------+--------+-------+----------+-------+

    Contains ab-initio molecular dynamics trajectories for eight small
    organic molecules.  Each frame provides atomic numbers, 3D positions,
    total energy, and per-atom forces.

    Data source: https://www.quantum-machine.org/datasets/

    Each NPZ file contains atomic numbers (z), positions (R), energies (E),
    and forces (F). The standard split contains 1,000 training and 1,000
    validation frames; all remaining frames form the test split.

    Args:
        path: Path to one molecule NPZ file.
        name: Molecule name used to locate its split indices.
        split: "train", "val", "test", or None for all frames.
        build_molecule_cfg: Configuration for BuildMolecule.
        build_graph_cfg: Configuration for the molecular graph converter.
        transforms: Optional per-sample transforms.
        cache_path: Optional molecule and graph cache path.
        overwrite: Whether to rebuild an existing cache.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz"
    md5 = "634cc25cc8a3fb0d99bd14245eb8dabd"
    name = "md17"

    def __init__(
        self,
        path: str,
        name: str = "benzene_old",
        split: Optional[str] = None,
        *,
        build_molecule_cfg: Optional[Dict] = None,
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            if not osp.exists(root_path):
                root_path = osp.dirname(root_path)
            path = osp.join(root_path, osp.basename(path))
            if not osp.exists(path):
                path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        self.transforms = transforms

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
        if build_graph_cfg is None:
            raise ValueError("MD17Dataset requires build_graph_cfg.")

        if split is None:
            sample_ids = None
        else:
            split_path = osp.join(
                osp.dirname(path), "splits", f"{name}_{split}_idx.npy"
            )
            sample_ids = np.load(split_path).astype(np.int64)

        self.raw_data, self.property_data, self.sample_ids = self.read_data(
            path, sample_ids
        )
        self.num_samples = len(self.sample_ids)
        logger.info(f"Load {self.num_samples} samples from {path}, split={split}")

        if cache_path is None:
            split_name = split if split is not None else "all"
            cache_path = osp.join(
                osp.dirname(path) + "_cache",
                f"{osp.splitext(osp.basename(path))[0]}_{split_name}",
            )
        self.cache_path = cache_path
        logger.info(f"Cache path: {self.cache_path}")

        cache_exists = osp.exists(self.cache_path)
        if cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. Existing cache settings will be checked before reuse."
            )
            try:
                cached_cfg = self.load_from_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl")
                )
                if not is_equal(cached_cfg, build_molecule_cfg):
                    logger.warning("build_molecule_cfg differs from cache. Rebuilding.")
                    overwrite = True
            except Exception as error:
                logger.warning(error)
                overwrite = True

            if not overwrite:
                try:
                    cached_cfg = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if not is_equal(cached_cfg, build_graph_cfg):
                        logger.warning(
                            "build_graph_cfg differs from cache. Rebuilding."
                        )
                        overwrite = True
                except Exception as error:
                    logger.warning(error)
                    overwrite = True

        molecule_cache_path = osp.join(self.cache_path, "molecules")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        if overwrite or not cache_exists:
            if dist.get_rank() == 0:
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl"),
                    build_molecule_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"),
                    build_graph_cfg,
                )

                molecule_data = [
                    {
                        "atomic_numbers": self.raw_data["z"],
                        "positions": positions,
                    }
                    for positions in self.raw_data["pos"]
                ]
                molecules = BuildMolecule(**build_molecule_cfg)(molecule_data)
                os.makedirs(molecule_cache_path, exist_ok=True)
                for index, molecule in enumerate(molecules):
                    self.save_to_cache(
                        osp.join(molecule_cache_path, f"{index:010d}.pkl"),
                        molecule,
                    )
                logger.info(
                    f"Save {self.num_samples} molecules to {molecule_cache_path}"
                )

                graphs = build_graph_converter(build_graph_cfg)(molecules)
                os.makedirs(graph_cache_path, exist_ok=True)
                for index, graph in enumerate(graphs):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{index:010d}.pkl"),
                        graph,
                    )
                logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

            if dist.is_initialized():
                dist.barrier()

        self.graphs = [
            osp.join(graph_cache_path, f"{index:010d}.pkl")
            for index in range(self.num_samples)
        ]

    def read_data(self, path: str, sample_ids: Optional[np.ndarray]):
        """Read and retain only the selected trajectory frames."""
        with np.load(path) as data:
            positions = data["R"]
            if sample_ids is None:
                sample_ids = np.arange(positions.shape[0], dtype=np.int64)

            raw_data = {
                "z": np.asarray(data["z"], dtype=np.int64),
                "pos": np.asarray(positions[sample_ids], dtype=np.float32),
            }
            property_data = {
                "energy": np.asarray(
                    data["E"][sample_ids], dtype=np.float32
                ).reshape(-1),
                "force": np.asarray(data["F"][sample_ids], dtype=np.float32),
            }
        return raw_data, property_data, sample_ids

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    def __getitem__(self, idx: int):
        data = {
            "graph": self.load_from_cache(self.graphs[idx]),
            "energy": np.asarray(
                [self.property_data["energy"][idx]], dtype=np.float32
            ),
            "force": ConcatData(self.property_data["force"][idx]),
            "id": int(self.sample_ids[idx]),
        }
        return self.transforms(data) if self.transforms is not None else data

    def __len__(self):
        return self.num_samples
