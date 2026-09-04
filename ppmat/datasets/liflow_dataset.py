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

from __future__ import absolute_import
from __future__ import annotations

import csv
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class LiFlowDataset(Dataset):
    """LiFlow Dataset Handler

    Utilities for loading and processing the LiFlow crystalline-transport trajectory
    dataset used by the LiFlow flow-matching models.

    **Dataset Overview**
    - **Source**: https://zenodo.org/records/14889658
      (DOI: https://doi.org/10.5281/zenodo.14889658)
    - **Reference code**: https://github.com/learningmatter-mit/liflow
    - Includes the universal MLIP set and LGPS. LPS data are available from the
      original authors upon request.

    **Data Format**
    After extracting the archive into ``path``, the directory should contain:

    | File | Description |
    |------|-------------|
    | `element_index.npy` | Element lookup table, shape `[n_elements]` |
    | `atomic_numbers.npy` | Dict of atomic numbers keyed by structure name |
    | `lattice.npy` | Dict of lattice matrices keyed by structure name |
    | `positions_{temp}K.npz` | Trajectory positions keyed by structure name |
    | `{train,test}_{temp}K.csv` | Trajectory index CSV files |

    The CSV index contains columns such as `name`, `temp`, `t_start`, `t_end`,
    `prior_Li`, and `prior_frame`.

    Each sample returns a trajectory pair separated by ``time_delay_steps``. Unique
    crystal objects are built with ``BuildStructure`` and serialized under
    ``cache_path``. Atomic masses used by the LiFlow temperature-mass prior are taken
    from the cached pymatgen ``Structure`` sites (same physical role as ``sqrt(kT/m)``
    scaling in the reference implementation).

    Args:
        path (str): Root directory of the LiFlow dataset. Defaults to
            ``"./data/liflow"``.
        index_file (str): Index CSV file name or absolute path. Relative paths are
            resolved under ``path``. Defaults to ``train_800K.csv``.
        time_delay_steps (int, optional): Frame gap between the two endpoints.
            Defaults to 100.
        prior_scale_li (Tuple[float, ...] or list, optional): Prior scale for Li atoms
            indexed by ``prior_Li``. Defaults to ``(1.0, 10.0)``.
        prior_scale_frame (Tuple[float, ...] or list, optional): Prior scale for frame
            atoms indexed by ``prior_frame``. Defaults to ``(0.316, 3.16)``.
        seed (int, optional): Random seed used for frame sampling and prior noise.
            Defaults to 42.
        random_time (bool, optional): If True (training), randomly sample the start
            frame inside ``[t_start, t_end - time_delay_steps)`` and the flow-matching
            time in ``[0, 1)``. If False (val / test / predict), use ``t_start`` and
            flow time ``0.5``. Defaults to True.
        build_structure_cfg (Dict, optional): Configs for ``BuildStructure``. Defaults
            to array format without lattice reduction so trajectory coordinates stay
            consistent. Defaults to None.
        cache_path (Optional[str], optional): Directory used to cache built structures,
            masses, and build configs. Defaults to ``<path>_cache/<index_stem>``.
        overwrite (bool, optional): Whether to rebuild an existing structure cache.
            Defaults to False.
        **kwargs: Reserved for compatibility with other dataset configs.
    """

    name = "liflow"
    url = None

    def __init__(
        self,
        path: str = "./data/liflow",
        index_file: str = "train_800K.csv",
        time_delay_steps: int = 100,
        prior_scale_li: Union[Tuple[float, ...], List[float]] = (1.0, 10.0),
        prior_scale_frame: Union[Tuple[float, ...], List[float]] = (0.316, 3.16),
        seed: int = 42,
        random_time: bool = True,
        build_structure_cfg: Optional[Dict] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ):
        super().__init__()

        if not osp.exists(path):
            raise FileNotFoundError(f"LiFlow dataset path not found: {path}")

        self.path = path
        index_path = index_file
        if not osp.isabs(index_path):
            index_path = osp.join(path, index_file)
        if not osp.exists(index_path):
            logger.error(f"LiFlow index file not found: {index_path}")
            raise FileNotFoundError(f"LiFlow index file not found: {index_path}")

        self.index_path = index_path
        self.time_delay_steps = int(time_delay_steps)
        self.prior_scale_li = np.asarray(prior_scale_li, dtype=np.float32)
        self.prior_scale_frame = np.asarray(prior_scale_frame, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.random_time = random_time
        self.overwrite = overwrite

        if build_structure_cfg is None:
            # 轨迹坐标与原始晶格对齐，不做约化，避免与 positions_*.npz 不一致
            build_structure_cfg = {
                "format": "array",
                "primitive": False,
                "niggli": False,
                "canocial": False,
                "num_cpus": 1,
            }
            logger.message(
                "The build_structure_cfg is not set, will use the default "
                f"configs: {build_structure_cfg}"
            )
        self.build_structure_cfg = build_structure_cfg

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            index_stem = osp.splitext(osp.basename(index_path))[0]
            self.cache_path = osp.join(path.rstrip("\\/") + "_cache", index_stem)
        logger.info(f"Cache path: {self.cache_path}")

        self.row_data = self.read_data(index_path)
        self.num_samples = len(self.row_data)
        logger.info(f"Load {self.num_samples} samples from {index_path}")

        (
            self.element_index,
            self.atomic_numbers,
            self.lattice,
            self.positions,
        ) = self.read_structure_data(path, self.row_data)

        self.structure_names = sorted({row["name"] for row in self.row_data})
        self.structures, self.masses = self.build_or_load_structures(
            self.structure_names
        )

    def read_data(self, index_path: str) -> List[Dict]:
        """Read trajectory index rows from a CSV file.

        Args:
            index_path (str): Path to the index CSV.

        Returns:
            List[Dict]: Parsed trajectory index rows.
        """
        with open(index_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        required = {"name", "temp", "t_start", "t_end"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                f"LiFlow index must contain columns {sorted(required)}, got "
                f"{list(rows[0].keys()) if rows else []}"
            )

        parsed = []
        for row in rows:
            item = dict(row)
            item["temp"] = int(float(row["temp"]))
            item["t_start"] = int(row["t_start"])
            item["t_end"] = int(row["t_end"])
            item["prior_Li"] = int(row.get("prior_Li") or 0)
            item["prior_frame"] = int(row.get("prior_frame") or 0)
            parsed.append(item)
        return parsed

    def read_structure_data(self, path: str, rows: List[Dict]):
        """Load element tables, lattices, and trajectory archives.

        Args:
            path (str): Dataset root directory.
            rows (List[Dict]): Parsed index rows used to decide which temperatures to
                load.

        Returns:
            Tuple: ``(element_index, atomic_numbers, lattice, positions)``.
        """
        element_index = np.load(osp.join(path, "element_index.npy"))
        atomic_numbers = np.load(
            osp.join(path, "atomic_numbers.npy"), allow_pickle=True
        ).item()
        lattice = np.load(osp.join(path, "lattice.npy"), allow_pickle=True).item()

        positions = {}
        for temp in sorted({row["temp"] for row in rows}):
            archive_path = osp.join(path, f"positions_{temp}K.npz")
            if not osp.exists(archive_path):
                logger.error(f"Trajectory archive not found: {archive_path}")
                raise FileNotFoundError(f"Trajectory archive not found: {archive_path}")
            positions[temp] = np.load(archive_path)
            logger.info(f"Load trajectory archive from {archive_path}")
        return element_index, atomic_numbers, lattice, positions

    def _structure_cache_files(self, name: str) -> Tuple[str, str]:
        safe_name = name.replace("/", "_").replace("\\", "_")
        structure_dir = osp.join(self.cache_path, "structures")
        mass_dir = osp.join(self.cache_path, "masses")
        return (
            osp.join(structure_dir, f"{safe_name}.pkl"),
            osp.join(mass_dir, f"{safe_name}.pkl"),
        )

    def _reference_cart_coords(self, name: str) -> np.ndarray:
        """取该结构在索引中的首帧笛卡尔坐标，用于构建 Structure。"""
        for row in self.row_data:
            if row["name"] != name:
                continue
            trajectory = self.positions[row["temp"]][name]
            return np.asarray(trajectory[row["t_start"]], dtype=np.float32)
        raise KeyError(f"No index row found for structure name: {name}")

    def _array_inputs_for_build(self, names: List[str]) -> List[Dict]:
        """构造 BuildStructure(format=array) 所需输入列表。"""
        crystals = []
        for name in names:
            lattice = np.asarray(self.lattice[name], dtype=np.float64)
            atomic_numbers = np.asarray(self.atomic_numbers[name], dtype=np.int64)
            cart = self._reference_cart_coords(name)
            frac = cart @ np.linalg.inv(lattice)
            crystals.append(
                {
                    "frac_coords": frac,
                    "atom_types": atomic_numbers.tolist(),
                    "lattice": lattice,
                }
            )
        return crystals

    def build_or_load_structures(self, names: List[str]):
        """使用 BuildStructure 构建晶体对象，并做序列化 cache。

        Args:
            names (List[str]): 当前索引涉及的唯一结构名。

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: 结构与原子质量的 cache 路径字典。
        """
        cache_exists = osp.exists(self.cache_path)
        overwrite = self.overwrite
        if cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used match your current settings."
            )
            try:
                build_structure_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl")
                )
                if is_equal(build_structure_cfg_cache, self.build_structure_cfg):
                    logger.info(
                        "The cached build_structure_cfg configuration matches "
                        "the current settings. Reusing previously generated "
                        "structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_structure_cfg is different from "
                        "build_structure_cfg_cache. Will rebuild the structures."
                    )
                    overwrite = True
            except Exception as error:
                logger.warning(error)
                logger.warning(
                    "Failed to load build_structure_cfg.pkl from cache. "
                    "Will rebuild the structures."
                )
                overwrite = True

            if not overwrite:
                missing = [
                    name
                    for name in names
                    if not (
                        osp.exists(self._structure_cache_files(name)[0])
                        and osp.exists(self._structure_cache_files(name)[1])
                    )
                ]
                if missing:
                    logger.warning(
                        f"Missing cached structures for {len(missing)} names. "
                        "Will rebuild the structures."
                    )
                    overwrite = True

        structure_paths = {}
        mass_paths = {}
        for name in names:
            structure_path, mass_path = self._structure_cache_files(name)
            structure_paths[name] = structure_path
            mass_paths[name] = mass_path

        if overwrite or not cache_exists:
            if dist.get_rank() == 0:
                os.makedirs(self.cache_path, exist_ok=True)
                os.makedirs(osp.join(self.cache_path, "structures"), exist_ok=True)
                os.makedirs(osp.join(self.cache_path, "masses"), exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    self.build_structure_cfg,
                )

                crystals_data = self._array_inputs_for_build(names)
                structures = BuildStructure(**self.build_structure_cfg)(crystals_data)
                for name, structure in zip(names, structures):
                    structure_path, mass_path = self._structure_cache_files(name)
                    # 原子质量用于 prior 缩放：scale ∝ sqrt(k_B * T / m)
                    masses = np.asarray(
                        [float(site.specie.atomic_mass) for site in structure],
                        dtype=np.float32,
                    )
                    self.save_to_cache(structure_path, structure)
                    self.save_to_cache(mass_path, masses)
                logger.info(
                    f"Save {len(names)} structures and masses to {self.cache_path}"
                )

            if dist.is_initialized():
                dist.barrier()

        return structure_paths, mass_paths

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """Build one noised trajectory-pair sample.

        Args:
            idx (int): Sample index.

        Returns:
            Dict: Model input fields for one LiFlow training / prediction sample.
        """
        row = self.row_data[idx]
        last_start = row["t_end"] - self.time_delay_steps - 1
        if last_start < row["t_start"]:
            raise ValueError(
                f"Trajectory interval for {row['name']} is shorter than "
                f"time_delay_steps={self.time_delay_steps}."
            )

        if self.random_time:
            start = int(self.rng.integers(row["t_start"], last_start + 1))
            flow_time = float(self.rng.random())
        else:
            start = row["t_start"]
            flow_time = 0.5
        end = start + self.time_delay_steps

        # 静态量来自 BuildStructure cache；轨迹帧仍从 positions archive 读取
        structure = self.load_from_cache(self.structures[row["name"]])
        masses = np.asarray(
            self.load_from_cache(self.masses[row["name"]]), dtype=np.float32
        )
        atomic_numbers = np.asarray(
            [int(site.specie.Z) for site in structure], dtype=np.int64
        )
        elements = np.asarray(self.element_index[atomic_numbers], dtype=np.int64)
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)

        trajectory = self.positions[row["temp"]][row["name"]]
        positions_1 = np.asarray(trajectory[start], dtype=np.float32)
        positions_2 = np.asarray(trajectory[end], dtype=np.float32)

        li_scale = self.prior_scale_li[row["prior_Li"]]
        frame_scale = self.prior_scale_frame[row["prior_frame"]]
        prefactor = np.where(atomic_numbers == 3, li_scale, frame_scale)
        # LiFlow prior：温度-质量条件高斯噪声，k_B 单位 eV/K
        scale = prefactor * np.sqrt(8.617333262e-5 * row["temp"] / masses)
        prior = self.rng.normal(scale=scale[:, None], size=positions_1.shape).astype(
            np.float32
        )
        source = positions_1 + prior
        target = positions_2 - source

        data = {
            "positions_1": ConcatNumpyWarper(positions_1),
            "positions_2": ConcatNumpyWarper(positions_2),
            "prior": ConcatNumpyWarper(prior),
            "target": ConcatNumpyWarper(target.astype(np.float32)),
            "time": np.asarray(flow_time, dtype=np.float32),
            "temp": np.asarray(row["temp"], dtype=np.float32),
            "elements": ConcatNumpyWarper(elements),
            "atomic_numbers": ConcatNumpyWarper(atomic_numbers),
            "lattice": lattice,
            "num_atoms": np.asarray(len(elements), dtype=np.int64),
            "name": row["name"],
            "frame_start": np.asarray(start, dtype=np.int64),
            "frame_end": np.asarray(end, dtype=np.int64),
        }
        return data
