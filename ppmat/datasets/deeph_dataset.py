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

import json
import os
import os.path as osp
import pickle
import time
from configparser import ConfigParser
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.crystal import lattices_to_params_shape_numpy
from ppmat.utils.misc import is_equal


def _as_tuple(config_files) -> Tuple[str, ...]:
    if config_files is None:
        return ()
    if isinstance(config_files, str):
        return (config_files,)
    return tuple(config_files)


def _load_config(config_files=None, config=None) -> ConfigParser:
    if config is not None:
        config = _to_container(config)
        parser = ConfigParser()
        for section, values in config.items():
            parser.add_section(section)
            for key, value in values.items():
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                parser.set(section, key, str(value))
        return parser

    config = ConfigParser()
    read_files = config.read(list(_as_tuple(config_files)))
    if not read_files:
        raise FileNotFoundError(f"Can not read DeepH config files: {config_files}")
    return config


def _np_dtype(dtype: str):
    if dtype == "float32":
        return np.float32
    if dtype == "float16":
        return np.float16
    if dtype == "float64":
        return np.float64
    raise ValueError(f"Unknown DeepH dtype: {dtype}")


def _to_container(config):
    if config is None:
        return None
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _list_structure_folders(raw_data_dir: str, interface: str, nums: int | None = None):
    if interface != "npz":
        raise NotImplementedError(
            "DeepHDataset currently supports the validated npz Hamiltonian format."
        )
    folder_list = sorted(
        root for root, _, files in os.walk(raw_data_dir) if "rc.npz" in files
    )
    if nums is not None:
        folder_list = folder_list[:nums]
    return folder_list


def _load_structure_arrays(folder: str):
    lattice = np.loadtxt(os.path.join(folder, "lat.dat")).T
    atom_types = np.loadtxt(os.path.join(folder, "element.dat")).astype(int).tolist()
    cart_coords = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
    frac_coords = cart_coords @ np.linalg.inv(lattice)
    lengths, angles = lattices_to_params_shape_numpy(lattice)
    return {
        "lengths": lengths,
        "angles": angles,
        "atom_types": atom_types,
        "cart_coords": cart_coords,
        "frac_coords": frac_coords,
    }


class DeepHDataset(paddle.io.Dataset):
    """DeepH dataset adapter for PaddleMaterials.

    This adapter exposes PaddleMaterials geometric `Data` objects and keeps the
    LCMP subgraph metadata required by DeepH.
    """

    _CACHE: Dict[Tuple, Dict] = {}
    _STRUCTURE_CACHE: Dict[Tuple[str, str], Dict] = {}

    def __init__(
        self,
        split: str,
        config_files=None,
        config=None,
        nums: int | None = None,
        split_seed: int | None = None,
        build_structure_cfg: Dict[str, Any] | None = None,
        build_graph_cfg: Dict[str, Any] | None = None,
        cache_path: str | None = None,
        overwrite: bool = False,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")
        if config is None and config_files is None:
            raise ValueError("Either `config` or `config_files` must be provided.")

        self.config_files = _as_tuple(config_files)
        self.config = _to_container(config)
        self.split = split
        self.nums = nums
        self.split_seed = int(split_seed) if split_seed is not None else None
        self.build_structure_cfg = _to_container(build_structure_cfg) or {
            "format": "array",
            "primitive": False,
            "niggli": False,
            "canocial": True,
        }
        self.build_graph_cfg = _to_container(build_graph_cfg)
        self.cache_path = cache_path
        self.overwrite = overwrite

        shared = self._load_shared()
        self.dataset = shared["dataset"]
        self.folder_list = shared["folder_list"]
        self.indices = shared["split_indices"][split]
        self.info = shared["info"]

    def _load_shared(self) -> Dict:
        cache_key = (
            self.config_files,
            json.dumps(self.config, sort_keys=True),
            self.nums,
            self.split_seed or -1,
            json.dumps(self.build_structure_cfg, sort_keys=True),
            json.dumps(self.build_graph_cfg, sort_keys=True),
            self.cache_path or "",
            int(self.overwrite),
        )
        shared = self._CACHE.get(cache_key)
        if shared is not None:
            return shared

        config = _load_config(self.config_files, self.config)
        folder_list = _list_structure_folders(
            config.get("basic", "raw_dir"),
            config.get("basic", "interface"),
            self.nums,
        )

        dataset, dataset_info = self._load_or_build_graphs(config, folder_list)

        target = config.get("basic", "target")
        if target != "hamiltonian":
            raise NotImplementedError(
                "DeepHDataset currently supports the validated hamiltonian target."
            )
        orbital = json.loads(config.get("basic", "orbital"))
        dataset = self._make_hamiltonian_mask(
            dataset,
            orbital=orbital,
            num_orbital=len(orbital),
            spinful=dataset_info["spinful"],
            index_to_Z=dataset_info["index_to_Z"],
        )

        dataset_size = len(dataset)
        train_size = int(config.getfloat("train", "train_ratio") * dataset_size)
        val_size = int(config.getfloat("train", "val_ratio") * dataset_size)
        test_size = int(config.getfloat("train", "test_ratio") * dataset_size)

        seed = (
            self.split_seed
            if self.split_seed is not None
            else config.getint("basic", "seed", fallback=42)
        )
        rng = np.random.RandomState(seed)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        split_indices = {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": indices[train_size + val_size : train_size + val_size + test_size],
        }

        shared = {
            "dataset": dataset,
            "folder_list": folder_list,
            "split_indices": split_indices,
            "info": {
                "num_species": len(dataset_info["index_to_Z"]),
                "spinful": dataset_info["spinful"],
                "dataset_size": dataset_size,
                "config_files": list(self.config_files),
            },
        }
        self._CACHE[cache_key] = shared
        return shared

    def _graph_cache_path(self, config) -> str:
        dataset_name = config.get("basic", "dataset_name")
        interface = config.get("basic", "interface")
        num_l = config.getint("network", "num_l")
        radius = config.getfloat("graph", "radius")
        max_num_nbr = config.getint("graph", "max_num_nbr")
        suffix = f"{radius}r{max_num_nbr}mn"
        if config.getboolean("graph", "create_from_DFT", fallback=True):
            suffix = "FromDFT"
        nums_suffix = "all" if self.nums is None else f"n{self.nums}"
        if self.cache_path is not None:
            return self.cache_path
        return osp.join(
            config.get("basic", "graph_dir"),
            f"PPMatDeepHGraph-{interface}-{dataset_name}-{num_l}l-{suffix}-{nums_suffix}",
        )

    def _load_or_build_graphs(self, config, folder_list):
        cache_path = self._graph_cache_path(config)
        graph_cache_path = osp.join(cache_path, "graphs.pkl")
        cfg_cache_path = osp.join(cache_path, "dataset_cfg.pkl")
        expected_cfg = self._cache_cfg(config, folder_list)

        if osp.exists(graph_cache_path) and not self.overwrite:
            try:
                cached_cfg = self.load_from_cache(cfg_cache_path)
                if is_equal(cached_cfg, expected_cfg):
                    loaded = self.load_from_cache(graph_cache_path)
                    return loaded["graphs"], loaded["info"]
                logger.warning("DeepH cache config differs from current settings.")
            except Exception as e:
                logger.warning(e)
                logger.warning("Failed to load DeepH graph cache, will rebuild.")

        begin = time.time()
        graphs = [
            self._build_graph_from_folder(folder, config) for folder in folder_list
        ]
        index_to_Z, Z_to_index = self._element_statistics(graphs)
        spinful = bool(graphs[0].spinful)
        for graph in graphs:
            assert spinful == graph.spinful

        info = {
            "spinful": spinful,
            "index_to_Z": index_to_Z,
            "Z_to_index": Z_to_index,
        }
        self.save_to_cache(cfg_cache_path, expected_cfg)
        self.save_to_cache(graph_cache_path, {"graphs": graphs, "info": info})
        logger.info(
            "Finish building PaddleMaterials graph cache with "
            f"{len(graphs)} structures, "
            f"cost {time.time() - begin:.0f} seconds"
        )
        return graphs, info

    def _cache_cfg(self, config, folder_list):
        return {
            "config_files": list(self.config_files),
            "folders": list(folder_list),
            "nums": self.nums,
            "build_structure_cfg": self.build_structure_cfg,
            "build_graph_cfg": self._graph_converter_cfg(config),
            "interface": config.get("basic", "interface"),
            "target": config.get("basic", "target"),
            "num_l": config.getint("network", "num_l"),
            "dtype": config.get("hyperparameter", "dtype"),
            "radius": config.getfloat("graph", "radius"),
            "max_num_nbr": config.getint("graph", "max_num_nbr"),
        }

    def _graph_converter_cfg(self, config):
        if self.build_graph_cfg is not None:
            return self.build_graph_cfg
        return {
            "__class_name__": "FindPointsInSpheres",
            "__init_params__": {
                "cutoff": config.getfloat("graph", "radius"),
                "pbc": (1, 1, 1),
                "eps": 1e-8,
            },
        }

    def _deeph_graph_converter_cfg(self, config):
        return {
            "__class_name__": "DeepHGraphConverter",
            "__init_params__": {
                "radius": config.getfloat("graph", "radius"),
                "max_num_nbr": config.getint("graph", "max_num_nbr"),
                "default_dtype": _np_dtype(config.get("hyperparameter", "dtype")),
                "interface": config.get("basic", "interface"),
                "num_l": config.getint("network", "num_l"),
                "create_from_DFT": config.getboolean(
                    "graph", "create_from_DFT", fallback=True
                ),
                "if_lcmp_graph": config.getboolean(
                    "graph", "if_lcmp_graph", fallback=True
                ),
                "separate_onsite": config.getboolean(
                    "graph", "separate_onsite", fallback=False
                ),
                "target": config.get("basic", "target"),
                "huge_structure": False,
                "if_new_sp": config.getboolean("graph", "new_sp", fallback=False),
            },
        }

    def _build_graph_from_folder(self, folder: str, config):
        structure_info = self._load_structure(folder)
        structure = structure_info["structure"]
        create_from_dft = config.getboolean("graph", "create_from_DFT", fallback=True)
        converter_graph = None
        if not create_from_dft:
            structure_graph_converter = build_graph_converter(
                self._graph_converter_cfg(config)
            )
            converter_graph = structure_graph_converter(structure)
            if converter_graph is None:
                raise ValueError(f"Failed to build graph for DeepH structure: {folder}")
        deeph_converter = build_graph_converter(self._deeph_graph_converter_cfg(config))
        return deeph_converter(structure, folder, converter_graph)

    @staticmethod
    def _make_hamiltonian_mask(dataset, orbital, num_orbital, spinful, index_to_Z):
        dataset_mask = []
        for data in dataset:
            oij_value = data.term_real
            if not np.all(data.term_mask):
                raise NotImplementedError(
                    "Graph radius including hopping without calculation is not "
                    "supported."
                )

            if spinful:
                out_fea_len = num_orbital * 8
            else:
                out_fea_len = num_orbital

            mask = np.zeros((data.edge_attr.shape[0], out_fea_len), dtype=np.int8)
            label = np.zeros(
                (data.edge_attr.shape[0], out_fea_len), dtype=oij_value.dtype
            )
            atomic_number_edge_i = index_to_Z[data.x[data.edge_index[0]]]
            atomic_number_edge_j = index_to_Z[data.x[data.edge_index[1]]]

            for index_out, orbital_dict in enumerate(orbital):
                for n_m_str, a_b in orbital_dict.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(
                        int, n_m_str.split()
                    )
                    condition_orbital_i, condition_orbital_j = a_b
                    condition = (atomic_number_edge_i == condition_atomic_number_i) & (
                        atomic_number_edge_j == condition_atomic_number_j
                    )

                    if spinful:
                        mask[:, 8 * index_out : 8 * (index_out + 1)] = np.where(
                            condition[:, None],
                            1,
                            0,
                        )
                    else:
                        mask[:, index_out] += np.where(condition, 1, 0)

                    if spinful:
                        value = oij_value[:, condition_orbital_i, condition_orbital_j]
                        label[:, 8 * index_out : 8 * (index_out + 1)] = np.where(
                            condition[:, None],
                            value,
                            np.zeros_like(value),
                        )
                    else:
                        label[:, index_out] += np.where(
                            condition,
                            oij_value[:, condition_orbital_i, condition_orbital_j],
                            np.zeros(data.edge_attr.shape[0], dtype=oij_value.dtype),
                        )

            assert len(np.where((mask != 1) & (mask != 0))[0]) == 0
            data.mask = mask.astype(bool)
            del data.term_mask
            data.label = label
            del data.term_real
            dataset_mask.append(data)
        return dataset_mask

    @staticmethod
    def _element_statistics(data_list):
        index_to_Z = np.unique(data_list[0].x).astype(np.int64)
        Z_to_index = np.full((100,), -1, dtype=np.int64)
        Z_to_index[index_to_Z] = np.arange(len(index_to_Z), dtype=np.int64)
        for data in data_list:
            data.x = Z_to_index[data.x]
        return index_to_Z, Z_to_index

    def _load_structure(self, folder: str) -> Dict:
        structure_cache_key = (
            folder,
            json.dumps(self.build_structure_cfg, sort_keys=True),
        )
        cached = self._STRUCTURE_CACHE.get(structure_cache_key)
        if cached is not None:
            return cached

        arrays = _load_structure_arrays(folder)
        builder = BuildStructure(**self.build_structure_cfg)
        structure = BuildStructure.build_one(
            {
                "lengths": arrays["lengths"],
                "angles": arrays["angles"],
                "frac_coords": arrays["frac_coords"],
                "atom_types": arrays["atom_types"],
            },
            builder.format,
            builder.primitive,
            builder.niggli,
            builder.canocial,
        )
        cached = {
            "folder": folder,
            "structure": structure,
            "lattice": np.asarray(structure.lattice.matrix, dtype=np.float32),
            "frac_coords": np.asarray(structure.frac_coords, dtype=np.float32),
            "cart_coords": np.asarray(structure.cart_coords, dtype=np.float32),
            "atomic_numbers": np.asarray(structure.atomic_numbers, dtype=np.int64),
        }
        self._STRUCTURE_CACHE[structure_cache_key] = cached
        return cached

    @staticmethod
    def save_to_cache(cache_path: str, data: Any):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_cache(cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        graph_index = self.indices[index]
        graph = self.dataset[graph_index]
        folder = self.folder_list[graph_index]
        structure_info = self._load_structure(folder)

        if hasattr(graph, "subgraph_dict") and graph.subgraph_dict is not None:
            subgraph_dict = graph.subgraph_dict
        elif hasattr(graph, "subgraph") and graph.subgraph is not None:
            subgraph = graph.subgraph
            subgraph_dict = {
                "subgraph_atom_idx": subgraph[0],
                "subgraph_edge_idx": subgraph[1],
                "subgraph_edge_ang": subgraph[2],
                "subgraph_index": subgraph[3],
            }
        else:
            raise ValueError("DeepH sample does not contain LCMP subgraph metadata.")

        return {
            "x": ConcatData(np.asarray(graph.x, dtype=np.int64)),
            "edge_index": ConcatData(np.asarray(graph.edge_index, dtype=np.int64)),
            "edge_attr": ConcatData(np.asarray(graph.edge_attr, dtype=np.float32)),
            "batch": ConcatData(np.zeros(graph.x.shape[0], dtype=np.int64)),
            "label": ConcatData(np.asarray(graph.label, dtype=np.float32)),
            "mask": ConcatData(np.asarray(graph.mask, dtype=bool)),
            "pos": ConcatData(
                np.asarray(structure_info["cart_coords"], dtype=np.float32)
            ),
            "sub_atom_idx": ConcatData(
                np.asarray(subgraph_dict["subgraph_atom_idx"], dtype=np.int64)
            ),
            "sub_edge_idx": ConcatData(
                np.asarray(subgraph_dict["subgraph_edge_idx"], dtype=np.int64)
            ),
            "sub_edge_ang": ConcatData(
                np.asarray(subgraph_dict["subgraph_edge_ang"], dtype=np.float32)
            ),
            "sub_index": ConcatData(
                np.asarray(subgraph_dict["subgraph_index"], dtype=np.int64)
            ),
            "structure_lattice": ConcatData(
                np.asarray(structure_info["lattice"], dtype=np.float32).reshape(1, 3, 3)
            ),
            "structure_frac_coords": ConcatData(
                np.asarray(structure_info["frac_coords"], dtype=np.float32)
            ),
            "structure_atomic_numbers": ConcatData(
                np.asarray(structure_info["atomic_numbers"], dtype=np.int64)
            ),
            "structure_folder": structure_info["folder"],
        }
