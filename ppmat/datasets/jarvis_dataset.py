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

import json
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
import zipfile
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle.distributed as dist
from jarvis.db.figshare import data as jdata
from jarvis.db.figshare import get_db_info
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.misc import is_equal

# -----------------------------------------------------------------------------
# JARVIS mirror dataset registry (preferred download entries)
# -----------------------------------------------------------------------------
JARVIS_MIRROR_DATASETS = [
    {
        "name": "dft_3d_2021",
        "url": "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/jarvis/jarvis_dft_3d-8-18-2021.json.zip",  # noqa
        "md5": "8f619035a2cd8030de1ce38ce8b561b2",
    },
    {
        "name": "alexandria_scan_3d_2024.10.1_jarvis_tools",
        "url": "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/jarvis/jarvis_alexandria_scan_3d_2024.10.1_jarvis_tools.json.zip",  # noqa
        "md5": "ddeee1df79789d8f2b4a89f625864e6b",
    },
    {
        "name": "cfid_3d",
        "url": "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/jarvis/jarvis_cfid_3d-8-18-2021.json.zip",  # noqa
        "md5": "6efe75ca51aa5fb5c23a5b08fb412a6e",
    },
    {
        "name": "dft_2d",
        "url": "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/jarvis/jdft_2d-4-26-2020.zip",  # noqa
        "md5": "022c6e321bef034f5bff40e67c81f483",
    },
]


class JarvisDataset(Dataset):
    """Jarvis Dataset Handler.

    Compatible with standard Jarvis datasets and specific URL-based datasets (e.g. 2D).
    """

    def __init__(
        self,
        path: str,
        jarvis_data_name: str = "custom",  # Default to custom if url is provided
        property_names: Union[str, List[str]] = None,
        url: Optional[str] = None,  # New argument
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.url = url

        # 1. Determine Path and Filename logic
        # If URL is explicitly provided (Adapter logic), use it to determine filename
        if self.url is not None:
            zip_basename = osp.basename(self.url)
            # e.g. jdft_2d-4-26-2020.zip
            self.path = osp.join(path, zip_basename)
            logger.info(f"Using provided URL: {self.url}")
        else:
            # Original logic: Lookup via jarvis_data_name
            db_info = get_db_info()
            if jarvis_data_name not in db_info:
                raise ValueError(f"Unknown dataset name: {jarvis_data_name}")

            _, jarvis_data_filename, _, _ = db_info[jarvis_data_name]
            self.path = osp.join(path, jarvis_data_filename + ".zip")

        # Obtain property names
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names if property_names is not None else []

        # Handle structure_cfg
        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "jarvis",
                "primitive": False,
                "niggli": True,
                "num_cpus": 1,
            }
            logger.message(
                "The build_structure_cfg is not set, will use the default "
                f"configs: {build_structure_cfg}"
            )
        self.build_structure_cfg = build_structure_cfg

        self.build_graph_cfg = build_graph_cfg

        # Determine cache directory name suffix
        if build_graph_cfg is not None:
            graph_converter_name = re.sub(
                r"(?<!^)([A-Z])", r"_\1", build_graph_cfg["__class_name__"]
            ).lower()
            cutoff_name = str(int(build_graph_cfg["__init_params__"]["cutoff"]))
        else:
            graph_converter_name = "none"
            cutoff_name = "none"

        # Construct Cache Path
        if cache_path is not None:
            base_cache_dir = cache_path
        else:
            base_cache_dir = path  # default to dataset root

        self.cache_path = osp.join(
            base_cache_dir,
            jarvis_data_name
            + "_cache_"
            + graph_converter_name
            + "_cutoff_"
            + cutoff_name,
        )

        logger.info(f"Cache path: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        # Additional parameters
        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # compute number of samples of raw file
        if osp.exists(self.path) and zipfile.is_zipfile(self.path):
            try:
                with zipfile.ZipFile(self.path) as zf:
                    # Logic to find json: check for name matching zip, or first .json
                    expected_member = os.path.splitext(os.path.basename(self.path))[0]
                    try:
                        bytes_data = zf.read(expected_member)
                    except KeyError:
                        json_members = [n for n in zf.namelist() if n.endswith(".json")]
                        if not json_members:
                            raise RuntimeError(
                                "No .json file found inside the zip archive."
                            )
                        bytes_data = zf.read(json_members[0])
                num_samples_raw_file = len(json.loads(bytes_data))
                logger.info(f"The raw file has {num_samples_raw_file} samples.")
            except Exception as e:
                logger.warning(str(e))
                logger.warning("The raw file is corrupted.")
                num_samples_raw_file = 0
        else:
            num_samples_raw_file = 0
            logger.warning("The raw file is not found.")

        # check if properties have been cached
        property_cache_path = osp.join(self.cache_path, "properties")
        if osp.exists(property_cache_path):
            try:
                for property_name in self.property_names:
                    data = self.load_from_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"),
                    )
                    logger.info(
                        f"Load {len(data)} {property_name} values "
                        f"from {property_cache_path}"
                    )
                    if len(data) != num_samples_raw_file:
                        logger.warning(
                            f"The number of {property_name} ({len(data)}) "
                            f"does not match the number of "
                            f"raw samples ({num_samples_raw_file}). "
                            f"Please check if overwrite is needed."
                        )
                logger.info("Property cache is found. Will load properties from cache.")
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    f"Failed to load {property_name}.pkl from cache. "
                    "Will rebuild properties."
                )
                overwrite = True
        else:
            logger.info("Property cache is not found. Will build properties.")
            overwrite = True

        # check if all raw structures have been built to crystal structure
        structure_cache_path = osp.join(self.cache_path, "structures")
        if osp.exists(structure_cache_path) and not overwrite:
            logger.info(
                "The cache file of built crystal structure is found. "
                "Will load structures from cache."
            )
            files_structure = [
                f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")
            ]
            num_cached_structures = len(files_structure)
            if num_samples_raw_file == num_cached_structures:
                logger.info(
                    f"All raw files have been built to crystal structures, "
                    f"and the number of structures are {num_cached_structures}."
                )
            else:
                logger.warning(
                    f"The number of cached structures "
                    f"({num_cached_structures}) does not match "
                    f"the number of raw samples ({num_samples_raw_file}). "
                    f"Please check if overwrite is needed."
                )
        else:
            logger.info("Structure cache is not found. Will build structures.")
            os.makedirs(structure_cache_path, exist_ok=True)
            os.makedirs(property_cache_path, exist_ok=True)

            # Load raw Jarvis dataset (Updated with url support)
            self.raw_data, self.num_samples = self.read_data(
                path=self.path, data_name=jarvis_data_name, url=self.url
            )
            logger.info(f"Load {self.num_samples} samples from {path}")

            # Extract property values from raw dataset
            self.property_data = self.read_property_data(
                data=self.raw_data, property_names=self.property_names
            )

            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                structures = BuildStructure(**build_structure_cfg)(
                    self.raw_data["atoms"]
                )
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info(
                    f"Save {self.num_samples} structures to {structure_cache_path}"
                )
                for property_name in self.property_names:
                    data = self.property_data[property_name]
                    self.save_to_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"),
                        data,
                    )
                    logger.info(
                        f"Save {self.num_samples} {property_name} to {property_cache_path}"  # noqa
                    )
            if dist.is_initialized():
                dist.barrier()

        # check if generate graph infomation
        graph_cache_path = osp.join(self.cache_path, "graphs")
        need_build_graphs = False

        # Determine if graphs need building (Logic merged from both versions)
        if build_graph_cfg is not None:
            if osp.exists(graph_cache_path) and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if not is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.warning(
                            "build_graph_cfg is different. Will rebuild graphs."
                        )
                        need_build_graphs = True
                    else:
                        logger.info("Graph config matches cache. Reusing graphs.")
                except Exception as e:
                    logger.warning(e)
                    logger.warning("Failed to load build_graph_cfg.pkl. Will rebuild.")
                    need_build_graphs = True

                # Check counts
                if not need_build_graphs:
                    files_graph = [
                        f for f in os.listdir(graph_cache_path) if f.endswith(".pkl")
                    ]
                    files_structure = [
                        f
                        for f in os.listdir(structure_cache_path)
                        if f.endswith(".pkl")
                    ]
                    if len(files_graph) != len(files_structure):
                        logger.warning("Graph/Structure count mismatch. Will rebuild.")
                        need_build_graphs = True
            else:
                logger.info(
                    "Graph cache not found or overwrite=True. Will build graphs."
                )
                need_build_graphs = True

        if build_graph_cfg is not None and need_build_graphs:
            os.makedirs(graph_cache_path, exist_ok=True)
            if dist.get_rank() == 0:
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                converter = build_graph_converter(build_graph_cfg)

                # Load structures in order to ensure alignment
                struct_files = sorted(
                    [f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")],
                    key=lambda x: int(x.replace(".pkl", "")),
                )
                # If structures variable exists (from init flow), use it, otherwise load
                if "structures" not in locals():
                    structures = [
                        self.load_from_cache(osp.join(structure_cache_path, f))
                        for f in struct_files
                    ]

                graphs = converter(structures)
                for i in range(len(graphs)):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                    )
                logger.info(f"Save {len(graphs)} graphs to {graph_cache_path}")

            if dist.is_initialized():
                dist.barrier()

            # Clean up
            if "graphs" in locals():
                del graphs
            if "structures" in locals():
                del structures

        # Obtain final properties, structures and graphs
        self.property_data = {
            property_name: self.load_from_cache(
                osp.join(property_cache_path, f"{property_name}.pkl")
            )
            for property_name in self.property_names
        }

        self.structures = [
            osp.join(structure_cache_path, f)
            for f in sorted(
                os.listdir(structure_cache_path),
                key=lambda x: int(x.replace(".pkl", "")),
            )
        ]

        if build_graph_cfg is not None:
            files = sorted(
                os.listdir(graph_cache_path) if osp.exists(graph_cache_path) else [],
                key=lambda x: int(x.replace(".pkl", "")),
            )
            self.graphs = [osp.join(graph_cache_path, f) for f in files]
        else:
            self.graphs = None

        if filter_unvalid:
            self.filter_unvalid_by_property()

        if self.graphs is not None:
            self.filter_unvalid_by_graph()

    def read_data(
        self,
        path: str,
        data_name: str,
        url: str = None,  # Added url argument
    ):
        """
        Load jarvis data. Support both standard registry and direct URL.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 1. Download Logic
        if not osp.exists(path) or not zipfile.is_zipfile(path):
            if osp.exists(path):
                logger.message(
                    f"Invalid dataset zip at '{path}'. Delete and re-download."
                )
                os.remove(path)
            else:
                logger.message("Dataset zip not found. Downloading.")

            # Priority 1: Direct URL provided (Adapter logic)
            if url is not None:
                tmp_path = path + ".downloading"
                try:
                    logger.message(f"Downloading from provided URL: {url}")
                    urllib.request.urlretrieve(url, tmp_path)
                    if not zipfile.is_zipfile(tmp_path):
                        raise ValueError("Downloaded file is not a valid zip archive.")
                    os.replace(tmp_path, path)
                    logger.message("Download succeeded.")
                except Exception as e:
                    if osp.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                    raise RuntimeError(f"Failed to download from URL. Error: {e}")

            # Priority 2: Mirror / Jarvis-Tools (Original logic)
            else:
                # Preferred mirror download
                _registry_map = {d["name"]: d for d in JARVIS_MIRROR_DATASETS}

                download_success = False
                if data_name in _registry_map:
                    tmp_path = path + ".downloading"
                    try:
                        logger.message(f"Trying mirror download for '{data_name}'.")
                        urllib.request.urlretrieve(
                            _registry_map[data_name]["url"], tmp_path
                        )
                        if zipfile.is_zipfile(tmp_path):
                            os.replace(tmp_path, path)
                            download_success = True
                            logger.message("Mirror download succeeded.")
                    except Exception as e:
                        logger.warning(f"Mirror download failed: {e}")
                        if osp.exists(tmp_path):
                            os.remove(tmp_path)

                # Fallback to jarvis-tools
                if not download_success:
                    if not osp.exists(path) or not zipfile.is_zipfile(path):
                        try:
                            logger.message(
                                f"Falling back to jarvis.db.figshare for "
                                f"'{data_name}'"
                            )
                            try:
                                raw_data = jdata(
                                    dataset=data_name,
                                    store_dir=os.path.dirname(path),
                                )
                            except TypeError:
                                raw_data = jdata(dataset=data_name)
                            assert (
                                raw_data is not None
                            ), f"Failed to download dataset {data_name}"
                            # If jdata returns the object directly, we handle it below
                            if raw_data:
                                property_data = defaultdict(list)
                                num_samples = len(raw_data)
                                for item in raw_data:
                                    for key, value in item.items():
                                        property_data[key].append(value)
                                return dict(property_data), num_samples

                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to download dataset {data_name}. Error: {e}"
                            )

        # 2. Reading Logic (from local zip)
        if osp.exists(path) and zipfile.is_zipfile(path):
            logger.message(f"Existing dataset zip found at '{path}'.")
            with zipfile.ZipFile(path) as zf:
                # Generic approach to find json inside
                expected_member = os.path.splitext(os.path.basename(path))[0]
                try:
                    bytes_data = zf.read(expected_member)
                except KeyError:
                    json_members = [n for n in zf.namelist() if n.endswith(".json")]
                    if not json_members:
                        raise RuntimeError(
                            "No .json file found inside the zip archive."
                        )
                    bytes_data = zf.read(json_members[0])
            raw_data = json.loads(bytes_data)
        else:
            # Should have been handled by download logic, but as failsafe
            raise RuntimeError(f"File not found or invalid at {path}")

        property_data = defaultdict(list)
        num_samples = len(raw_data)
        for item in raw_data:
            for key, value in item.items():
                property_data[key].append(value)

        for key, value in dict(property_data).items():
            if len(value) != num_samples:
                # Check for mismatch length
                raise ValueError(
                    f"Property {key} has different length than other properties."
                )

        return dict(property_data), num_samples

    def read_property_data(self, data: Dict, property_names: List[str]):
        property_data = {}
        for property_name in property_names:
            if property_name not in data:
                raise ValueError(f"{property_name} not found in the data")
            property_data[property_name] = data[property_name]
        return property_data

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def filter_unvalid_by_property(self):
        """
        Filter out samples that have invalid properties (Updated with stricter checks).
        """
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            old_num_structs = len(self.structures)

            for i, data_item in enumerate(data):
                # Convert 'na' strings to NaN for proper filtering
                if isinstance(data_item, str):
                    if data_item.lower() in ['na', 'nan', 'none', '']:
                        data_item = np.nan
                    else:
                        # Skip non-numeric strings (they're invalid for numeric properties)
                        continue
                # Keep only valid numeric values (not None, not NaN)
                if data_item is not None and not math.isnan(data_item):
                    reserve_idx.append(i)

            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.structures = [self.structures[i] for i in reserve_idx]

            # Graphs reindex: compare with original structure count
            if self.graphs is not None:
                if len(self.graphs) == old_num_structs:
                    self.graphs = [self.graphs[i] for i in reserve_idx]
                else:
                    logger.warning(
                        "Graphs count mismatches structures during property "
                        "filtering. Rebuilding graphs."
                    )
                    self.graphs = self._build_graphs_for_structures(self.structures)

            kept = len(reserve_idx)
            total = len(data)
            logger.warning(
                f"After property filtering '{property_name}': "
                f"kept {kept}/{total} samples."
            )

        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def filter_unvalid_by_graph(self):
        """
        Filter out samples that have invalid graphs.
        """
        # If graphs and structures are misaligned, rebuild graphs for current structures
        if len(self.graphs) != len(self.structures):
            logger.warning(
                "Rebuilding graphs to match structures before graph filtering."
            )
            self.graphs = self._build_graphs_for_structures(self.structures)

        reserve_idx = []
        for i, g in enumerate(self.graphs):
            data = self.load_from_cache(g) if isinstance(g, str) else g
            if data is not None:
                reserve_idx.append(i)

        for key in self.property_data.keys():
            self.property_data[key] = [self.property_data[key][i] for i in reserve_idx]
        self.structures = [self.structures[i] for i in reserve_idx]
        self.graphs = [self.graphs[i] for i in reserve_idx]
        logger.warning(
            f"Filter out {len(self.graphs) - len(reserve_idx)} "
            f"samples with invalid graphs."
        )

        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def _build_graphs_for_structures(self, structures_list):
        """Helper to rebuild graphs in-memory if needed (Ported from new version)."""
        if self.build_graph_cfg is None:
            logger.warning("build_graph_cfg is None, cannot build graphs.")
            return []
        converter = build_graph_converter(self.build_graph_cfg)
        structures = []
        for s in structures_list:
            if isinstance(s, str):
                structures.append(self.load_from_cache(s))
            else:
                structures.append(s)
        graphs = converter(structures)
        return graphs

    def get_structure_array(self, structure):
        atom_types = np.array([site.specie.Z for site in structure])
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        structure_array = {
            "frac_coords": ConcatData(structure.frac_coords.astype("float32")),
            "cart_coords": ConcatData(structure.cart_coords.astype("float32")),
            "atom_types": ConcatData(atom_types),
            "lattice": ConcatData(lattice.reshape(1, 3, 3)),
            "lengths": ConcatData(lengths),
            "angles": ConcatData(angles),
            "num_atoms": ConcatData(np.array([tuple(atom_types.shape)[0]])),
        }
        return structure_array

    def __getitem__(self, idx: int):
        data = {}
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)

        for property_name in self.property_names:
            if property_name in self.property_data:
                value = self.property_data[property_name][idx]
                # Check for 'na' strings - these should have been filtered out during initialization
                if isinstance(value, str) and value.lower() in ['na', 'nan', 'none', '']:
                    raise ValueError(
                        f"Found invalid property value '{value}' at index {idx} for property "
                        f"'{property_name}'. This should have been filtered out during dataset "
                        f"initialization. Please ensure 'filter_unvalid=True' is set and "
                        f"consider clearing the cache to regenerate filtered data."
                    )
                # Check for NaN values - these should also have been filtered out
                if value is not None and (isinstance(value, float) and math.isnan(value)):
                    raise ValueError(
                        f"Found NaN value at index {idx} for property '{property_name}'. "
                        f"This should have been filtered out during dataset initialization."
                    )
                data[property_name] = np.array([value]).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = (
            self.property_data["id"][idx] if "id" in self.property_data else idx
        )
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
