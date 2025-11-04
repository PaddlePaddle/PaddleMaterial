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
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.misc import is_equal

JARVIS_2D_DEFAULT_URL = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/jarvis/jdft_2d-4-26-2020.zip"


class Jarvis2DDataset(Dataset):
    """Jarvis 2D Dataset Handler for the 2020 release (jdft_2d-4-26-2020).

    This dataset handler mirrors the behavior of `JarvisDataset` but targets the
    specific 2D JSON zip provided by `JARVIS_2D_DEFAULT_URL` (or a user-provided URL).

    Args:
        path (str): Directory to store the dataset zip and caches. The zip file
            will be downloaded into this directory if missing.
        property_names (Union[str, List[str]]): Target property names to extract,
            e.g., ["formation_energy_peratom"], ["optb88vdw_bandgap"].
        url (str, optional): Dataset download URL. Defaults to the Paddle BOS
            mirror `JARVIS_2D_DEFAULT_URL`.
        build_structure_cfg (Dict, optional): Structure build config for
            `BuildStructure` with format "jarvis". Defaults to a sensible
            config when not provided.
        build_graph_cfg (Dict, optional): Graph conversion configuration. When
            provided, graphs will be generated and cached.
        transforms (Optional[Callable], optional): Sample transforms.
        cache_path (Optional[str], optional): Root cache directory. When None,
            defaults under `path`.
        overwrite (bool, optional): Overwrite existing caches. Defaults to False.
        filter_unvalid (bool, optional): Filter invalid samples. Defaults to True.
    """

    def __init__(
        self,
        path: str,
        property_names: Union[str, List[str]],
        url: Optional[str] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ):
        super().__init__()

        dataset_url = url or JARVIS_2D_DEFAULT_URL
        zip_basename = osp.basename(dataset_url)
        filename_no_zip, _ = osp.splitext(zip_basename)
        self.path = osp.join(path, zip_basename)

        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names if property_names is not None else []

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

        if build_graph_cfg is not None:
            graph_converter_name = re.sub(
                r"(?<!^)([A-Z])", r"_\1", build_graph_cfg["__class_name__"]
            ).lower()
            cutoff_name = str(int(build_graph_cfg["__init_params__"]["cutoff"]))
        else:
            graph_converter_name = "none"
            cutoff_name = "none"

        if cache_path is not None:
            self.cache_path = osp.join(
                cache_path,
                "dft_2d" + "_cache_" + graph_converter_name + "_cutoff_" + cutoff_name,
            )
        else:
            self.cache_path = osp.join(
                path,
                "dft_2d" + "_cache_" + graph_converter_name + "_cutoff_" + cutoff_name,
            )
        logger.info(f"Cache path: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # compute number of samples of raw file
        if osp.exists(self.path) and zipfile.is_zipfile(self.path):
            try:
                with zipfile.ZipFile(self.path) as zf:
                    try:
                        bytes_data = zf.read(filename_no_zip)
                    except KeyError:
                        json_members = [n for n in zf.namelist() if n.endswith(".json")]
                        if not json_members:
                            raise RuntimeError(
                                "No .json file found inside the downloaded zip archive."
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
            self.raw_data, self.num_samples = self.read_data(
                path=self.path,
                filename_no_zip=filename_no_zip,
                url=dataset_url,
            )
            logger.info(f"Load {self.num_samples} samples from {path}")
            self.property_data = self.read_property_data(
                data=self.raw_data, property_names=self.property_names
            )

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
                        f"Save{self.num_samples} {property_name}to{property_cache_path}"
                    )
            if dist.is_initialized():
                dist.barrier()

        graph_cache_path = osp.join(self.cache_path, "graphs")
        need_build_graphs = False
        if build_graph_cfg is not None:
            if osp.exists(graph_cache_path):
                # check config equivalence
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if not is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.warning(
                            "build_graph_cfg is different. Will rebuild graphs."
                        )
                        need_build_graphs = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load builded_graph_cfg.pkl. Will rebuild the graphs."
                    )
                    need_build_graphs = True

                # check count mismatch
                files_graph = [
                    f for f in os.listdir(graph_cache_path) if f.endswith(".pkl")
                ]
                num_cached_graphs = len(files_graph)
                files_structure = [
                    f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")
                ]
                num_cached_structures = len(files_structure)
                if num_cached_graphs != num_cached_structures:
                    need_build_graphs = True
            else:
                logger.info("Graph cache is not found. Will build graphs.")
                need_build_graphs = True

        if build_graph_cfg is not None and need_build_graphs:
            os.makedirs(graph_cache_path, exist_ok=True)
            if dist.get_rank() == 0:
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                converter = build_graph_converter(build_graph_cfg)
                # load structures in order
                struct_files = sorted(
                    [f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")],
                    key=lambda x: int(x.replace(".pkl", "")),
                )
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
            if "graphs" in locals():
                del graphs
            if "structures" in locals():
                del structures

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

    def read_data(self, path: str, filename_no_zip: str, url: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not osp.exists(path) or not zipfile.is_zipfile(path):
            if osp.exists(path):
                logger.message(
                    f"Invalid Jarvis 2D dataset zip at '{path}'.Delete and re-download."
                )
                os.remove(path)
            else:
                logger.message(
                    "Jarvis 2D dataset zip not found. Downloading from provided URL."
                )
            tmp_path = path + ".downloading"
            try:
                urllib.request.urlretrieve(url, tmp_path)
                if not zipfile.is_zipfile(tmp_path):
                    raise ValueError("Downloaded file is not a valid zip archive.")
                os.replace(tmp_path, path)
                logger.message("Download succeeded. Using the downloaded archive.")
            except Exception as e:
                if osp.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Failed to download Jarvis 2D dataset from URL. Error: {e}"
                )
        else:
            logger.message(f"Existing Jarvis 2D dataset zip archive found at '{path}'.")

        with zipfile.ZipFile(path) as zf:
            try:
                bytes_data = zf.read(filename_no_zip)
            except KeyError:
                json_members = [n for n in zf.namelist() if n.endswith(".json")]
                if not json_members:
                    raise RuntimeError("No .json file found inside the zip archive.")
                bytes_data = zf.read(json_members[0])
        raw_data = json.loads(bytes_data)

        property_data = defaultdict(list)
        num_samples = len(raw_data)
        for item in raw_data:
            for key, value in item.items():
                property_data[key].append(value)
        for key, value in dict(property_data).items():
            if len(value) != num_samples:
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
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            old_num_structs = len(self.structures)
            for i, data_item in enumerate(data):
                is_valid = False
                if isinstance(data_item, (int, float, np.floating)):
                    try:
                        is_valid = (data_item is not None) and (
                            not math.isnan(float(data_item))
                        )
                    except Exception:
                        is_valid = False
                # strings like 'na' are treated invalid
                if is_valid:
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
                        "Graphs count mismatches structures during property filtering."
                    )
                    self.graphs = self._build_graphs_for_structures(self.structures)
            kept = len(reserve_idx)
            total = len(data)
            logger.warning(
                f"After property filtering'{property_name}':kept{kept}/{total}samples."
            )
        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def filter_unvalid_by_graph(self):
        # If graphs and structures are misaligned, rebuild graphs for current structures
        if len(self.graphs) != len(self.structures):
            logger.warning("Rebuilding graphs to match structures.")
            self.graphs = self._build_graphs_for_structures(self.structures)
        reserve_idx = []
        for i, g in enumerate(self.graphs):
            data = self.load_from_cache(g) if isinstance(g, str) else g
            if data is not None:
                reserve_idx.append(i)
        # if all valid, keep as-is
        if len(reserve_idx) != len(self.structures):
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]
            self.structures = [self.structures[i] for i in reserve_idx]
            self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(f"Filter out {len(reserve_idx)} samples with valid graphs.")
            self.num_samples = len(self.structures)
            logger.warning(f"Remaining {self.num_samples} samples after filtering.")
        else:
            logger.warning(
                "All graphs are valid after rebuild/alignment. No further filtering."
            )

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
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = (
            self.property_data["id"][idx] if "id" in self.property_data else idx
        )
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples

    def _build_graphs_for_structures(self, structures_list):
        """Build graphs for given structures (paths or Structure objects).

        Returns list of graphs (in-memory)."""
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
