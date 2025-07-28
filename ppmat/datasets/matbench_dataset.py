# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

import math
import os
import os.path as osp
import pickle
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle.distributed as dist
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


class MatbenchDataset(Dataset):
    """MatBench Formation Energy Dataset Handler.

    **MatBench Dataset Overview**

    The MatBench benchmark dataset provides formation energies of inorganic materials
    from the Materials Project database. This dataset is commonly used for benchmarking
    machine learning models in materials science.

    **Dataset Details**
    - **Source**: MatBench benchmark (matbench_mp_e_form)
    - **Total Samples**: Variable (e.g., 500 for subset, 132,752 for full dataset)
    - **Property**: Formation energy per atom (eV/atom)
    - **Structure Format**: pymatgen.core.structure.Structure objects
    - **Target Range**: Typically -4.61 to 2.50 eV/atom

    **Data Format**
    The dataset is stored as a pandas DataFrame in pickle format with columns:
    - 'structure': pymatgen.core.structure.Structure objects containing atomic positions,
      lattice parameters, and element types
    - 'e_form': Formation energy per atom values (float64)

    **Example Usage**
    ```python
    # 使用本地文件（如果存在）
    dataset = MatbenchDatasetNew(
        path="data/matbench/matbench_mp_e_form_full.pkl",
        property_names=["e_form"],
        build_graph_cfg={
            "__class_name__": "FindPointsInSpheres",
            "__init_params__": {"cutoff": 4.0},
        }
    )

    # 自动下载（如果文件不存在）
    dataset = MatbenchDatasetNew(
        path="./data/matbench/matbench_mp_e_form_full.pkl",  # 文件不存在时会自动下载
        property_names=["e_form"]
    )
    ```

    **Automatic Download**
    If the dataset file doesn't exist at the specified path, it will be automatically
    downloaded from: https://paddle-org.bj.bcebos.com/paddlematerial/datasets/matbench/matbench_mp_e_form_full.pkl

    Args:
        path (str): Path to the matbench dataset pickle file. If the file doesn't exist,
            it will be automatically downloaded to this location.

        property_names (Union[str, List[str]]): Property names to extract from the dataset.
            For matbench, this is typically ["e_form"] for formation energy.

        build_graph_cfg (Dict, optional): Configuration for building graphs from crystal
            structures. If None, only structure arrays will be provided. Defaults to None.

        transforms (Optional[Callable], optional): Transform functions to apply to each
            sample. Defaults to None.

        cache_path (Optional[str], optional): Directory path for caching processed structures
            and graphs. If None, cache will be created in the same directory as the data file.
            Defaults to None.

        overwrite (bool, optional): Whether to overwrite existing cache files. Defaults to False.

        filter_unvalid (bool, optional): Whether to filter out samples with invalid properties
            (NaN, None, or non-numeric values). Defaults to True.

    """

    # 数据集下载配置，模仿mp2018dataset
    name = "matbench_mp_e_form_full"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/matbench/matbench_mp_e_form_full.pkl"
    md5 = None  # 如果需要可以添加MD5校验

    def __init__(
        self,
        path: str,
        property_names: Union[str, List[str]],
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        # Handle automatic download if file doesn't exist (模仿mp2018dataset)
        if not osp.exists(path):
            logger.message("The MatBench dataset is not found. Will download it now.")
            # get_datasets_path_from_url 返回的就是完整的文件路径
            path = download.get_datasets_path_from_url(self.url, self.md5)

        self.path = path

        # Handle property names
        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names if property_names is not None else []

        # Handle graph configuration
        self.build_graph_cfg = build_graph_cfg

        # Determine cache directory
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
                "matbench_cache_" + graph_converter_name + "_cutoff_" + cutoff_name,
            )
        else:
            # Create cache in same directory as data file
            data_dir = osp.dirname(self.path)
            self.cache_path = osp.join(
                data_dir,
                "matbench_cache_" + graph_converter_name + "_cutoff_" + cutoff_name,
            )

        logger.info(f"Cache path: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        # Additional parameters
        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # Load raw data
        logger.info(f"Loading MatBench dataset from {self.path}")
        self.raw_data = self.load_raw_data()
        self.num_samples = len(self.raw_data)
        logger.info(f"Loaded {self.num_samples} samples from MatBench dataset")

        # Extract property values
        self.property_data = self.extract_property_data()

        # Setup cache paths
        structure_cache_path = osp.join(self.cache_path, "structures")
        property_cache_path = osp.join(self.cache_path, "properties")
        graph_cache_path = osp.join(self.cache_path, "graphs")

        # Check if properties have been cached
        if osp.exists(property_cache_path) and not overwrite:
            try:
                for property_name in self.property_names:
                    data = self.load_from_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"),
                    )
                    logger.info(f"Load {len(data)} {property_name} values from cache")
                logger.info("Property cache found. Will load properties from cache.")
            except Exception as e:
                logger.warning(f"Failed to load properties from cache: {e}")
                overwrite = True
        else:
            logger.info("Property cache not found. Will build properties.")
            overwrite = True

        # Check if structures have been cached
        if osp.exists(structure_cache_path) and not overwrite:
            files_structure = [
                f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")
            ]
            num_cached_structures = len(files_structure)
            if self.num_samples == num_cached_structures:
                logger.info(
                    f"All structures cached ({num_cached_structures} files found)"
                )
            else:
                logger.warning(
                    f"Structure cache mismatch: {num_cached_structures} cached vs "
                    f"{self.num_samples} samples. Will rebuild."
                )
                overwrite = True
        else:
            logger.info("Structure cache not found. Will build structures.")
            overwrite = True

        # Process and cache structures if needed
        if overwrite and dist.get_rank() == 0:
            os.makedirs(structure_cache_path, exist_ok=True)
            os.makedirs(property_cache_path, exist_ok=True)

            # Since MatBench already contains Structure objects, save them directly
            structures = self.raw_data["structure"].tolist()
            for i in range(self.num_samples):
                self.save_to_cache(
                    osp.join(structure_cache_path, f"{i:010d}.pkl"),
                    structures[i],
                )
            logger.info(f"Saved {self.num_samples} structures to cache")

            # Save property data to cache
            for property_name in self.property_names:
                data = self.property_data[property_name]
                self.save_to_cache(
                    osp.join(property_cache_path, f"{property_name}.pkl"),
                    data,
                )
                logger.info(f"Saved {property_name} data to cache")

        # Sync all processes
        if dist.is_initialized():
            dist.barrier()

        # Handle graph generation
        if build_graph_cfg is not None:
            if osp.exists(graph_cache_path) and not overwrite:
                # Check if graph config matches
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.info("Graph cache found with matching config.")
                    else:
                        logger.warning("Graph config mismatch. Will rebuild graphs.")
                        overwrite = True
                except Exception as e:
                    logger.warning(f"Failed to load graph config: {e}")
                    overwrite = True
            else:
                logger.info("Graph cache not found. Will build graphs.")
                overwrite = True

            # Build graphs if needed
            if overwrite and dist.get_rank() == 0:
                os.makedirs(graph_cache_path, exist_ok=True)

                # Save graph config
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )

                # Convert structures to graphs
                converter = build_graph_converter(build_graph_cfg)
                structures = self.raw_data["structure"].tolist()
                graphs = converter(structures)

                # Save graphs to cache
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                    )
                logger.info(f"Saved {self.num_samples} graphs to cache")

                # Clean up memory
                del graphs
                del structures

            # Sync all processes
            if dist.is_initialized():
                dist.barrier()

        # Load final data from cache
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
                os.listdir(graph_cache_path), key=lambda x: int(x.replace(".pkl", ""))
            )
            self.graphs = [osp.join(graph_cache_path, f) for f in files]
        else:
            self.graphs = None

        # Filter invalid samples
        if filter_unvalid:
            self.filter_unvalid_by_property()

        if self.graphs is not None:
            self.filter_unvalid_by_graph()

    def load_raw_data(self):
        """Load raw MatBench data from pickle file.

        Returns:
            pd.DataFrame: DataFrame with 'structure' and target columns.
        """
        with open(self.path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(data)}")

        required_cols = ["structure"] + self.property_names
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return data

    def extract_property_data(self):
        """Extract property data from raw DataFrame.

        Returns:
            Dict[str, List]: Dictionary mapping property names to value lists.
        """
        property_data = {}
        for property_name in self.property_names:
            if property_name not in self.raw_data.columns:
                raise ValueError(f"Property {property_name} not found in data")
            property_data[property_name] = self.raw_data[property_name].tolist()
        return property_data

    def save_to_cache(self, cache_path: str, data: Any):
        """Save data to cache file.

        Args:
            cache_path (str): Path to cache file.
            data (Any): Data to save.
        """
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        """Load data from cache file.

        Args:
            cache_path (str): Path to cache file.

        Returns:
            Any: Loaded data.
        """
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

    def filter_unvalid_by_property(self):
        """Filter out samples with invalid properties."""
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (
                    data_item is not None and not math.isnan(data_item)
                ):
                    reserve_idx.append(i)

            # Update all data structures
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]

            logger.info(
                f"Filtered to {len(reserve_idx)} samples with valid {property_name}"
            )

        self.num_samples = len(self.structures)
        logger.info(f"Final sample count: {self.num_samples}")

    def filter_unvalid_by_graph(self):
        """Filter out samples with invalid graphs."""
        reserve_idx = []
        for i, g in enumerate(self.graphs):
            try:
                data = self.load_from_cache(g)
                if data is not None:
                    reserve_idx.append(i)
            except Exception:
                continue

        # Update all data structures
        for key in self.property_data.keys():
            self.property_data[key] = [self.property_data[key][i] for i in reserve_idx]
        self.structures = [self.structures[i] for i in reserve_idx]
        self.graphs = [self.graphs[i] for i in reserve_idx]

        logger.info(f"Filtered to {len(reserve_idx)} samples with valid graphs")
        self.num_samples = len(self.structures)

    def get_structure_array(self, structure):
        """Convert pymatgen Structure to array format.

        Args:
            structure: pymatgen Structure object.

        Returns:
            Dict: Dictionary containing structure arrays.
        """
        # Get atom types
        atom_types = np.array([site.specie.Z for site in structure])

        # Get lattice parameters and matrix
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
            "num_atoms": ConcatData(np.array([len(atom_types)])),
        }
        return structure_array

    def __getitem__(self, idx: int):
        """Get item at index idx.

        Args:
            idx (int): Sample index.

        Returns:
            Dict: Sample data containing graph/structure and properties.
        """
        data = {}

        # Get graph or structure
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

        # Add property data
        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        # Add sample ID
        data["id"] = idx

        # Apply transforms if provided
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __len__(self):
        """Return dataset length."""
        return self.num_samples
