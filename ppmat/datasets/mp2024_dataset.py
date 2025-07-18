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

import os
import os.path as osp
import pickle
import math
import re
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import read_json_lines, count_samples_json_lines
from ppmat.utils.misc import is_equal

class MP2024Dataset(Dataset):
    """MP2024 Dataset Handler

    **Dataset Overview**

    - **Preprocessed Version**:
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │
    ├───────────────────┼─────────┼─────────┼─────────┤
    │ Sample Count      │ 130000  │  10000  │  15361  │
    └───────────────────┴─────────┴─────────┴─────────┘
    ```
    Download preprocessed data: https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2024/mp2024_train_130k.zip

    **Data Format**
    Each sample in the dataset is represented as a `dict` with the following keys and value types:

    General Identifiers:
    --------------------
    - '_id': `dict` — Unique identifier for the sample.
    - 'material_id': `str` — Unique identifier for Materials Project.
    - 'database_IDs': `dict` — Other database identifiers.
    - 'task_ids': `list` — Related task identifiers.
    - 'last_updated': `dict` — Last update timestamp.   
    - `deprecated`: `bool` — Whether the material entry is deprecated.
    - `deprecation_reasons`: `str` — Reasons for deprecation, if any.
    - 'builder_meta': `dict` — Metadata about the dataset construction tools and environment.
        - 'emmet_version': `str` — Version of the Emmet library used to process the data.
        - 'pymatgen_version': `str` — Version of the Pymatgen library used to process the data.
        - 'pull_request'
        - 'database_version'
        - 'build_date'
        - 'license'
    - 'origins': `list` — Source information for structure, energy, magnetism, etc.
        - 'name': `str` — The property type or task name (e.g., 'structure', 'energy', 'magnetism').
        - 'task_id': `str` — The computation task ID associated with this property.
        - 'last_updated': `dict` — Dictionary containing the timestamp of the last update for this property source.
    - 'warnings': `list` — Any known issues or notes about the entry.

    Chemical Information:   
    ---------------------
    - `formula_pretty`: `str` — Human-readable chemical formula.
    - `formula_anonymous`: `str` — Element-anonymized formula (e.g., "ABC3").
    - `chemsys`: `str` — Chemical system (alphabetically sorted element symbols).

    Symmetry Data:
    --------------
    - 'symmetry': `dict` — Symmetry information of the crystal structure.
        - 'crystal_system': `str` — The crystal system.
        - 'symbol': `str` — Space group symbol of the structure.
        - 'number': `int` — International space group number.
        - 'point_group': `str` — Point group classification.
        - 'symprec': `float` — Tolerance used in symmetry detection.
        - 'version': `str` — Version of the software used for symmetry analysis.

    Composition and Structure:
    --------------------------
    - `nsites`: `int` — Number of atomic sites.
    - `elements`: `list` — Element symbols present.
    - `nelements`: `int` — Number of distinct elements.
    - `composition`: `dict` — Elemental amounts.
    - `composition_reduced`: `dict` — Reduced stoichiometry.
    - `structure`: `dict` — Full crystal structure. **Details as follows:**
        - `@module`: `str` — Python module where the structure class is defined, e.g., `'pymatgen.core.structure'`.
        - `@class`: `str` — Class name of the structure, e.g., `'Structure'`.
        - `charge`: `int` — Net charge of the entire structure.
        - `lattice`: `dict` — Lattice information:
            - `matrix`: `list` — 3x3 lattice vectors defining the unit cell.
            - `pbc`: `list[bool]` — Periodic boundary conditions flags for each spatial direction.
            - `a`, `b`, `c`: `float` — Lattice constants lengths.
            - `alpha`, `beta`, `gamma`: `float` — Lattice angles in degrees.
            - `volume`: `float` — Volume of the unit cell.
        - `properties`: `dict` — Optional additional properties related to the structure.
        - `sites`: `list[dict]` — List of atomic sites, each site including:
            - `species`: `list[dict]` — List of species at the site, with:
                - `element`: `str` — Chemical element symbol.
                - `occu`: `float` — Occupancy of the element at the site.
            - `abc`: `list[float]` — Fractional coordinates within the unit cell.
            - `xyz`: `list[float]` — Cartesian coordinates in Ångstroms.
            - `properties`: `dict` — Site-specific properties, e.g., magnetic moment (`magmom`).
            - `label`: `str` — Element label for the site.

    Thermodynamic Properties:
    -------------------------
    - `volume`: `float` — Cell volume.
    - `density`: `float` — Density.
    - `density_atomic`: `float` — Atomic density.
    - `uncorrected_energy_per_atom`: `float` — Uncorrected energy per atom.
    - `energy_per_atom`: `float` — Final total energy per atom.
    - `formation_energy_per_atom`: `float` — Formation energy per atom.
    - `energy_above_hull`: `float` — Energy above convex hull (thermodynamic stability).
    - `equilibrium_reaction_energy_per_atom`: `float` — Equilibrium reaction energy per atom.
    - `is_stable`: `bool` — Whether the material is thermodynamically stable.
    - `decomposes_to`: `list` — Decomposition products.


    Electronic Properties:
    ----------------------
    - `band_gap`: `float` — Band gap.
    - `cbm`: `float` — Conduction band minimum.
    - `vbm`: `float` — Valence band maximum.
    - `efermi`: `float` — Fermi energy.
    - `is_gap_direct`: `bool` — Whether the band gap is direct.
    - `is_metal`: `bool` — Whether the material is metallic.
    - `es_source_calc_id`
    - `bandstructure`
    - `dos`:  Density of states. **Details as follows:**
        - `total`: `dict` — Total density of states aggregated over all elements and orbitals. Contains spin channels keyed by `'1'` (spin up) and `'-1'` (spin down), each holding:
            - `task_id`: `str` — Identifier of the calculation task.
            - `band_gap`: `float` — Band gap energy in eV.
            - `cbm`: `float` — Conduction band minimum energy in eV.
            - `vbm`: `float` — Valence band maximum energy in eV.
            - `efermi`: `float` — Fermi energy in eV.
            - `spin_polarization`: `float` or `None` — Degree of spin polarization.
        - `elemental`: `dict` — Density of states resolved by element symbol. Each element maps to:
            - `total`: `dict` — Total DOS for that element (with same spin structure as above).
            - Orbital-resolved DOS: keys like `'s'`, `'p'`, `'d'`, each mapping spin-resolved DOS dicts.
        - `orbital`: `dict` — Density of states resolved by orbital type (`'s'`, `'p'`, `'d'`), with spin-resolved data similar to above.
        - `magnetic_ordering`: `str` — Magnetic ordering of the system, e.g., `'FM'` (ferromagnetic).
    - `dos_energy_up`: `float` — Spin-up DOS energy
    - `dos_energy_down`: `float` — Spin-down DOS energy.


    Magnetic Properties:
    --------------------
    - `is_magnetic`: `bool` — Whether the material is magnetic.
    - `ordering`: `str` — Magnetic ordering type (e.g., 'FM', 'AFM').
    - `total_magnetization`: `float` — Total magnetization.
    - `total_magnetization_normalized_vol`: `float` — Magnetization per unit volume.
    - `total_magnetization_normalized_formula_units`: `float` — Magnetization per formula unit.
    - `num_magnetic_sites`: `int` — Number of magnetic sites.
    - `num_unique_magnetic_sites`: `int` — Number of unique magnetic sites.
    - `types_of_magnetic_species`: `list` — Magnetic element types.


    Mechanical Properties:
    ----------------------
    - `bulk_modulus`: float` — Bulk modulus.
    - `shear_modulus`: `float` — Shear modulus.
    - `universal_anisotropy`: `float` — Universal anisotropy index.
    - `homogeneous_poisson`: `float` — Homogeneous Poisson's ratio.


    Surface Properties:
    -------------------
    - `weighted_surface_energy_EV_PER_ANG2`: `float` — Weighted surface energy.
    - `weighted_surface_energy`: `float` — Surface energy.
    - `weighted_work_function`
    - `surface_anisotropy`: `float` — Surface anisotropy.
    - `shape_factor`: `float` — Shape factor describing surface morphology.
    - `has_reconstructed`: `bool` — Indicates if surface reconstruction has occurred.

    XAS:
    ----
    - 'xas'

    ----
    - 'grain_boundaries' 

    Electronic Energy:
    ------------------
    - 'e_total' 
    - 'e_ionic' 
    - 'e_electronic' 
    - 'n'
    - 'e_ij_max' 

    Other Fields:
    -------------
    - `possible_species`: `list` — Estimated possible oxidation states.
    - `has_props`: `dict` — Flags indicating presence of specific property data, e.g.,
        'materials', 'thermo', 'xas', 'grain_boundaries', 'chemenv', 'electronic_structure', 'absorption', 
        'bandstructure', 'dos', 'magnetism', 'elasticity', 'dielectric', 'piezoelectric', 'surface_properties', 
        'oxi_states', 'provenance', 'charge_density', 'eos', 'phonon', 'insertion_electrodes', 'substrates'.
    - `theoretical`: `bool` — Whether the structure is theoretical.
    - `property_name`: `str`


    **Notes**
    - Missing values are represented as `None` 
    - CIF parsing requires additional dependencies (e.g., pymatgen)
    - For custom data, ensure index consistency across all fields


    Args:
        path (str): The path of the dataset, if path is not exists, it will
            be downloaded.
        property_names (Optional[list[str]], optional): Property names you want to use,
            for mp2018.6.1, the property_names should be selected from
            ["formation_energy_per_atom", "band_gap", "G", "K"]. Defaults to None.
        build_structure_cfg (Dict, optional): The configs for building the pymatgen
            structure from cif string, if not specified, the default setting will be
            used. Defaults to None.
        build_graph_cfg (Dict, optional): The configs for building the graph from
            structure. Defaults to None.
        transforms (Optional[Callable], optional): The preprocess transforms for each
            sample. Defaults to None.
        cache_path (Optional[str], optional): If a cache_path is set, structures and
            graph will be read directly from this path; if the cache does not exist,
            the converted structures and graph will be saved to this path. Defaults
            to None.
        overwrite (bool, optional): Overwrite the existing cache file at the given cache
            path if it already exists. Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out unvalid samples. Defaults
            to True.
    """

    name = "mp2024_train_130k"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2024/mp2024_train_130k.zip"
    md5 = "6aa4d9f52e3f39270719e163465fcea8"

    def __init__(
        self,
        path: str,
        property_names: Optional[list[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))
        
        self.path = path
        if isinstance(property_names, str):
            property_names = [property_names]        

        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "dict",
                "primitive": False,
                "niggli": True,
                "num_cpus": 1,
            }
            logger.message(
                "The build_structure_cfg is not set, will use the default "
                f"configs: {build_structure_cfg}"
            )
        
        self.property_names = property_names if property_names is not None else []
        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # for example:
            # path = ./data/mp2024_train_130k/mp2024_train.txt
            # cache_path = ./data/mp2024_train_130k_cache_find_points_in_spheres_cutoff_4/mp2024_train
            graph_converter_name = re.sub(r'(?<!^)([A-Z])', r'_\1', build_graph_cfg['__class_name__']).lower()
            cutoff_name = str( int(build_graph_cfg['__init_params__']['cutoff']) ) 
            self.cache_path = osp.join( osp.split(path)[0] + "_cache_" + graph_converter_name + "_cutoff_" + cutoff_name,
                 osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # compute number of samples of raw file
        if osp.exists(self.path):
            num_samples_raw_file = count_samples_json_lines(self.path)
            logger.info(f"The raw file has {num_samples_raw_file} samples.")
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
                    logger.info(f"Load {len(data)} {property_name} values from {property_cache_path}")
                    if len(data) != num_samples_raw_file:
                        logger.warning(
                            f"The number of {property_name} ({len(data)}) " 
                            f"does not match the number of raw samples ({num_samples_raw_file}). "
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
            logger.info("The cache file of built crystal structure is found. Will load structures from cache.")
            
            # compute number of cached structures
            files_structure = [f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")]
            num_cached_structures = len(files_structure)
            if num_samples_raw_file == num_cached_structures:
                logger.info(f"All raw files have been built to crystal structures, and the number of structures are {num_cached_structures}.")
            else:
                logger.warning( 
                    f"The number of cached structures ({num_cached_structures}) does not match "
                    f"the number of raw samples ({num_samples_raw_file}). "
                    f"Please check if overwrite is needed." 
                )
        else:
            logger.info("Structure cache is not found. Will build structures.")
            os.makedirs(structure_cache_path, exist_ok=True)
            os.makedirs(property_cache_path, exist_ok=True)
            self.row_data, self.num_samples = self.read_data(path)
            logger.info(f"Load {self.num_samples} samples from {path}")
            self.property_data = self.read_property_data(self.row_data, self.property_names) 
            
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_structure_cfg to cache file
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                # convert strucutes
                structures = BuildStructure(**build_structure_cfg)(
                    self.row_data["structure"]
                )
                # save structures to cache file
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info( f"Save {self.num_samples} structures to {structure_cache_path}" )
                # save property data to cache file
                for property_name in self.property_names:
                    data = self.property_data[property_name]
                    self.save_to_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"), 
                        data,
                    )
                    logger.info( f"Save {self.num_samples} {property_name} to {property_cache_path}" )
            # sync all processes
            if dist.is_initialized():
                dist.barrier()

        # check if generate graph infomation
        graph_cache_path = osp.join(self.cache_path, "graphs")
        graph_cache_exists = build_graph_cfg is not None and osp.exists(graph_cache_path)
        # check if create graph configures is same with cache.
        if graph_cache_exists and not overwrite:
            try:
                build_graph_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl")
                )
                if is_equal(build_graph_cfg_cache, build_graph_cfg):
                    logger.info(
                        "The cached build_structure_cfg configuration "
                        "matches the current settings. Reusing previously "
                        "generated structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_graph_cfg is different from build_graph_cfg_cache"
                        ". Will rebuild the graphs."
                    )
                    logger.warning(
                        "If you want to use the cached structures and graphs, "
                        "please ensure that the settings used in match your "
                        "current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_graph_cfg.pkl from cache. "
                    "Will rebuild the graphs."
                )
                overwrite = True

        # check if all structures are built into the graph
        if osp.exists(graph_cache_path) and not overwrite:
            logger.info("The cache file of built crystal structure is found. Will load structures from cache.")
            # compute number of cached graph
            files_graph = [f for f in os.listdir(graph_cache_path) if f.endswith(".pkl")]
            num_cached_graphs = len(files_graph)
            if num_cached_structures == num_cached_graphs:
                logger.info(f"All cached structures have been built graphs, and the number of graphes are {num_cached_graphs}.")
            else:
                logger.warning( 
                    f"The number of cached structures ({num_cached_structures}) does not match "
                    f"the number of cached graphs ({num_cached_graphs}). "
                    f"Please check if overwrite is needed." 
                )
        else:
            logger.info("Graph cache is not found. Will build graphs.")
            os.makedirs(graph_cache_path, exist_ok=True)
            # convert graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_graph_cfg to cache file
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), 
                    build_graph_cfg
                )
                # convert graph
                converter = build_graph_converter(build_graph_cfg)
                graphs = converter(structures)
                # save graphs to cache file
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                    )
                logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")
            # sync all processes
            if dist.is_initialized():
                dist.barrier()

        # Obtain finial properies, structures and graphs
        self.property_data = {property_name: self.load_from_cache(osp.join(property_cache_path, f"{property_name}.pkl")) for property_name in self.property_names}
        
        self.structures = [
            osp.join(structure_cache_path, f) 
            for f in sorted(
                os.listdir(structure_cache_path),
                key=lambda x: int(x.replace(".pkl", ""))
            )
        ]
        
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f) 
                for f in sorted(
                    os.listdir(graph_cache_path),
                    key=lambda x: int(x.replace(".pkl", ""))
                )
            ]
        else:
            self.graphs = None

        # filter by property data, since some samples may have no valid properties
        if filter_unvalid:
            self.filter_unvalid_by_property()
        
        # filter by graph data, as some samples may not have edges and need to be removed
        if self.graphs is not None:
            self.filter_unvalid_by_graph()
        
        # assert (
        #     self.graphs is None or len(self.graphs) == self.num_samples
        # ), "The number of graphs must be equal to the number of samples."

        # # filter by property data, since some samples may have no valid properties
        # if filter_unvalid:
        #     self.filter_unvalid_by_property()

    def filter_unvalid_by_graph(self):
        reserve_idx = []
        for i, g in enumerate(self.graphs):
            data = self.load_from_cache(g)
            if data is not None:
                reserve_idx.append(i)
            
        for key in self.property_data.keys():
            self.property_data[key] = [
                self.property_data[key][i] for i in reserve_idx
            ]
        self.structures = [self.structures[i] for i in reserve_idx]
        self.graphs = [self.graphs[i] for i in reserve_idx]
        logger.warning(
            f"Filter out {len(reserve_idx)} samples with valid graphs."
        )

        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def read_data(self, path: str):
        """Read the data from the given dataset file.

        Args:
            path (str): Path to the data.
        """
        data = read_json_lines(path)
        num_samples = len(data["structure"])

        return data, num_samples

    def read_property_data(self, data: Dict, property_names: list[str]):
        """Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
            property_names (list[str]): Property names.
        """
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
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (data_item is not None and not math.isnan(data_item)):
                    reserve_idx.append(i)
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def get_structure_array(self, structure):
        atom_types = np.array([site.specie.Z for site in structure])
        # get lattice parameters and matrix
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
        """Get item at index idx."""
        data = {}
        # get graph
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
