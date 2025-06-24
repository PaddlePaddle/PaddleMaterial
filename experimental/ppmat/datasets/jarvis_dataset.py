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
import re
import math
import json
import pickle
import zipfile
import os.path as osp
import numpy as np
from collections import defaultdict
from typing import Optional, Union, Dict, List, Any, Tuple, Callable

import paddle.distributed as dist
from paddle.io import Dataset

from jarvis.db.figshare import data as jdata
from jarvis.db.figshare import get_db_info

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.misc import is_equal

class JarvisDataset(Dataset):
    """Jarvis Dataset Handler.
    
    **Jarvis Dataset Overview**

    Download preprocessed data: https://jarvis-materials-design.github.io/dbdocs/thedownloads/
    Github: https://github.com/usnistgov/jarvis/tree/master 

    ```
    ------------------------------------------------------------------------------------------
    | Database name      |  Number of data-points	|  Description      
    ------------------------------------------------------------------------------------------
    | AGRA_CHO	         |  214	                    |  AGRA CHO catalyst dataset
    | AGRA_COOH	         |  280	                    |  AGRA COOH catalyst dataset
    | AGRA_CO	         |  193	                    |  AGRA CO catalyst dataset
    | AGRA_OH	         |  875	                    |  AGRA OH catalyst dataset
    | AGRA_O	         |  1000	                |  AGRA Oxygen catalyst dataset
    | aflow2	         |  400k	                |  AFLOW dataset
    | alex_pbe_1d_all	 |  100k	                |  Alexandria DB all 1D materials with PBE
    | alex_pbe_2d_all	 |  200k	                |  Alexandria DB all 2D materials with PBE
    | alex_pbe_3d_all	 |  5 million	            |  Alexandria DB all 3D materials with PBE
    | alex_pbe_hull	     |  116k	                |  Alexandria DB convex hull stable materials with PBE functional
    | alex_pbesol_3d_all |	500k	                |  Alexandria DB all 3D materials with PBEsol
    | alex_scan_3d_all	 |  500k	                |  Alexandria DB all 3D materials with SCAN
    | alignn_ff_db	     |  307113	                |  Energy per atom, forces and stresses for ALIGNN-FF trainig for 75k materials.
    | arXiv	             |  1796911	                |  arXiv dataset 1.8 million title, abstract and id dataset
    | arxiv_summary	     |  137927	                |  arXiv summary dataset
    | c2db	             |  3514	                |  Various properties in C2DB database
    | cccbdb	         |  1333	                |  CCCBDB dataset
    | cfid_3d	         |  55723	                |  Various 3D materials properties in JARVIS-DFT database computed with OptB88vdW and TBmBJ methods with CFID
    | cod	             |  431778	                |  Atomic structures from crystallographic open database
    | dft_2d_2021	     |  1079	                |  Various 2D materials properties in JARVIS-DFT database computed with OptB88vdW
    | dft_2d	         |  1109	                |  Various 2D materials properties in JARVIS-DFT database computed with OptB88vdW
    | dft_3d_2021	     |  55723	                |  Various 3D materials properties in JARVIS-DFT database computed with OptB88vdW and TBmBJ methods
    | dft_3d	         |  75993	                |  Various 3D materials properties in JARVIS-DFT database computed with OptB88vdW and TBmBJ methods
    | edos_pdos	         |  48469	                |  Normalized electron and phonon density of states with interpolated values and fixed number of bins
    | halide_peroskites	 |  229	                    |  Halide perovskite dataset
    | hmof	             |  137651	                |  Hypothetical MOF database
    | hopv	             |  4855	                |  Various properties of molecules in HOPV15 dataset
    | interfacedb	     |  593	                    |  Interface property dataset
    | jff	             |  2538	                |  Various 3D materials properties in JARVIS-FF database computed with several force-fields
    | m3gnet_mpf_1.5mil	 |  1.5 million	            |  1.5 million structures and their energy, forces and stresses in MP
    | m3gnet_mpf	     |  168k	                |  168k structures and their energy, forces and stresses in MP
    | megnet2	         |  133k	                |  133k materials and their formation energy in MP
    | megnet	         |  69239	                |  Formation energy and bandgaps of 3D materials properties in Materials project database as on 2018, used in megnet
    | mlearn	         |  1730	                |  Machine learning force-field for elements datasets
    | mp_3d_2020	     |  127k	                |  CFID descriptors for materials project
    | mp_3d	             |  84k	                    |  CFID descriptors for 84k materials project
    | mxene275	         |  275	                    |  MXene dataset
    | ocp100k	         |  149886	                |  Open Catalyst 100000 training, rest validation and test dataset
    | ocp10k	         |  59886	                |  Open Catalyst 10000 training, rest validation and test dataset
    | ocp_all	         |  510214	                |  Open Catalyst 460328 training, rest validation and test dataset
    | omdb	             |  12500	                |  Bandgaps for organic polymers in OMDB database
    | oqmd_3d_no_cfid	 |  817636	                |  Formation energies and bandgaps of 3D materials from OQMD database
    | oqmd_3d	         |  460k	                |  CFID descriptors for 460k materials in OQMD
    | pdbbind_core	     |  195	                    |  Bio-molecular complexes database from PDBBind core
    | pdbbind	         |  11189	                |  Bio-molecular complexes database from PDBBind v2015
    | polymer_genome	 |  1073	                |  Electronic bandgap and diecltric constants of crystall ine polymer in polymer genome database
    | qe_tb	             |  829574	                |  Various 3D materials properties in JARVIS-QETB database
    | qm9_dgl	         |  130829	                |  Various properties of molecules in QM9 dgl database
    | qm9_std_jctc	     |  130829	                |  Various properties of molecules in QM9 database
    | qmof	             |  20425	                |  Bandgaps and total energies of metal organic frameowrks in QMOF database
    | raw_files	         |  144895	                |  Figshare links to download raw calculations VASP files from JARVIS-DFT
    | snumat	         |  10481	                |  Bandgaps with hybrid functional
    | ssub	             |  1726	                |  SSUB formation energy for chemical formula dataset
    | stm	             |  1132	                |  2D materials STM images in JARVIS-STM database
    | supercon_2d	     |  161	                    |  2D superconductor DFT dataset
    | supercon_3d	     |  1058	                |  3D superconductor DFT dataset
    | supercon_chem	     |  16414	                |  Superconductor chemical formula dataset
    | surfacedb	         |  607	                    |  Surface property dataset
    | tinnet_N	         |  329	                    |  TinNet Nitrogen catalyst dataset
    | tinnet_OH	         |  748	                    |  TinNet OH group catalyst dataset
    | tinnet_O	         |  747	                    |  TinNet Oxygen catalyst dataset
    | twod_matpd	     |  6351	                |  Formation energy and bandgaps of 2D materials properties in 2DMatPedia database
    | vacancydb	         |  464	                    |  Vacancy formation energy dataset
    | wtbh_electron	     |  1440	                |  3D and 2D materials Wannier tight-binding Hamiltonian dtaabase for electrons with spin-orbit coupling in JARVIS-WTB (Keyword: 'WANN')
    | wtbh_phonon	     |  15502	                |  3D and 2D materials Wannier tight-binding Hamiltonian for phonons at Gamma with finite difference (Keyword:FD-ELAST)
    ------------------------------------------------------------------------------------------
    ```


    **dft_3d (3D-materials curated data) Data Format (Example)**

    The dataset contains metadata for JARVIS-DFT data for 3D materials. 
    Specifically, the `dft_3d` dataset is a list of dictionaries, where each sample (`dict`) contains keys such as:
        Basic Information:
        ------------------
        - jid (str): Unique Jarvis material ID
        - formula (str): Chemical formula
        - search (str): Elemental search keyword
        - spg (int): Space group number (same as spg_number)
        - spg_number (int): Space group number
        - spg_symbol (str): Space group symbol
        - crys (str): Crystal system (e.g., tetragonal)
        - dimensionality (str): Material dimensionality (e.g., 3D bulk)
        - typ (str): Material type (e.g., bulk, monolayer)
        - reference (str): Cross-reference ID from Materials Project
        - icsd (str): ICSD database ID (if available)
        - xml_data_link (str): Link to full DFT result in XML format
        - raw_files (List): Raw files (if available)

        Crystal Structure:
        ------------------
        - atoms (dict[str, list[Any]]): 
            - lattice_mat (List): Lattice matrix
            - coords (List): Atomic coordinates
            - elements (List): Element types
            - abc (List): Lattice parameters
            - angles (List): Lattice angles
            - cartesian (bool): Whether coordinates are Cartesian
            - props (List): properties
        - nat (int): Number of atoms in the unit cell
        - density (float): Material density

        DFT Calculation Settings:
        -------------------------
        - func (str): Exchange-correlation functional used (e.g. OptB88vdW)
        - encut (int): Plane-wave energy cutoff
        - kpoint_length_unit (int): k-point sampling density
        
        Thermodynamic Properties:
        -------------------------
        - formation_energy_peratom (float): Formation energy per atom
        - optb88vdw_total_energy (float): Total DFT energy
        - ehull (float): Energy above the convex hull (measures stability)
        - exfoliation_energy (float): Exfoliation energies for van der Waals bonded materials
        
        Electronic Properties:
        ----------------------
        - optb88vdw_bandgap (float): Band gap from OptB88vdW functional
        - mbj_bandgap (float): Band gap from modified Becke-Johnson (MBJ) functional
        - hse_gap (float): Band gap from HSE hybrid functional
        - effective_masses_300K (dict): Effective masses of electrons and holes at 300K
        - avg_elec_mass (float): Average effective mass of electrons
        - avg_hole_mass (float): Average effective mass of holes

        Magnetic Properties:
        --------------------
        - magmom_outcar (float): Initial magnetic moment (from OSZICAR file)
        - magmom_oszicar (float): Final magnetic moment (from OUTCAR file)
        
        Dielectric and Optical Properties:
        ----------------------------------
        - epsx (float): Dielectric tensor component along x-axis
        - epsy (float): Dielectric tensor component along y-axis
        - epsz (float): Dielectric tensor component along z-axis
        - mepsx (float): Electronic contribution to dielectric constant along x
        - mepsy (float): Electronic contribution to dielectric constant along y
        - mepsz (float): Electronic contribution to dielectric constant along z
        - slme (float): Spectroscopy limited maximum efficiency
    
        Elastic and Mechanical Properties:
        ----------------------------------
        - elastic_tensor (List): Elastic tensor matrix
        - bulk_modulus_kv (float): Bulk modulus
        - shear_modulus_gv (float): Shear modulus
        - poisson (float): Poisson ratio
        - max_ir_mode (float): Maximum infrared (IR) mode intensity
        - min_ir_mode (float): Minimum infrared (IR) mode intensity
        - max_efg (float): Maximum electric field gradient
        - efg (float): Electric field gradients

        Thermoelectric Properties:
        --------------------------
        - n_seebeck (float): Seebeck coefficient for n-type carriers
        - p_seebeck (float): Seebeck coefficient for p-type carriers
        - ncond (float): Electrical conductivity for n-type carriers
        - pcond (float): Electrical conductivity for p-type carriers
        - nkappa (float): Thermal conductivity for n-type carriers
        - pkappa (float): Thermal conductivity for p-type carriers
        - n-powerfact (float): Power factor for n-type
        - p-powerfact (float): Power factor for p-type
        
        Vibrational and Phonon Properties:
        ----------------------------------
        - modes (List): Phonon modes
        - maxdiff_mesh (float): Maximum difference in mesh calculations
        - maxdiff_bz (float): Maximum difference in BZ calculations 

        Piezoelectric and Dielectric Tensor (DFPT):
        --------------------------------------------
        - dfpt_piezo_max_eij (float): Max piezoelectric tensor (strain-charge form)
        - dfpt_piezo_max_dij (float): Max piezoelectric tensor (stress-charge form)
        - dfpt_piezo_max_dielectric (float): Max total dielectric constant
        - dfpt_piezo_max_dielectric_electronic (float): Electronic part of dielectric constant
        - dfpt_piezo_max_dielectric_ionic (float): Ionic part of dielectric constant

        Superconductivity:
        ------------------
        - Tc_supercon (float): Superconducting critical temperature


    **Notes:**
        - Missing values are represented as `na`


    Args:
        path (str): The path of the dataset, if path is not exists, it will be downloaded.
        
        jarvis_data_name (str): The name of the jarvis dataset.

        property_names (Union[str, List[str]]): Property names you want to use,
            for jarvis dataset.

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

    def __init__(
        self, 
        path: str,
        jarvis_data_name: str,
        property_names: Union[str, List[str]],
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        # Extract jarvis dataset name, and construct dataset zip file path
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

        # Handle graph_cfg
<<<<<<< HEAD
<<<<<<< HEAD
=======
        # if build_graph_cfg is None:
        #     build_graph_cfg = {
        #         "__class_name__": "FindPointsInSpheres",
        #         "__init_params__": {"cutoff": 4},
        #         "__call_params__": {},
        #     }
        #     logger.message(
        #         "The build_graph_cfg is not set, will use the default "
        #         f"configs: {build_graph_cfg}"
        #     )
>>>>>>> Add Jarvis dataset support.
=======
>>>>>>> modify train loop logic
        self.build_graph_cfg = build_graph_cfg

        # Determine cache directory
        if build_graph_cfg is not None:
            graph_converter_name = re.sub(r'(?<!^)([A-Z])', r'_\1', build_graph_cfg['__class_name__']).lower()
            cutoff_name = str( int(build_graph_cfg['__init_params__']['cutoff']) )  
        else:
            graph_converter_name = 'none'
            cutoff_name = 'none'

        if cache_path is not None:
            self.cache_path = osp.join(
                cache_path, 
                jarvis_data_name + "_cache_" + graph_converter_name + "_cutoff_" + cutoff_name
            )
        else:
            # for example:
            # path = ./data/jarvis/
            # cache_path = ./data/jarvis/
            # self.path = ./data/jarvis/jdft_3d-12-12-2022.json.zip
            # self.cache_path = ./data/jarvis/dft_3d_cache_find_points_in_spheres_cutoff_4/
            self.cache_path = osp.join(
                path,
                jarvis_data_name + "_cache_" + graph_converter_name + "_cutoff_" + cutoff_name
            )
        logger.info(f"Cache path: {self.cache_path}")
        self.cache_exists = True if osp.exists(self.cache_path) else False

        # Additional parameters
        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # Load raw Jarvis dataset
        self.raw_data, self.num_samples = self.read_data(path=self.path, data_name=jarvis_data_name)
        logger.info(f"Load {self.num_samples} samples from {self.path}")

        # Extract property values from raw dataset
        self.property_data = self.read_property_data(data=self.raw_data, property_names=self.property_names)

        # Check structure and graph config
        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache exists. If used, the current configuration will be ignored."
                "Ensure your current settings match the cached configuration."
            )
            # Check structure config
            try:
                build_structure_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl")
                )
                if is_equal(build_structure_cfg_cache, build_structure_cfg):
                    logger.info(
                        "The cached build_structure_cfg configuration matches "
                        "the current settings. Reusing previously generated"
                        " structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_structure_cfg is different from "
                        "build_structure_cfg_cache. Will rebuild the structures and "
                        "graphs."
                    )
                    logger.warning(
                        "If you want to use the cached structures and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_structure_cfg.pkl from cache. "
                    "Will rebuild the structures and graphs(if need)."
                )
                overwrite = True
            # Check graph config
            if build_graph_cfg is not None and not overwrite:
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
        
        # Set cache directory of structures and graphs 
        structure_cache_path = osp.join(self.cache_path, "structures")
        graph_cache_path = osp.join(self.cache_path, "graphs")

        if overwrite or not self.cache_exists:
            # convert strucutes and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_structure_cfg and build_graph_cfg to cache file
                os.makedirs(self.cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), 
                    build_graph_cfg,
                )
                
                # convert strucutes
                structures = BuildStructure(**build_structure_cfg)(
                    self.raw_data["atoms"]
                )
                # save structures to cache file 
                os.makedirs(structure_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info( f"Save {self.num_samples} structures to {structure_cache_path}" )

                # convert graphs
                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(structures)
                    # save graphs to cache file
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), 
                            graphs[i],
                        )
                    logger.info( f"Save {self.num_samples} graphs to {graph_cache_path}" )
                    del graphs
                del structures 
            
            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        
        self.structures = [
            osp.join(structure_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None
            logger.warning("Graph configuration not provided. Graph data will not be generated or loaded.")

        assert (
            len(self.structures) == self.num_samples
        ), "The number of structures must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # filter by property data, since some samples may have no valid properties
        if filter_unvalid:
            self.filter_unvalid_by_property()
        
                       
    def read_data(
        self,
        path: str,
        data_name: str,
    ):
        """
        Load jarvis data, and convert from list-of-dict to dict-of-lists by property.

        Args:
            path (str): The directory of data file.
            data_name (str): Name of the jarvis data.

        Returns:
            property_data (dict[str, list[Any]]): 
                Key is a property name, and 
                value is a list containing that property's values for all samples.
            num_samples (int): 
                Total number of samples in the dataset.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        property_data = {}
        
        # If file is missing or invalid, remove it and redownload
        if not osp.exists(path) or not zipfile.is_zipfile(path):
            if osp.exists(path):
                logger.message(f"Invalid Jarvis dataset zip archive detected at '{path}'. Delete it and initiating re-download of dataset '{data_name}'.")
                os.remove(path)
            else:
                logger.message(f"Jarvis dataset zip archive not found. Downloading of dataset {data_name}.")
            try:
                raw_data = jdata(dataset=data_name, store_dir=os.path.dirname(path))
                assert raw_data is not None, f"Failed to download dataset {data_name}"
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset {data_name}. Error: {e}")
        # File is valid
        else:
            logger.message(f"Existing Jarvis dataset zip archive found at '{path}'.")
            raw_data = json.loads( zipfile.ZipFile(path).read(os.path.splitext(os.path.basename(path))[0]) )

        # Convert list-of-dict raw data to dict-of-lists by property.
        property_data = defaultdict(list)
        num_samples = len(raw_data)

        # Test, only load part of data
        # for idx, item in enumerate(raw_data):
        #     if idx == 301:
        #         break
        #     for key, value in item.items():
        #         property_data[key].append(value)
        # num_samples = len(property_data[key])
        
        for item in raw_data:
            for key, value in item.items():
                property_data[key].append(value)

        for key, value in dict(property_data).items():
            if len(value) != num_samples:
                raise ValueError(f"Property {key} has different length than other properties.")
        
        return dict(property_data), num_samples


    def read_property_data(
        self, 
        data: Dict, 
        property_names: List[str]
    ):
        """
        Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
            property_names (List[str]): Property names.

        Returns:
            property_data (dict[str, list[Any]]): 
                Key is a property name, and 
                value is a list containing that property's values for all samples.
        """
        property_data = {}
        for property_name in property_names:
            if property_name not in data:
                raise ValueError(f"{property_name} not found in the data")
            property_data[property_name] = data[property_name]
        return property_data

    def filter_unvalid_by_property(self):
        """
        Filter out samples that have invalid properties (e.g., NaN, string, or None).

        This method updates:
            - self.structures
            - self.graphs (if not None)
            - self.num_samples

        Returns:
            None
        """

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

            # self.raw_data = [self.raw_data[i] for i in reserve_idx]
            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"Filter out {len(reserve_idx)} samples with valid properties: "
                f"{property_name}"
            )
        self.num_samples = len(self.structures)
        logger.warning(f"Remaining {self.num_samples} samples after filtering.")

    def save_to_cache(
        self, 
        cache_path: str, 
        data: Any
    ):
        """
        Save data to a cache file.

        Args:
            cache_path (str): The path to the cache file.
            data (Any): The data to be saved.

        Returns:
            None

        """
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(
        self, 
        cache_path: str
    ):
        """
        Load data from a cached .pkl file.

        Args:
            cache_path (str): The path to the cached file.

        Returns:
            data: The data loaded from the cache.
        """

        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

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
