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
# List available datasets in the format similar to mp2018_dataset
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

    **Jarvis Dataset Overview**

    Download preprocessed data: https://jarvis-materials-design.github.io/dbdocs/thedownloads/
    Github: https://github.com/usnistgov/jarvis/tree/master

    ```
    -------------------------------------------------------------------------------
    | Database name      |  Number of data-points	|  Description
    -------------------------------------------------------------------------------
    | AGRA_CHO	         |  214	         |  AGRA CHO catalyst dataset
    | AGRA_COOH	         |  280	         |  AGRA COOH catalyst dataset
    | AGRA_CO	         |  193	         |  AGRA CO catalyst dataset
    | AGRA_OH	         |  875	         |  AGRA OH catalyst dataset
    | AGRA_O	         |  1000	     |  AGRA Oxygen catalyst dataset
    | aflow2	         |  400k	     |  AFLOW dataset
    | alex_pbe_1d_all	 |  100k	     |  Alexandria DB all 1D materials with PBE
    | alex_pbe_2d_all	 |  200k	     |  Alexandria DB all 2D materials with PBE
    | alex_pbe_3d_all	 |  5 million	 |  Alexandria DB all 3D materials with PBE
    | alex_pbe_hull	     |  116k	     |  Alexandria DB convex hull stable materials
                                        with PBE functional
    | alex_pbesol_3d_all |	500k	     |  Alexandria DB all 3D materials
                                        with PBEsol
    | alex_scan_3d_all	 |  500k	     |  Alexandria DB all 3D materials
                                        with SCAN
    | alignn_ff_db	     |  307113	     |  Energy per atom, forces and stresses
                                        for ALIGNN-FF trainig for 75k materials.
    | arXiv	             |  1796911	     |  arXiv dataset 1.8 million title,
                                        abstract and id dataset
    | arxiv_summary	     |  137927	     |  arXiv summary dataset
    | c2db	             |  3514	     |  Various properties in C2DB database
    | cccbdb	         |  1333	     |  CCCBDB dataset
    | cfid_3d	         |  55723	     |  Various 3D materials properties
                                        in JARVIS-DFT database computed
                                        with OptB88vdW and TBmBJ methods with CFID
    | cod	             |  431778	     |  Atomic structures from
                                crystallographic open database
    | dft_2d_2021	     |  1079	     |  Various 2D materials
        properties in JARVIS-DFT database computed with OptB88vdW
    | dft_2d	         |  1109	     |  Various 2D materials properties
                            in JARVIS-DFT database computed with OptB88vdW
    | dft_3d_2021	     |  55723	     |  Various 3D materials properties in
                                JARVIS-DFT database computed with
                                OptB88vdW and TBmBJ methods
    | dft_3d	         |  75993	     |  Various 3D materials
        properties in JARVIS-DFT database computed with OptB88vdW and TBmBJ methods
    | edos_pdos	         |  48469	     |  Normalized electron and phonon density
                    of states with interpolated values and fixed number of bins
    | halide_peroskites	 |  229	         |  Halide perovskite dataset
    | hmof	             |  137651	     |  Hypothetical MOF database
    | hopv	             |  4855	     |  Various properties of molecules
                                        in HOPV15 dataset
    | interfacedb	     |  593	         |  Interface property dataset
    | jff	             |  2538	     |  Various 3D materials properties in
                            JARVIS-FF database computed with several force-fields
    | m3gnet_mpf_1.5mil	 |  1.5 million	 |  1.5 million structures and their energy,
                                        forces and stresses in MP
    | m3gnet_mpf	     |  168k	     |  168k structures and their energy,
                                        forces and stresses in MP
    | megnet2	         |  133k	     |  133k materials and their
                                        formation energy in MP
    | megnet	         |  69239	     |  Formation energy and bandgaps
                            of 3D materials properties in Materials project database
                            as on 2018, used in megnet
    | mlearn	         |  1730	     |  Machine learning force-field
                                    for elements datasets
    | mp_3d_2020	     |  127k	     |  CFID descriptors for materials project
    | mp_3d	             |  84k	         |  CFID descriptors for 84k materials project
    | mxene275	         |  275	         |  MXene dataset
    | ocp100k	         |  149886	     |  Open Catalyst 100000 training,
                                        rest validation and test dataset
    | ocp10k	         |  59886	     |  Open Catalyst 10000 training,
                                        rest validation and test dataset
    | ocp_all	         |  510214	     |  Open Catalyst 460328 training,
                                        rest validation and test dataset
    | omdb	             |  12500	     |  Bandgaps for organic polymers
                                        in OMDB database
    | oqmd_3d_no_cfid	 |  817636	     |  Formation energies
                    and bandgaps of 3D materials from OQMD database
    | oqmd_3d	         |  460k	     |  CFID descriptors for 460k materials in OQMD
    | pdbbind_core	     |  195	         |  Bio-molecular complexes database
                                    from PDBBind core
    | pdbbind	         |  11189	     |  Bio-molecular complexes database
                                    from PDBBind v2015
    | polymer_genome	 |  1073	     |  Electronic bandgap and diecltric constants
                    of crystall ine polymer in polymer genome database
    | qe_tb	             |  829574	     |  Various 3D materials properties
                                        in JARVIS-QETB database
    | qm9_dgl	         |  130829	     |  Various properties of molecules
                                        in QM9 dgl database
    | qm9_std_jctc	     |  130829	     |  Various properties of molecules
                                        in QM9 database
    | qmof	             |  20425	     |  Bandgaps and total energies of
                                        metal organic frameowrks in QMOF database
    | raw_files	         |  144895	     |  Figshare links to download
                                        raw calculations VASP files from JARVIS-DFT
    | snumat	         |  10481	     |  Bandgaps with hybrid functional
    | ssub	             |  1726	     |  SSUB formation energy
                                        for chemical formula dataset
    | stm	             |  1132	     |  2D materials STM images
                                        in JARVIS-STM database
    | supercon_2d	     |  161	         |  2D superconductor DFT dataset
    | supercon_3d	     |  1058	     |  3D superconductor DFT dataset
    | supercon_chem	     |  16414	     |  Superconductor chemical formula dataset
    | surfacedb	         |  607	         |  Surface property dataset
    | tinnet_N	         |  329	         |  TinNet Nitrogen catalyst dataset
    | tinnet_OH	         |  748	         |  TinNet OH group catalyst dataset
    | tinnet_O	         |  747	         |  TinNet Oxygen catalyst dataset
    | twod_matpd	     |  6351	     |  Formation energy and bandgaps
                        of 2D materials properties in 2DMatPedia database
    | vacancydb	         |  464	         |  Vacancy formation energy dataset
    | wtbh_electron	     |  1440	     |  3D and 2D materials
        Wannier tight-binding Hamiltonian database
        for electrons with spin-orbit coupling in JARVIS-WTB (Keyword: 'WANN')
    | wtbh_phonon	     |  15502	     |  3D and 2D materials
        Wannier tight-binding Hamiltonian
        for phonons at Gamma with finite difference (Keyword:FD-ELAST)
    -------------------------------------------------------------------------------
    dft_3d (3D-materials curated data) Data Format (Example)**

    The dataset contains metadata for JARVIS-DFT data for 3D materials.
    Specifically, the `dft_3d` dataset is a list of dictionaries,
    where each sample (`dict`) contains keys such as:

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
        - exfoliation_energy (float):
            Exfoliation energies for van der Waals bonded materials

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
        - dfpt_piezo_max_eij (float):
            Max piezoelectric tensor (strain-charge form)
        - dfpt_piezo_max_dij (float):
            Max piezoelectric tensor (stress-charge form)
        - dfpt_piezo_max_dielectric (float):
            Max total dielectric constant
        - dfpt_piezo_max_dielectric_electronic (float):
            Electronic part of dielectric constant
        - dfpt_piezo_max_dielectric_ionic (float):
            Ionic part of dielectric constant

        Superconductivity:
        ------------------
        - Tc_supercon (float): Superconducting critical temperature


    **Notes:**
        - Missing values are represented as `na`


    Args:
        path (str): The path of the dataset,
            if path is not exists, it will be downloaded.

        jarvis_data_name (str): The name of the jarvis dataset. Default is "custom".

        property_names (Union[str, List[str]]): Property names you want to use,
            for jarvis dataset.

        url (Optional[str], optional): Custom dataset download URL. If provided,
            the dataset will be downloaded from this URL instead of checking the
            registry. This supports specific datasets like jdft_2d.

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

        Args:
            path (str): The directory of data file.
            data_name (str): Name of the jarvis data.
            url (str, optional): Direct URL to download dataset from.

        Returns:
            property_data (dict[str, list[Any]]):
                Key is a property name, and
                value is a list containing that property's values for all samples.
            num_samples (int):
                Total number of samples in the dataset.
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

    def save_to_cache(self, cache_path: str, data: Any):
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

    def load_from_cache(self, cache_path: str):
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

    def filter_unvalid_by_property(self):
        """
        Filter out samples that have invalid properties (e.g., NaN, string, or None).

        This method updates:
            - self.structures
            - self.graphs (if not None)
            - self.property_data
            - self.num_samples

        Returns:
            None
        """
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            old_num_structs = len(self.structures)

            for i, data_item in enumerate(data):
                # Convert 'na' strings to NaN for proper filtering
                if isinstance(data_item, str):
                    if data_item.lower() in ["na", "nan", "none", ""]:
                        data_item = np.nan
                    else:
                        # Skip non-numeric strings
                        continue
                # Keep only valid numeric values
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

        This method updates:
            - self.structures
            - self.graphs (if not None)
            - self.property_data
            - self.num_samples

        Returns:
            None
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
        """Get item at index idx."""
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
                # Check for 'na' strings
                if isinstance(value, str) and value.lower() in [
                    "na",
                    "nan",
                    "none",
                    "",
                ]:
                    raise ValueError(
                        f"Found invalid property value '{value}' at index {idx} for property "  # noqa
                        f"'{property_name}'. This should have been filtered out during dataset "  # noqa
                        f"initialization. Please ensure 'filter_unvalid=True' is set and "  # noqa
                        f"consider clearing the cache to regenerate filtered data."
                    )
                # Check for NaN values - these should also have been filtered out
                if value is not None and (
                    isinstance(value, float) and math.isnan(value)
                ):
                    raise ValueError(
                        f"Found NaN value at index {idx} for property '{property_name}'. "  # noqa
                        f"This should have been filtered out during dataset initialization."  # noqa
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
