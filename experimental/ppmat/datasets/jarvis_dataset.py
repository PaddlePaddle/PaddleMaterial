import os
import os.path as osp
import json
import zipfile

from typing import Optional, Union, Dict, List, Any, Tuple, Callable
from collections import defaultdict

# import pickle

# import numpy as np

# import paddle.distributed as dist
from paddle.io import Dataset

from jarvis.db.figshare import data as jdata
from jarvis.db.figshare import get_db_info

from ppmat.utils import logger


class JarvisDataset(Dataset):
    """Jarvis Dataset Handler.
    
    **Dataset Overview**
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┐
    │ Dataset Name      │ dft_3d  │         │         │
    ├───────────────────┼─────────┼─────────┼─────────┤
    │ Sample Count      │ 75993   │         │         │
    └───────────────────┴─────────┴─────────┴─────────┘
    ```
    Download preprocessed data: https://jarvis-materials-design.github.io/dbdocs/thedownloads/

    **3D-materials curated data (dft_3d) Data Format**

    The dataset contains metadata for JARVIS-DFT data for 3D materials. 
    Specifically, the `dft_3d` dataset is a list of dictionaries, where each sample (`dict`) contains keys such as:
        - 'jid', 
        - 'atoms', 
        - 'formation_energy_peratom', 
        - 'optb88vdw_bandgap', 
        - 'elastic_tensor',
        - 'effective_masses_300K', 
        - 'kpoint_length_unit', 
        - 'encut',
        - 'optb88vdw_total_energy', 
        - 'mbj_bandgap', 
        - 'epsx', 
        - 'mepsx', 
        - 'epsy',
        - 'mepsy', 
        - 'epsz', 
        - 'mepsz', 
        - 'kpoints_array', 
        - 'bulk_modulus_kv', 
        - 'shear_modulus_gv', 
        - 'modes', 
        - 'magmom_outcar',
        - 'magmom_oszicar', 
        - 'icsd', 
        - 'spillage', 
        - 'slme', 
        - 'dfpt_piezo_max_eij',
        - 'dfpt_piezo_max_dij', 
        - 'dfpt_piezo_max_dielectric',
        - 'dfpt_piezo_max_dielectric_electronic', 
        - 'dfpt_piezo_max_dielectric_ionic', 
        - 'max_ir_mode', 
        - 'min_ir_mode', 
        - 'n-Seebeck', 
        - 'p-Seebeck', 
        - 'exfoliation_energy', 
        - 'n-powerfact', 
        - 'p-powerfact', 
        - 'ehull', 
        - 'dfpt_piezo_max_dielectric_ioonic'
    
    """

    def __init__(
        self, 
        path: str,
        jarvis_data_name: str,
        property_names: Union[str, List[str]],
        # build_structure_cfg: Dict = None,
        # build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()

        if isinstance(property_names, str):
            property_names = [property_names]  
        self.property_names = property_names if property_names is not None else []

        self.raw_data, self.num_samples = self.read_data(path=path, data_name=jarvis_data_name)
        logger.info(f"Load {self.num_samples} samples from {osp.join(path, jarvis_data_name+'.zip')}")
        self.property_data = self.read_property_data(data=self.raw_data, property_names=self.property_names)

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used match your current settings."
            )
            
    def read_data(
        self,
        path: str,
        data_name: str,
    ):
        """
        Load jarvis data, and convert from list-of-dict to dict-of-lists by property.

        Args:
            path (str): Path to the directory containing the data files.
            data_name (str): Name of the jarvis data.

        Returns:
            property_data (dict[str, list[Any]]): 
                Key is a property name, and 
                value is a list containing that property's values for all samples.
            num_samples (int): 
                Total number of samples in the dataset.
        """

        property_data = {}
        os.makedirs(path, exist_ok=True)
        
        # Extract file names and paths
        db_info = get_db_info()
        if data_name not in db_info:
            raise ValueError(f"Unknown dataset name: {data_name}")
        _, filename, _, _ = db_info[data_name]
        raw_data_zip = osp.join(path, filename + ".zip")

        # If file is missing or invalid, remove it and redownload
        if not osp.exists(raw_data_zip) or not zipfile.is_zipfile(raw_data_zip):
            if osp.exists(raw_data_zip):
                logger.warning(f"Invalid Jarvis zip file detected: {raw_data_zip}, re-downloading: {data_name}")
                os.remove(raw_data_zip)
            else:
                logger.info(f"Jarvis zip file not found. Downloading: {data_name}")
            raw_data = jdata(dataset=data_name, store_dir=path)
        # File is valid
        else:
            logger.info(f"Valid zip file found: {raw_data_zip}.")
            raw_data = json.loads(zipfile.ZipFile(raw_data_zip).read(filename))

        num_samples = len(raw_data)
        
        # Convert list-of-dict raw data to dict-of-lists by property.
        property_data = defaultdict(list)
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

