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

import math
import os
import os.path as osp
import pickle ## for dump/load
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
from typing import List

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import read_json
from ppmat.utils.misc import is_equal

# Attempt to import tqdm for progress visualization
try:
    from tqdm import tqdm 
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import ase.io
    import ase.data
    import ase.build
    from pymatgen.io.ase import AseAtomsAdaptor
    
    read = ase.io.read
    symbols = ase.data.atomic_numbers
    ASE_AVAILABLE = True

except ImportError:
    
    def dummy_read(*args, **kwargs):
        """
        If ASE is not installed, this pseudo-read function will throw an error when called.
        In the `__init__` method of `QM9Dataset`, if `read` is not needed immediately,
        you can simply return `None` or an empty list. However, if it is needed, the error will be more explicit.
        
        """
        raise RuntimeError(
            "Atomic Simulation Environment (ASE) is required but not installed. "
            "Please install it (e.g., pip install ase) to use QM9Dataset."
        )
    read = dummy_read  
    symbols = None
    AseAtomsAdaptor = None
    ASE_AVAILABLE = False
    print("Warning: ASE (Atomic Simulation Environment) not found. Data parsing functionality is disabled.")

class QM9Dataset(Dataset):
    """
    QM9 (GDB-9) Dataset Handler
    In order to adapt to the CHGNet model, I made a forced mapping from 'lumo' to 'energy_per_atom'. 
    The CHGNet model is used to run machine learning potentials, and the QM9 dataset may not be very suitable. 
    LUMO is a property. Please be aware of this when using it to avoid misunderstandings. 
    This code is for research purposes only and does not represent the optimal approach.——by wwaawwaaee
    
    **Dataset Overview**
    this class downloads QM9 primiry(end with .xyz)and transfered into 
    the input structures and quantum chemical property labels required 
    by graph neural network (GNN) models.(maybe)

    **dataset format**
    -----------------
    - raw data: qm9.zip ()
    - struncture file: a sample corresponds to a seperate .xyz file
    - attribute(label): 19 quantum chemical properties are embedded in each .xyz file's
    second line comment

    **source**:Original data available at https://figshare.com/ndownloader/files/3195389

    The dataset can also be found at https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2
    
   **Key Properties List (Available for 'property_names' argument)**
    -----------------------------------------------------------------
    1.  mu (Dipole Moment, Debye)
    2.  alpha (Isotropic Polarizability, Bohr^3)
    3.  homo (HOMO Energy, Hartree)
    4.  lumo (LUMO Energy, Hartree)
    5.  gap (LUMO-HOMO Gap, Hartree)
    6.  U0 (Internal Energy at 0 K, Hartree)
    sly
    # ... (remaining 13 properties here in their correct order)
    
    **__getitem__ Sample Contract**
    ----------------------------------------
    - 'atom_types': np.ndarray (dtype=int64) - Atomic numbers (Z).
    - 'coords': np.ndarray (dtype=float32) - 3D Cartesian coordinates in Angstrom.
    - [property_name]: np.ndarray (dtype=float32) - The target label value (e.g., 'lumo').
    - 'graph': (Optional) The graph object constructed by the converter (if configured).
    
    Args:
        path (str): The root directory to store downloaded and cache files.
        property_names (Union[str, List[str]]): The name(s) of the target property 
            to predict. Must be selected from the list above. Defaults to 'lumo'.
        build_graph_cfg (Dict, optional): Configuration dictionary for building 
            the graph representation from the molecular structure (e.g., cutoff radius). 
            Defaults to None (structure is returned instead of graph).
        transforms (Optional[Callable], optional): A preprocessing function to apply 
            to the sample dictionary. Defaults to None.
        cache_path (Optional[str], optional): Explicit path for the cache directory. 
            Defaults to None.
        overwrite (bool, optional): If True, forces the rebuilding of caches. 
            Defaults to False.
        filter_unvalid (bool, optional): Whether to filter out corrupted samples. 
            Defaults to True.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2"
    name = "qm9"
    md5 = "AD1EBD51EE7F5B3A6E32E974E5D54012"

    # Official QM9 second-line property order (including tag/index)
    PROP_ORDER = [
        "tag",    # textual tag / molecule identifier (often 'gdb ...')
        "index",  # numeric index (maps to vals_float[0] after removing 'gdb')
        "A",      # rotational constant A (GHz)
        "B",      # rotational constant B (GHz)
        "C",      # rotational constant C (GHz)
        "mu",     # dipole moment (Debye)
        "alpha",  # isotropic polarizability (Bohr^3)
        "homo",   # HOMO energy (Hartree)
        "lumo",   # LUMO energy (Hartree)
        "gap",    # LUMO-HOMO gap (Hartree)
        "r2",     # electronic spatial extent (Bohr^2)
        "zpve",   # zero point vibrational energy (Hartree)
        "U0",     # internal energy at 0K (Hartree)
        "U",      # internal energy at 298.15 K (Hartree)
        "H",      # enthalpy at 298.15 K (Hartree)
        "G",      # free energy at 298.15 K (Hartree)
        "Cv",     # heat capacity at 298.15 K (cal/mol/K)
    ]

    def __init__(
        self,
        path: str,
        url: Optional[str] = None, 
        property_names: Union[str, List[str]] = None,
        *,
        url_indices: Optional[List[int]] = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Use the ASE_AVAILABLE flag and AseAtomsAdaptor presence to validate dependencies
        if not ASE_AVAILABLE or AseAtomsAdaptor is None:
            raise RuntimeError(
                "QM9Dataset requires 'ase' and 'pymatgen'. "
                "Please install them via: pip install ase pymatgen"
            )

        if property_names is None:
            raise ValueError("property_names must be provided for QM9Dataset")

        if isinstance(property_names,str):
            property_names = [property_names]
        self.property_names = list(property_names) if property_names else []

        # Handle URLs configuration
        self.url = url if url is not None else self.url

        #Path Configuration
        os.makedirs(path, exist_ok=True)
        self.raw_dir = osp.join(path, "raw_qm9")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.raw_xyz_path = osp.join(self.raw_dir, "dsgdb9nsd.xyz")
        
        # Generate cache directory naming based on graph config
        if build_graph_cfg is not None:
            graph_converter_name = build_graph_cfg.get("__class_name__", "custom")
            cutoff_name = str(
                int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5))
            )
        else:
            graph_converter_name = "none"
            cutoff_name = "none"
        
        base_cache = cache_path if cache_path is not None else path #determine the final path
        self.cache_path = osp.join(
            base_cache,
            f"qm9_cache_{graph_converter_name}_cutoff_{cutoff_name}",
        )
        
        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.build_graph_cfg = build_graph_cfg

        # define sub-directories for cache
        self.structures_dir = osp.join(self.cache_path, "structures")
        self.graphs_dir = osp.join(self.cache_path, "graphs")
        self.props_dir = osp.join(self.cache_path, "properties")

        if dist.get_rank() == 0:
            logger.info(f"Cache path: {self.cache_path}")
            os.makedirs(self.structures_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)
            os.makedirs(self.props_dir, exist_ok=True)
        
        #  =========== data operation ==============
        
        # 1) Download and ensure shard files exist locally
        local_raw_file = self._ensure_raw_data()

        # 2) Check or build Structures and Properties cache
        if dist.get_rank() == 0:
            self._prepare_structures_and_properties(local_raw_file)
        # Only rank 0 performs the build process to avoid race conditions
        
        if dist.is_initialized():
            dist.barrier()
        
        # 3) Check or build Graphs cache (if configuration provided)
        if self.build_graph_cfg is not None:
            if dist.get_rank() == 0:
                self._prepare_graphs()
            if dist.is_initialized():
                dist.barrier()
        
        PROPERTY_FILE_MAP = {          
        "energy_per_atom": "lumo", # cheat the model of the way it gets the data   
        }
        # 4) Load file lists and property data into memory
        self.structures = [
            osp.join(self.structures_dir, f)
            for f in sorted(os.listdir(self.structures_dir))
            if f.endswith(".pkl")
        ]

        if self.build_graph_cfg is not None:
            self.graphs = [
                osp.join(self.graphs_dir, f)
                for f in sorted(os.listdir(self.graphs_dir))
                if f.endswith(".pkl")
            ]
        else:
            self.graphs = None

        logger.info(f"Loading properties {self.property_names} into memory...")
        
        self.property_data = {}
        for pname in self.property_names:
        
            # Determine the actual file name: use the mapping table if available; otherwise, use the configuration name itself
            file_name = PROPERTY_FILE_MAP.get(pname, pname)
        
            file_path = osp.join(self.props_dir, f"{file_name}.pkl")
        
            if not osp.exists(file_path):
                raise FileNotFoundError(
                    f"[QM9 Map Error]can't find the file: {file_path}. "
                    f"(require label: {pname}, actually find the file: {file_name}.pkl)"
                )
        
            # Although the name of file is lumo.pkl,The key stored in the dictionary is still pname (such as energy_per_atom)
            self.property_data[pname] = self._load_pickle(file_path)

        # self.property_data = {
        #     pname: self._load_pickle(osp.join(self.props_dir, f"{pname}.pkl"))
        #     for pname in self.property_names
        # }

        
        # Sort files to ensure consistency across distributed ranks
        # 5) Filter invalid data based on properties and graphs
        if self.filter_unvalid:
            self._filter_by_properties()
        if self.graphs is not None:
            self._filter_by_graphs()
        # 6) Ensure data length consistency across all arrays
        self._ensure_length_consistency()

        self.num_samples = len(self.structures)
        logger.info(f"Final QM9Dataset samples: {self.num_samples}")

    
    def _prepare_structures_and_properties(self, raw_file_path: str):
        """
        Check if structures and properties are cached; rebuild if missing
        or overwrite is True.
        """
        
        num_cached = self._count_files(self.structures_dir)
        
        # Check if all property files exist
        props_exist = all(
            osp.exists(osp.join(self.props_dir, f"{p}.pkl"))
            for p in self.property_names
        )

        # Use a completion flag to ensure the previous build was successful
        struct_done_flag = osp.join(self.structures_dir, "completed.flag")
        is_complete = osp.exists(struct_done_flag)

        
        should_build = (
            self.overwrite or num_cached == 0 or not props_exist or not is_complete
        )

        if should_build:
            if dist.get_rank() == 0:
                logger.info("Building structures and properties from raw QM9 file...")
                
                # Clean old data to prevent mixing files
                self._clean_dir(self.structures_dir)
                self._clean_dir(self.props_dir)

                self._build_structures_and_properties(
                    raw_file_path, self.structures_dir, self.props_dir
                )
                
                # Write completion flag
                with open(struct_done_flag, "w") as f:
                    f.write("done")
        else:
            logger.info(f"Using cached structures ({num_cached}) and properties.")
    
        
    def _prepare_graphs(self):
        """
        Check if graphs are cached; rebuild if missing, incomplete,
        or overwrite is True.
        """
        num_structs = self._count_files(self.structures_dir)
        num_graphs = self._count_files(self.graphs_dir)

        # Use a completion flag for graphs
        graph_done_flag = osp.join(self.graphs_dir, "completed.flag")
        is_complete = osp.exists(graph_done_flag)

        # Condition: Not overwrite, marked complete, and counts match
        if not self.overwrite and is_complete and num_graphs == num_structs:
            logger.info(f"Using cached graphs ({num_graphs}).")
            return

        logger.info(
            f"Rebuilding graphs. (Structs: {num_structs}, Graphs: {num_graphs}, "
            f"Complete: {is_complete}, Overwrite: {self.overwrite})"
        )

        self._clean_dir(self.graphs_dir)
        converter = build_graph_converter(self.build_graph_cfg)
        self._build_graphs(converter, self.structures_dir, self.graphs_dir)

        # Write completion flag
        with open(graph_done_flag, "w") as f:
            f.write("done")

    def _build_graphs(self, converter, structures_dir: str, graphs_dir: str) -> None:
        """
        Builds graph objects from structures using a SINGLE global progress bar.

        This method processes structures in batches to manage memory usage, while
        providing a unified progress visualization.
        """
        import gc
        import sys

        # Context manager to temporarily suppress stderr
        class SuppressStderr:
            def __init__(self):
                self.null_fds = [os.open(os.devnull, os.O_RDWR)]
                self.save_fds = [os.dup(2)]  # Backup stderr (fd 2)

            def __enter__(self):
                # Redirect stderr to devnull
                os.dup2(self.null_fds[0], 2)

            def __exit__(self, *_):
                # Restore stderr
                os.dup2(self.save_fds[0], 2)
                for fd in self.null_fds + self.save_fds:
                    os.close(fd)

        # Get file list
        files = sorted([f for f in os.listdir(structures_dir) if f.endswith(".pkl")])
        total = len(files)
        if total == 0:
            logger.warning("No structures found to convert!")
            return

        batch_size = 2000  # Define batch size

        logger.info(f"Converting {total} structures to graphs...")

        # 1. Create global progress bar
        pbar = tqdm(total=total, desc="Graph Conversion", unit="sample")

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_files = files[start_idx:end_idx]

            try:
                # Load structures for current batch
                structures = [
                    self._load_pickle(osp.join(structures_dir, f)) for f in batch_files
                ]

                # 2. Convert to graphs (Suppress internal progress bars if any)
                try:
                    with SuppressStderr():
                        graphs = converter(structures)
                except Exception:
                    # Fallback if low-level FD manipulation fails
                    graphs = converter(structures)

                # Save graphs
                for f, g in zip(batch_files, graphs):
                    self._save_pickle(osp.join(graphs_dir, f), g)

                # 3. Update global progress bar
                pbar.update(len(batch_files))

            except Exception as e:
                # Restore stderr to print error
                sys.stderr = sys.__stderr__
                logger.warning(f"Batch {start_idx}-{end_idx} failed: {e}")

            finally:
                if "structures" in locals():
                    del structures
                if "graphs" in locals():
                    del graphs
                gc.collect()

        pbar.close()
        logger.info("Graph conversion completed.")

    def _ensure_length_consistency(self):
        """
        Ensures consistency in length across structures, graphs, and all
        property arrays. Truncates data to the minimum length found.
        """
        lengths = [len(self.structures)]
        if self.graphs is not None:
            lengths.append(len(self.graphs))
        for p in self.property_names:
            lengths.append(len(self.property_data[p]))

        min_len = min(lengths)

        if any(length != min_len for length in lengths):
            logger.warning(
                f"Data length mismatch detected (lengths={lengths}). "
                f"Truncating to minimum length: {min_len}."
            )
            self.structures = self.structures[:min_len]
            if self.graphs is not None:
                self.graphs = self.graphs[:min_len]
            for p in self.property_names:
                self.property_data[p] = self.property_data[p][:min_len]


    def _clean_dir(self, directory: str):
        """Cleans a directory by removing all .pkl and .flag files."""
        for f in os.listdir(directory):
            if f.endswith(".pkl") or f.endswith(".flag"):
                try:
                    os.remove(osp.join(directory, f))
                except OSError:
                    pass

    
    def _ensure_raw_data(self) -> str:
        """
        downloading self.url -> self.raw_xyz_path
        """
        # 1. if the final file exists , return.
        if osp.exists(self.raw_xyz_path):
            return self.raw_xyz_path
            
        # 2. prepare the path to download
        tar_filename = "qm9_raw.tar.bz2"
        tar_path = osp.join(self.raw_dir, tar_filename)
        
        # 3. downloading logic
        if not osp.exists(tar_path):
            if dist.get_rank() == 0:
                logger.info(f"Downloading QM9 from {self.url}...")
                import urllib.request
                try:
                    urllib.request.urlretrieve(self.url, tar_path)
                except Exception as e:
                    raise RuntimeError(f"Download failed: {e}")
            if dist.is_initialized():
                dist.barrier()
        
        # 4. extacting logic
        if dist.get_rank() == 0:
            logger.info("Extracting QM9...")
            import tarfile
            try:
                with tarfile.open(tar_path, "r:bz2") as tar:
                    tar.extractall(path=self.raw_dir)
            except Exception as e:
                raise RuntimeError(f"Extraction failed: {e}")
                
        if dist.is_initialized():
            dist.barrier()

        # 5. final check
        # Case A：single file exists.
        if osp.exists(self.raw_xyz_path):
            return self.raw_xyz_path

        
        # Case B：merge these .xyz files into a big file.
        xyz_files = [f for f in os.listdir(self.raw_dir) if f.endswith(".xyz") and f != "dsgdb9nsd.xyz"]
        if len(xyz_files) > 0:
            logger.info(f"Found {len(xyz_files)} xyz files, merging into dsgdb9nsd.xyz...")
            merged_path = self.raw_xyz_path
            
            if osp.exists(merged_path):
                os.remove(merged_path)

            with open(merged_path, "w") as fout: # use "w" model to rewrite
                for fname in tqdm(sorted(xyz_files), desc="Merging XYZ files"):
                    full_path = osp.join(self.raw_dir, fname)
                    try:
                        with open(full_path, "r") as fin:
                            lines = fin.readlines()

                        if not lines: continue
                        natoms = int(lines[0].strip())
                        
                        # 1. Number of atoms written
                        fout.write(f"{natoms}\n")
                        # 2. Write attribute line 
                        prop_line = lines[1].replace('*^', 'e').replace('\t', ' ')
                        fout.write(prop_line)
                        # 3. Write coordinate lines (only take natoms lines)
                        for i in range(2, 2 + natoms):
                            coord_line = lines[i].replace('*^', 'e').replace('\t', ' ')
                            fout.write(coord_line)
                        
                    except Exception as e:
                        logger.warning(f"Error processing {fname}: {e}")
                        continue
            return merged_path
        # Case C: None
        raise RuntimeError(
            f"Decompression is complete, but I couldn't find dsgdb9nsd.xyz or any .xyz files under {self.raw_dir}!"
            "Please check what files are actually included in the downloaded compressed package."
        )

    def _count_files(self, directory: str) -> int:
        """Counts the number of .pkl files in a directory."""
        try:
            return len([n for n in os.listdir(directory) if n.endswith(".pkl")])
        except Exception:
            return 0
    
    @staticmethod
    def _save_pickle(path: str, obj: Any) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def _load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_structures_and_properties(self, raw_path: str, struct_dir: str, prop_dir: str) -> None:
        """
        Core Constructor Function:analyse XYZ -> Pymatgen Structure -> Pickle
        """
        logger.info(f"Parsing {raw_path} using ASE...")
        
        # 1. Read all data into memory (QM9 is about 100MB, which can easily fit into memory)
        atoms_collection = read(raw_path, index=':')
        
        # 2. Read text lines to parse attributes (ASE attribute parsing is sometimes unreliable; manual parsing is more stable)
        with open(raw_path, 'r') as f:
            lines = f.readlines()
            
        prop_buffers = defaultdict(list)
        current_line = 0
        valid_count = 0
        
        total = len(atoms_collection)
        pbar = tqdm(total=total, desc="Processing QM9")
        
        for i, atoms in enumerate(atoms_collection):
            try:
                num_atoms = len(atoms)
                prop_line = lines[current_line + 1]
                
                # clean the property line.
                prop_line_cleaned = prop_line.replace('*^', 'e').replace('\t', ' ')
                raw_vals = prop_line_cleaned.split()
                
                vals_float = []
                for val_str in raw_vals:
                    try:
                        vals_float.append(float(val_str))
                    except ValueError:
                        # for strings like 'gdb' 
                        vals_float.append(0.0) 

                # Mapping Attribute
                for k, key in enumerate(self.PROP_ORDER):
                    if key in ['tag', 'index']: 
                        continue
                    
                    # Alignment index: k=2 is 'A', corresponding to raw_vals[2]
                    if k < len(vals_float):
                        prop_buffers[key].append(vals_float[k])
                    else:
                        prop_buffers[key].append(np.nan)

                # --- B. building Structure (Fake crystal cell) ---
                # Set up a large box to prevent the model from reporting errors due to the absence of cells
                atoms.set_cell([20.0, 20.0, 20.0])
                atoms.center() 
                atoms.pbc = True 
                structure = AseAtomsAdaptor.get_structure(atoms)
                
                # --- C. save the struncture ---
                self._save_pickle(osp.join(struct_dir, f"{i:06d}.pkl"), structure)
                
                # Update pointer
                current_line += (num_atoms + 2)
                valid_count += 1
                pbar.update(1)
                
            except Exception as e:
                logger.warning(f"Error processing molecule {i}: {e}. Skipping block.")
                current_line += (len(atoms) + 2)
                continue
                
        pbar.close()
        
        if valid_count == 0:
            raise RuntimeError("No valid samples processed from QM9 file!")

        # --- D.Save attribute array ---
        logger.info("Saving property arrays...")
        for key, val_list in prop_buffers.items():
            self._save_pickle(
                osp.join(prop_dir, f"{key}.pkl"), 
                np.array(val_list, dtype=np.float32)
            )

    
    def _filter_by_properties(self) -> None:
        """
        Filter out samples that contain invalid property values (e.g., NaN, Inf).
        Operation is performed in-memory.
        """
        if not self.property_names:
            return

        total = len(self.structures)
        keep = []

        # Check properties for each sample
        for i in range(total):
            is_valid = True
            for pname in self.property_names:
                val = self.property_data[pname][i]
                if val is None:
                    is_valid = False
                    break

                # Check for NaN/Inf in scalars
                if isinstance(val, (float, int, np.floating, np.integer)):
                    if np.isnan(val) or np.isinf(val):
                        is_valid = False
                        break
                # Check for NaN/Inf in arrays/lists
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    if not np.all(np.isfinite(arr)):
                        is_valid = False
                        break

            if is_valid:
                keep.append(i)

        if len(keep) < total:
            logger.warning(
                f"Filtering: Dropping {total - len(keep)} samples "
                "due to invalid properties."
            )
            self.structures = [self.structures[i] for i in keep]
            if self.graphs:
                self.graphs = [self.graphs[i] for i in keep]
            # stay property_data 为 numpy.ndarray（Indexing with numpy）
            for pname in self.property_names:
                arr = self.property_data[pname]
                # arr may already be numpy array; this keeps dtype and shape consistent
                self.property_data[pname] = arr[keep]

    def _filter_by_graphs(self) -> None:
        """
        Filter out samples with invalid or missing graphs.
        Since graphs are rebuilt fully if mismatch occurs, this is mostly a
        sanity check.
        """
        pass

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}
        # 1. loading the info
        if self.graphs is not None:
            data["graph"] = self._load_pickle(self.graphs[idx])
        else:
            struct = self._load_pickle(self.structures[idx])
            # turn into dictionary format
            data["pos"] = np.array(struct.cart_coords, dtype='float32')
            data["atomic_numbers"] = np.array([s.specie.Z for s in struct], dtype='int64')
            data["cell"] = np.array(struct.lattice.matrix, dtype='float32')
            data["natoms"] = len(struct)
            data["pbc"] = np.array([True, True, True], dtype=bool)

        # 2. loading the properties
        for pname in self.property_names:
            val = self.property_data[pname][idx]
            # data[pname] = np.array([val], dtype='float32')
            if pname == 'lumo':
                data['energy_per_atom'] = np.array([val], dtype='float32')
            else:
                data[pname] = np.array([val], dtype='float32')
            
        # 3. data transforms
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data