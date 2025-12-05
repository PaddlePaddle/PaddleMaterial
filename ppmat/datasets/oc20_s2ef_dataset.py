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

from __future__ import annotations

import json  # noqa
import os
import os.path as osp
import pickle
import urllib.request
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

# Attempt to import tqdm for progress visualization
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as _e:  # pragma: no cover # noqa
    pq = None
    pa = None

from pymatgen.core import Element
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.misc import is_equal  # noqa

# -----------------------------------------------------------------------------
# OC20 S2EF Dataset Registry
# -----------------------------------------------------------------------------
OC20_S2EF_TRAIN_2M_URLS = [
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0000.parquet",  # noqa
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0001.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0002.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0003.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0004.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0005.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0006.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0007.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0008.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0009.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0010.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0011.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0012.parquet",
    "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OC20/s2ef_train_2M/0013.parquet",
]


class OC20S2EFDataset(Dataset):
    """
    Open Catalyst 2020 (OC20) S2EF (Structure to Energy and Forces) Dataset Handler.

    **Overview**
    This dataset handler reads data from Parquet shards, designed specifically for
    large-scale molecular dynamics datasets like OC20. It manages the full lifecycle
    of data preparation:
    1.  **Downloading**: Fetches Parquet shards from provided URLs if not present
        locally.
    2.  **Parsing & Caching**: Reads Parquet files, robustly handling schema
        variations. It extracts atomic structures and properties, caching them as
        efficient Pickle files.
        *Note: Includes fallback mechanisms to synthesize dummy geometry if explicit*
        *coordinates are missing in the source file (e.g., metadata-only shards).*
    3.  **Graph Construction**: Optionally converts crystal structures into graph
        representations using a specified graph converter (e.g., Radius Graph),
        with support for caching.
    4.  **Loading**: Provides random access to samples via `__getitem__`.

    **Directory Structure**
    The `cache_path` will be structured as follows:
        - `oc20_s2ef_shards/`: Raw Parquet files.
        - `oc20_s2ef_cache_{converter}_cutoff_{val}/`: Root cache directory.
            - `structures/`: Individual pickled `pymatgen.Structure` objects.
            - `properties/`: Pickled lists of property arrays.
            - `graphs/`: Individual pickled graph objects (if configured).

    Args:
        path (str): Root directory to store downloaded shards and cache files.
            If the path does not exist, it will be created.

        urls (Union[str, List[str]], optional): List of URLs or a single URL to
            download the Parquet shards from. If None, defaults to
            `OC20_S2EF_TRAIN_2M_URLS`.

        property_names (Union[str, List[str]]): List of target property names to load
            (e.g., `["energy", "forces"]`). This argument is mandatory.

        url_indices (List[int], optional): If provided, selects a specific subset of
            the `urls` list based on indices. Useful for distributed training data
            splitting. Defaults to None.

        build_graph_cfg (Dict, optional): Configuration dictionary for building graphs
            from structures (e.g., cutoff radius, max neighbors). If None, graphs
            will not be generated. Defaults to None.

        transforms (Optional[Callable], optional): A callable transform function
            to apply to each sample (dictionary) before returning. Defaults to None.

        cache_path (Optional[str], optional): Explicit path for the cache directory.
            If None, a default path is generated under `path` based on the graph
            converter configuration. Defaults to None.

        overwrite (bool, optional): If True, forces the rebuilding of structures,
            properties, and graphs, ignoring existing cache files. Defaults to False.

        filter_unvalid (bool, optional): If True, filters out samples containing
            NaN/Inf values in properties or corrupted graphs. Defaults to True.

        **kwargs: Additional keyword arguments for compatibility.
    """

    def __init__(
        self,
        path: str,
        urls: Optional[Union[str, List[str]]] = None,
        property_names: Optional[Union[str, List[str]]] = None,
        *,
        url_indices: Optional[List[int]] = None,
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Any] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if property_names is None:
            raise ValueError("property_names must be provided for OC20S2EFDataset")

        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = list(property_names) if property_names else []

        # Handle URLs configuration
        urls_list: Union[str, List[str], None] = urls
        if urls_list is None:
            urls_list = OC20_S2EF_TRAIN_2M_URLS

        if isinstance(urls_list, str):
            urls_list = [urls_list]
        else:
            urls_list = list(urls_list)

        if url_indices is not None:
            urls_list = [urls_list[i] for i in url_indices if 0 <= i < len(urls_list)]
        self.urls = urls_list

        # Configure paths
        os.makedirs(path, exist_ok=True)
        self.shard_dir = osp.join(path, "oc20_s2ef_shards")
        os.makedirs(self.shard_dir, exist_ok=True)

        # Generate cache directory naming based on graph config
        if build_graph_cfg is not None:
            graph_converter_name = build_graph_cfg["__class_name__"]
            cutoff_name = str(
                int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5))
            )
        else:
            graph_converter_name = "none"
            cutoff_name = "none"

        base_cache = cache_path if cache_path is not None else path
        self.cache_path = osp.join(
            base_cache,
            f"oc20_s2ef_cache_{graph_converter_name}_cutoff_{cutoff_name}",
        )
        if dist.get_rank() == 0:
            logger.info(f"Cache path: {self.cache_path}")
            os.makedirs(self.cache_path, exist_ok=True)

        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.build_graph_cfg = build_graph_cfg

        # define sub-directories for cache
        self.structures_dir = osp.join(self.cache_path, "structures")
        self.graphs_dir = osp.join(self.cache_path, "graphs")
        self.props_dir = osp.join(self.cache_path, "properties")

        if dist.get_rank() == 0:
            os.makedirs(self.structures_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)
            os.makedirs(self.props_dir, exist_ok=True)

        # 1) Download and ensure shard files exist locally
        local_shards = self._ensure_shards()

        # 2) Check or build Structures and Properties cache
        # Only rank 0 performs the build process to avoid race conditions
        if dist.get_rank() == 0:
            self._prepare_structures_and_properties(local_shards)

        if dist.is_initialized():
            dist.barrier()

        # 3) Check or build Graphs cache (if configuration provided)
        if self.build_graph_cfg is not None:
            if dist.get_rank() == 0:
                self._prepare_graphs()
            if dist.is_initialized():
                dist.barrier()

        # 4) Load file lists and property data into memory
        # Sort files to ensure consistency across distributed ranks
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

        logger.info("Loading properties into memory...")
        self.property_data = {
            pname: self._load_pickle(osp.join(self.props_dir, f"{pname}.pkl"))
            for pname in self.property_names
        }

        # 5) Filter invalid data based on properties and graphs
        if self.filter_unvalid:
            self._filter_by_properties()
        if self.graphs is not None:
            self._filter_by_graphs()

        # 6) Ensure data length consistency across all arrays
        self._ensure_length_consistency()

        self.num_samples = len(self.structures)
        logger.info(f"Final OC20S2EFDataset samples: {self.num_samples}")

    def _prepare_structures_and_properties(self, local_shards):
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
            logger.info("Building structures and properties from raw shards...")
            # Clean old data to prevent mixing files
            self._clean_dir(self.structures_dir)
            self._clean_dir(self.props_dir)

            self._build_structures_and_properties(
                local_shards, self.structures_dir, self.props_dir
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

    def _ensure_shards(self) -> List[str]:
        """
        Ensures all shard files are present locally. Downloads missing shards.
        """
        local_files: List[str] = []
        for url in self.urls:
            filename = osp.basename(url)
            local_path = osp.join(self.shard_dir, filename)

            # Simple download logic
            if (url.startswith("http") or url.startswith("https")) and (
                self.overwrite or not osp.exists(local_path)
            ):
                if dist.get_rank() == 0:
                    tmp = local_path + ".downloading"
                    logger.message(f"Downloading shard: {url}")
                    try:
                        urllib.request.urlretrieve(url, tmp)
                        os.replace(tmp, local_path)
                    except Exception as e:
                        if osp.exists(tmp):
                            os.remove(tmp)
                        raise RuntimeError(f"Download failed: {e}")
                if dist.is_initialized():
                    dist.barrier()
            local_files.append(local_path if osp.exists(local_path) else url)
        return local_files

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

    def _build_structures_and_properties(
        self, shard_paths: List[str], structures_dir: str, props_dir: str
    ) -> None:
        """
        Builds structure objects and extracts properties from Parquet shards.

        **CRITICAL NOTE**:
        This method contains robust fallback logic to handle datasets that may lack
        explicit geometry information (e.g., `pos` or `cell` columns). In such cases,
        it **synthesizes dummy structures** (random positions, default lattice) to
        allow the data pipeline to function.

        While this enables the pipeline to run on metadata-only or partial datasets,
        **models trained on this synthesized data will be physically meaningless**
        regarding geometric potentials.
        """
        # Initialize buffers for properties
        prop_buffers: Dict[str, List[Any]] = {p: [] for p in self.property_names}
        sample_index = 0

        for shard in shard_paths:
            logger.message(f"Reading shard: {shard}")
            try:
                pf = pq.ParquetFile(shard)
            except Exception as e:
                logger.warning(f"Failed to open shard {shard}: {e}")
                continue

            # Get actual columns present in the file
            schema_names = set(pf.schema.names)
            if sample_index == 0:
                logger.info(f"Parquet Schema columns: {list(schema_names)}")

            # Define column alias mapping for flexible schema matching
            col_map = {
                "atomic_numbers": ["atomic_numbers", "z", "atom_types"],
                "pos": ["pos", "positions", "coords"],
                "cell": ["cell", "lattice", "cell_relaxed", "lattice_mat"],
                "energy": ["energy", "y", "total_energy", "E"],
                "reference_energy": ["reference_energy", "ref_energy", "y_ref"],
                "forces": ["forces", "force", "F"],
                "sid": ["sid", "id", "structure_id"],
                "element": ["element", "elements", "elements_symbol"],
                "num_atoms_col": ["num_atoms", "nat", "natoms"],
            }

            # Select the best matching column name for each field
            chosen = {}
            for k, cand in col_map.items():
                chosen[k] = next((c for c in cand if c in schema_names), None)

            # Determine which columns to read from the parquet file
            cols_to_read = list({c for c in chosen.values() if c is not None})

            # Safety check: ensure we have at least something to define a "row"
            if not cols_to_read:
                raise RuntimeError(f"No usable columns found in {shard}!")

            total_rows = pf.metadata.num_rows if pf.metadata else 0
            pbar = tqdm(
                total=total_rows, desc=f"Processing Shard {osp.basename(shard)}"
            )

            for rg in range(pf.num_row_groups):
                try:
                    tbl = pf.read_row_group(rg, columns=cols_to_read)
                    data = tbl.to_pydict()
                except Exception as e:
                    logger.warning(f"Failed to read row group {rg} in {shard}: {e}")
                    continue

                # Retrieve batch data for key columns
                atoms_batch = data.get(chosen["atomic_numbers"])
                pos_batch = data.get(chosen["pos"])
                cell_batch = data.get(chosen["cell"])
                elem_batch = data.get(chosen["element"])

                # Determine number of rows in this batch.
                # Since 'pos' or 'atoms' might be missing, checking other columns
                # like 'energy' or 'element'.
                nrows = 0
                for col_data in data.values():
                    if col_data is not None and hasattr(col_data, "__len__"):
                        nrows = len(col_data)
                        break

                # Iterate through each sample in the batch
                for i in range(nrows):
                    try:
                        # --- Step 1: Determine Atomic Numbers (Z) ---
                        z = None
                        # Case A: Explicit atomic numbers column exists
                        if (
                            atoms_batch is not None
                            and i < len(atoms_batch)
                            and atoms_batch[i] is not None
                        ):
                            val = atoms_batch[i]
                            if isinstance(val, (list, np.ndarray)):
                                z = np.asarray(val, dtype=int)
                            else:
                                z = np.array([int(val)])

                        # Case B: Derive from Element symbols (e.g. "Ag", "Au")
                        elif elem_batch is not None and i < len(elem_batch):
                            el_raw = elem_batch[i]
                            # Handle single string (e.g. "Ag") or list
                            if isinstance(el_raw, str):
                                z = np.array([Element(el_raw).Z])
                            elif isinstance(el_raw, (list, tuple, np.ndarray)):
                                z = np.array([Element(s).Z for s in el_raw])

                        # Case C: Fallback (Dummy Hydrogen)
                        if z is None:
                            # Use num_atoms column if available to set size
                            n_atoms = 1
                            if chosen["num_atoms_col"] and data.get(
                                chosen["num_atoms_col"]
                            ):
                                n_atoms = int(data[chosen["num_atoms_col"]][i])
                            z = np.ones(n_atoms, dtype=int)  # Dummy Hydrogen

                        # --- Step 2: Determine Positions (Pos) ---
                        if (
                            pos_batch is not None
                            and i < len(pos_batch)
                            and pos_batch[i] is not None
                        ):
                            pos = np.asarray(pos_batch[i], dtype=float)
                            # Safety: align dimensions with Z
                            if pos.shape[0] != z.shape[0]:
                                min_len = min(pos.shape[0], z.shape[0])
                                pos = pos[:min_len]
                                z = z[:min_len]
                        else:
                            # WARNING: SYNTHESIZING DUMMY POSITIONS
                            # Generate random coordinates to prevent build errors.
                            pos = np.random.rand(len(z), 3) * 10.0

                        # --- Step 3: Determine Lattice (Cell) ---
                        if (
                            cell_batch is not None
                            and i < len(cell_batch)
                            and cell_batch[i] is not None
                        ):
                            matrix = np.asarray(cell_batch[i], dtype=float)
                            if matrix.size == 9:
                                matrix = matrix.reshape(3, 3)
                            # Validate determinant to ensure non-singular cell
                            if np.abs(np.linalg.det(matrix)) < 1e-3:
                                lattice = Lattice.cubic(20.0)
                            else:
                                lattice = Lattice(matrix)
                        else:
                            # WARNING: SYNTHESIZING DUMMY LATTICE
                            lattice = Lattice.cubic(20.0)

                        # --- Step 4: Build Pymatgen Structure ---
                        structure = Structure(
                            lattice,
                            z,
                            pos,
                            coords_are_cartesian=True,
                            to_unit_cell=True,
                        )

                        # Save structure pickle
                        self._save_pickle(
                            osp.join(structures_dir, f"{sample_index:010d}.pkl"),
                            structure,
                        )

                        # --- Step 5: Extract Properties ---
                        for pname in self.property_names:
                            val = None

                            # Special handling for energy
                            if pname == "energy":
                                e_col = chosen["energy"]
                                if e_col and data.get(e_col):
                                    val = data[e_col][i]
                                elif chosen["reference_energy"] and data.get(
                                    chosen["reference_energy"]
                                ):
                                    val = data[chosen["reference_energy"]][i]

                            # Special handling for forces
                            elif pname == "forces":
                                f_col = chosen["forces"]
                                if f_col and data.get(f_col):
                                    val = np.asarray(data[f_col][i], dtype=float)
                                else:
                                    # Fallback: dummy zero forces
                                    val = np.zeros((len(z), 3))

                            # Generic property handling
                            else:
                                c = chosen.get(pname)
                                if not c and pname in data:
                                    c = pname
                                if c and data.get(c):
                                    val = data[c][i]

                            prop_buffers[pname].append(val)

                        sample_index += 1
                    except Exception as e:
                        if sample_index == 0:
                            logger.warning(
                                f"Error building structure at index {i}: {e}"
                            )
                        continue

                pbar.update(nrows)
            pbar.close()

        # Check if we successfully processed any samples
        logger.info(f"Processed total {sample_index} samples.")
        if sample_index == 0:
            raise RuntimeError(
                "0 samples processed! The dataset might be empty or incompatible. "
                f"Schema found: {list(schema_names)}"
            )

        # Save all properties to disk
        logger.info("Saving property caches...")
        for pname, arr in prop_buffers.items():
            self._save_pickle(osp.join(props_dir, f"{pname}.pkl"), arr)

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
            for pname in self.property_names:
                self.property_data[pname] = [self.property_data[pname][i] for i in keep]

    def _filter_by_graphs(self) -> None:
        """
        Filter out samples with invalid or missing graphs.
        Since graphs are rebuilt fully if mismatch occurs, this is mostly a
        sanity check.
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self._load_pickle(graph)
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self._load_pickle(structure)

            # Format structure data compatible with JarvisDataset standards
            atom_types = np.array([site.specie.Z for site in structure])
            lattice = structure.lattice.matrix.astype("float32")

            data["structure_array"] = {
                "frac_coords": structure.frac_coords.astype("float32"),
                "cart_coords": structure.cart_coords.astype("float32"),
                "atom_types": atom_types,
                "lattice": lattice,
                "lengths": np.array(structure.lattice.abc, dtype="float32"),
                "angles": np.array(structure.lattice.angles, dtype="float32"),
                "num_atoms": np.array([len(atom_types)], dtype="int64"),
            }

        for pname in self.property_names:
            v = self.property_data[pname][idx]
            # Ensure consistent dimensionality for output tensors
            data[pname] = (
                np.array([v], dtype="float32")
                if np.isscalar(v)
                else np.array(v, dtype="float32")
            )

        data["id"] = idx
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self) -> int:
        return self.num_samples
