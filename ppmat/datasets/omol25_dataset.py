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

import ast
import json
import os
import os.path as osp
import pickle
import sys
import zlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lmdb
import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger


class OMol25Dataset(Dataset):

    """OMol25 Dataset Handler

    This class provides utilities for loading and processing the OMol25
    dataset, which is utilized in molecular property prediction models such as CHGNet
    and MEGNet. The implementation supports efficient loading from LMDB shards,
    automatic graph construction, and smart decompression of molecular data.

    **Dataset Overview**
    - **Source**: Large-scale molecular dataset hosted by PaddleMaterials, containing
      DFT-calculated properties for millions of equilibrium.
    ```
    ┌───────────────────┬──────────────────────────────────┐
    │ Storage Format    │ LMDB (Lightning Memory-Mapped DB)│
    ├───────────────────┼──────────────────────────────────┤
    │ Total Samples     │ ~4,000,000 (Full Train Set)      │
    ├───────────────────┼──────────────────────────────────┤
    │ Data Type         │ Small Organic Molecules          │
    └───────────────────┴──────────────────────────────────┘
    ```
    The dataset can be downloaded from: https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25/train_4M.tar.gz

    **Data Format**
    The dataset is structured as LMDB files where each entry is a Zlib-compressed
    JSON or Pickle object containing structural information and quantum chemical labels:

    | Key / Attribute           | Description                         | Unit          |
    |---------------------------|-------------------------------------|---------------|
    | `atomic_numbers` (Z)      | List of atomic numbers              | -             |
    | `positions`               | 3D Cartesian coordinates            | Å (Angstrom)  |
    | `u0`                      | Internal Energy at 0K               | eV            |
    | `gap`                     | HOMO-LUMO Gap                       | eV            |
    | `homo`                    | Highest Occupied Molecular Orbital  | eV            |
    | `lumo`                    | Lowest Unoccupied Molecular Orbital | eV            |
    | `dipole`                  | Dipole Moment magnitude             | Debye         |
    | `forces`                  | Atomic Forces (Nx3 array)           | eV/Å          |

    **Example Data Object (Decoded):**
    ```json
    {
        "atomic_numbers": [6, 1, 1, 1, 1],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ...],
        "energy": -40.5,
        "data": {
            "homo_lumo_gap": 6.5,
            "dipole_moment": 0.0
        }
    }
    ```

    Args:
        path (str): The root directory to store downloaded shards and cache files.
            If the path does not exist, it will be created.
        urls (Optional[Union[str, List[str]]], optional): List of URLs or a single URL
            to download the LMDB shards from. If None, defaults to the standard
            OMol25 training set URL.
        property_names (Optional[Union[str, List[str]]], optional): List of target
            property names to load (e.g., `["u0"]`, `["gap"]`, `["dipole"]`).
            **This argument is mandatory.**
        url_indices (Optional[List[int]], optional): If provided, selects a specific
            subset of the `urls` list based on indices. Useful for distributed
            training or testing on a subset. Defaults to None.
        build_graph_cfg (Dict, optional): Configuration dictionary for building graphs
            from structures (e.g., cutoff radius). If provided, graphs will be
            constructed and cached. Defaults to None.
        transforms (Optional[Callable], optional): A callable transform function
            to apply to each sample before returning. Defaults to None.
        cache_path (Optional[str], optional): Explicit path for the cache directory.
            If None, a default path is generated under `path` based on the graph
            converter configuration. Defaults to None.
        overwrite (bool, optional): If True, forces the rebuilding of structures,
            properties, and graphs, ignoring existing cache files. Defaults to False.
        filter_unvalid (bool, optional): If True, filters out samples containing
            NaN/Inf values in properties. Defaults to True.
    """

    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25/train_4M.tar.gz"

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
            raise ValueError("property_names required")
        self.property_names = (
            list(property_names)
            if isinstance(property_names, str)
            else (list(property_names) if property_names else [])
        )

        # Handle URLs configuration
        # Priority: Constructor Argument > Class Attribute
        target_urls = urls
        if target_urls is None:
            target_urls = [self.url]

        self.urls = (
            target_urls
            if isinstance(target_urls, list)
            else ([target_urls] if isinstance(target_urls, str) else [])
        )
        if url_indices:
            self.urls = [self.urls[i] for i in url_indices if 0 <= i < len(self.urls)]

        # Configure paths
        os.makedirs(path, exist_ok=True)
        self.root_path = path
        self.raw_dir = osp.join(path, "omol25_raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        # Generate cache directory based on graph converter config
        gc_name = build_graph_cfg["__class_name__"] if build_graph_cfg else "none"
        cutoff = (
            str(int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5)))
            if build_graph_cfg
            else "none"
        )
        self.cache_path = osp.join(
            cache_path if cache_path else path,
            f"omol25_cache_{gc_name}_cutoff_{cutoff}",
        )

        if dist.get_rank() == 0:
            logger.info(f"Cache path: {self.cache_path}")
            os.makedirs(self.cache_path, exist_ok=True)

        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.build_graph_cfg = build_graph_cfg

        # Define sub-directories for cache
        self.structures_dir = osp.join(self.cache_path, "structures")
        self.graphs_dir = osp.join(self.cache_path, "graphs")
        self.props_dir = osp.join(self.cache_path, "properties")

        if dist.get_rank() == 0:
            os.makedirs(self.structures_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)
            os.makedirs(self.props_dir, exist_ok=True)

        # 1. Ensure raw data exists (Download & Extract)
        local_files = self._ensure_data()

        # 2. Build structures and properties
        if dist.get_rank() == 0:
            self._prepare_structures_and_properties(local_files)
        if dist.is_initialized():
            dist.barrier()

        # 3. Build graphs (if configured)
        if self.build_graph_cfg and dist.get_rank() == 0:
            self._prepare_graphs()
        if self.build_graph_cfg and dist.is_initialized():
            dist.barrier()

        # 4. Load file lists into memory
        self.structures = [
            osp.join(self.structures_dir, f)
            for f in sorted(os.listdir(self.structures_dir))
            if f.endswith(".pkl")
        ]
        self.graphs = (
            [
                osp.join(self.graphs_dir, f)
                for f in sorted(os.listdir(self.graphs_dir))
                if f.endswith(".pkl")
            ]
            if self.build_graph_cfg
            else None
        )

        logger.info("Loading properties...")
        self.property_data = {
            p: self._load_pickle(osp.join(self.props_dir, f"{p}.pkl"))
            for p in self.property_names
        }

        # 5. Filter invalid data
        if self.filter_unvalid:
            self._filter_by_properties()
        self._ensure_length_consistency()
        self.num_samples = len(self.structures)
        logger.info(f"Final Samples: {self.num_samples}")

    def _prepare_structures_and_properties(self, local_files):
        num_cached = self._count_files(self.structures_dir)
        is_complete = osp.exists(osp.join(self.structures_dir, "completed.flag"))
        if self.overwrite or num_cached == 0 or not is_complete:
            self._clean_dir(self.structures_dir)
            self._clean_dir(self.props_dir)
            self._build_structures_and_properties(
                local_files, self.structures_dir, self.props_dir
            )
            with open(osp.join(self.structures_dir, "completed.flag"), "w") as f:
                f.write("done")
        else:
            logger.info(f"Using cached data: {num_cached}")

    def _prepare_graphs(self):
        if not self.overwrite and osp.exists(
            osp.join(self.graphs_dir, "completed.flag")
        ):
            return
        self._clean_dir(self.graphs_dir)
        converter = build_graph_converter(self.build_graph_cfg)
        self._build_graphs(converter, self.structures_dir, self.graphs_dir)
        with open(osp.join(self.graphs_dir, "completed.flag"), "w") as f:
            f.write("done")

    def _build_graphs(self, converter, s_dir, g_dir):
        class SuppressStderr:
            def __init__(self):
                self.null_fds = [os.open(os.devnull, os.O_RDWR)]
                self.save_fds = [os.dup(2)]

            def __enter__(self):
                os.dup2(self.null_fds[0], 2)

            def __exit__(self, *_):
                os.dup2(self.save_fds[0], 2)
                for fd in self.null_fds + self.save_fds:
                    os.close(fd)

        files = sorted([f for f in os.listdir(s_dir) if f.endswith(".pkl")])
        if not files:
            return

        # Global progress bar
        pbar = tqdm(total=len(files), desc="Graph Conversion", unit="sample")
        batch_size = 1000

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            try:
                structs = [self._load_pickle(osp.join(s_dir, f)) for f in batch]

                # Attempt to suppress converter output
                try:
                    with SuppressStderr():
                        graphs = converter(structs)
                except Exception:
                    # Fallback if suppression fails
                    graphs = converter(structs)

                for f, g in zip(batch, graphs):
                    self._save_pickle(osp.join(g_dir, f), g)
                pbar.update(len(batch))
            except Exception as e:
                sys.stderr = sys.__stderr__
                logger.warning(f"Graph convert error: {e}")
        pbar.close()

    def _build_structures_and_properties(
        self, file_paths: List[str], structures_dir: str, props_dir: str
    ) -> None:
        prop_buffers = {p: [] for p in self.property_names}
        sample_index = 0

        # Helper function to decode ASE's special dictionary format for numpy arrays
        def smart_decode_item(item):
            if isinstance(item, dict) and "__ndarray__" in item:
                content = item["__ndarray__"]
                if isinstance(content, list) and len(content) >= 3:
                    dtype = "float64"
                    candidates = []
                    for x in content:
                        if isinstance(x, str):
                            dtype = x
                        elif isinstance(x, list):
                            candidates.append(x)
                    shape, data = None, None

                    # Heuristic to identify shape vs data
                    if len(candidates) == 2:
                        c1, c2 = candidates[0], candidates[1]
                        l1, l2 = len(c1), len(c2)
                        # Shape is usually short, Data is usually long
                        if l1 <= 5 and l2 > 5:
                            shape, data = c1, c2
                        elif l2 <= 5 and l1 > 5:
                            shape, data = c2, c1
                        else:
                            data, shape = c1, c2  # Default order
                    try:
                        if data is not None:
                            arr = np.array(data, dtype=dtype)
                            if shape:
                                try:
                                    arr = arr.reshape(shape)
                                except Exception:
                                    pass
                            return arr
                    except Exception:
                        pass
                return item
            return item

        for filepath in file_paths:
            logger.message(f"Reading LMDB (Smart Unpacker): {filepath}")
            try:
                env = lmdb.open(
                    filepath,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            except Exception as e:
                logger.warning(f"Open Error: {e}")
                continue

            with env.begin() as txn:
                cursor = txn.cursor()
                total = env.stat()["entries"]
                pbar = tqdm(total=total, desc=f"Processing {osp.basename(filepath)}")

                for key, value in cursor:
                    try:
                        # Skip non-integer keys (metadata)
                        try:
                            _ = int(key.decode("ascii"))
                        except ValueError:
                            continue

                        # Attempt to decode payload (Zlib -> Pickle -> JSON -> AST)
                        raw_obj = None
                        payload = value
                        try:
                            payload = zlib.decompress(value)
                        except Exception:
                            pass

                        try:
                            raw_obj = pickle.loads(payload)
                        except Exception:
                            try:
                                raw_obj = json.loads(payload.decode("utf-8"))
                            except Exception:
                                try:
                                    raw_obj = ast.literal_eval(payload.decode("utf-8"))
                                except Exception:
                                    continue

                        # Standardize to dictionary
                        row_data = {}
                        if isinstance(raw_obj, dict):
                            row_data = raw_obj
                        elif hasattr(raw_obj, "__dict__"):
                            row_data = raw_obj.__dict__
                        else:
                            row_data = raw_obj

                        def get_v(obj, k):
                            val = (
                                obj.get(k)
                                if isinstance(obj, dict)
                                else getattr(obj, k, None)
                            )
                            return smart_decode_item(val)

                        # Extract atoms and positions
                        z = get_v(row_data, "numbers")
                        if z is None:
                            z = get_v(row_data, "atomic_numbers")
                        pos = get_v(row_data, "positions")
                        if z is None or pos is None:
                            continue

                        # Build Structure
                        lattice = Lattice.cubic(50.0)
                        if isinstance(z, dict):
                            z = smart_decode_item(z)
                        if isinstance(pos, dict):
                            pos = smart_decode_item(pos)

                        z = np.array(z, dtype=int)
                        pos = np.array(pos, dtype=float)

                        structure = Structure(
                            lattice,
                            z,
                            pos,
                            coords_are_cartesian=True,
                            to_unit_cell=True,
                        )
                        self._save_pickle(
                            osp.join(structures_dir, f"{sample_index:010d}.pkl"),
                            structure,
                        )

                        # Extract properties
                        extra = get_v(row_data, "data") or {}
                        for pname in self.property_names:
                            val = get_v(row_data, pname)
                            if val is None and isinstance(extra, dict):
                                val = extra.get(pname)
                            # Property aliases
                            if val is None:
                                if pname == "gap" and isinstance(extra, dict):
                                    val = extra.get("homo_lumo_gap")
                                elif pname == "u0":
                                    val = get_v(row_data, "energy")

                            val = smart_decode_item(val)
                            # Fill missing forces with zeros
                            if pname == "forces" and val is None:
                                val = np.zeros((len(z), 3))
                            prop_buffers[pname].append(val)

                        sample_index += 1
                        pbar.update(1)
                    except Exception:
                        continue
                pbar.close()
            env.close()

        logger.info(f"Processed total {sample_index} samples.")
        if sample_index == 0:
            raise RuntimeError("0 samples processed!")

        logger.info("Saving props...")
        for pname, arr in prop_buffers.items():
            self._save_pickle(osp.join(props_dir, f"{pname}.pkl"), arr)

    def _ensure_data(self):
        lmdb_files = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".aselmdb"):
                    lmdb_files.append(osp.join(root, file))

        # Check if we need to download
        if not lmdb_files:
            logger.warning(f"No .aselmdb files found in {self.raw_dir}")

            if self.urls:
                for url in self.urls:
                    if not url:
                        continue
                    filename = osp.basename(url)
                    local_path = osp.join(self.root_path, filename)

                    if dist.get_rank() == 0:
                        try:
                            # === PROGRESS BAR ADDED HERE ===
                            class TqdmUpTo(tqdm):
                                def update_to(self, b=1, bsize=1, tsize=None):
                                    if tsize is not None:
                                        self.total = tsize
                                    self.update(b * bsize - self.n)

                            logger.message(f"Downloading {url}...")
                            import urllib.request

                            # Using TqdmUpTo as reporthook
                            with TqdmUpTo(
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                                miniters=1,
                                desc=filename,
                            ) as t:
                                urllib.request.urlretrieve(
                                    url, local_path, reporthook=t.update_to
                                )
                            # ==============================

                            if filename.endswith("tar.gz"):
                                logger.info(f"Extracting {filename}...")
                                import tarfile

                                with tarfile.open(local_path, "r:gz") as tar:
                                    tar.extractall(path=self.raw_dir)
                                logger.info("Extraction complete.")

                        except Exception as e:
                            logger.warning(f"Download/Extract failed: {e}")

                    if dist.is_initialized():
                        dist.barrier()

                    # Re-check for files after potential download
                    for root, _, files in os.walk(self.raw_dir):
                        for file in files:
                            if file.endswith(".aselmdb"):
                                lmdb_files.append(osp.join(root, file))

        return sorted(lmdb_files)

    def _clean_dir(self, d):
        for f in os.listdir(d):
            if f.endswith(".pkl") or f.endswith(".flag"):
                os.remove(osp.join(d, f))

    def _count_files(self, d):
        return len([n for n in os.listdir(d) if n.endswith(".pkl")])

    def _ensure_length_consistency(self):
        length_list = [len(self.structures)]
        if self.graphs:
            length_list.append(len(self.graphs))
        for p in self.property_names:
            length_list.append(len(self.property_data[p]))
        m = min(length_list)
        if any(x != m for x in length_list):
            self.structures = self.structures[:m]
            if self.graphs:
                self.graphs = self.graphs[:m]
            for p in self.property_names:
                self.property_data[p] = self.property_data[p][:m]

    def _filter_by_properties(self) -> None:
        if not self.property_names:
            return
        total = len(self.structures)
        keep = []
        for i in range(total):
            is_valid = True
            for pname in self.property_names:
                val = self.property_data[pname][i]
                if val is None:
                    is_valid = False
                    break
                if isinstance(val, (float, int, np.floating, np.integer)):
                    if np.isnan(val) or np.isinf(val):
                        is_valid = False
                        break
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    if not np.all(np.isfinite(arr)):
                        is_valid = False
                        break
            if is_valid:
                keep.append(i)

        if len(keep) < total:
            logger.warning(f"Filtering: Dropping {total - len(keep)} samples.")
            self.structures = [self.structures[i] for i in keep]
            if self.graphs:
                self.graphs = [self.graphs[i] for i in keep]
            for pname in self.property_names:
                self.property_data[pname] = [self.property_data[pname][i] for i in keep]

    def _filter_by_graphs(self) -> None:
        pass

    def _load_pickle(self, p):
        with open(p, "rb") as f:
            return pickle.load(f)

    def _save_pickle(self, p, o):
        with open(p, "wb") as f:
            pickle.dump(o, f)

    def __getitem__(self, idx):
        global _DEBUG_PRINT_ONCE
        data = {}
        if self.graphs:
            g = self.graphs[idx]
            graph_obj = self._load_pickle(g) if isinstance(g, str) else g

            # Debug: Print keys once
            if not _DEBUG_PRINT_ONCE:
                if hasattr(graph_obj, "node_feat"):
                    print(f"\n[DEBUG] Node Keys: {graph_obj.node_feat.keys()}")
                if hasattr(graph_obj, "edge_feat"):
                    print(f"[DEBUG] Edge Keys: {graph_obj.edge_feat.keys()}")
                _DEBUG_PRINT_ONCE = True

            # --- PATCH 1: Composition Feature (Dim 94) ---
            source_key = None
            for k in ["atom_types", "atom_type", "type", "atomic_numbers", "Z"]:
                if k in graph_obj.node_feat:
                    source_key = k
                    break

            atom_codes = None
            if source_key:
                atom_codes = graph_obj.node_feat[source_key]
            else:
                # Fallback to structure file
                try:
                    s_path = self.structures[idx]
                    s = self._load_pickle(s_path) if isinstance(s_path, str) else s_path
                    atom_codes = np.array([site.specie.Z for site in s], dtype="int64")
                    graph_obj.node_feat["atom_type"] = atom_codes
                except Exception:
                    pass

            # Create One-Hot Embedding (94 elements)
            if atom_codes is not None:
                if not isinstance(atom_codes, np.ndarray):
                    atom_codes = np.array(atom_codes)
                N = atom_codes.shape[0]
                padded_fea = np.zeros((N, 94), dtype="float32")
                for i, z in enumerate(atom_codes.flatten()):
                    idx_val = int(z) - 1
                    if 0 <= idx_val < 94:
                        padded_fea[i, idx_val] = 1.0
                graph_obj.node_feat["composition_fea"] = padded_fea

            # --- PATCH 2: Missing 'atom_graph' ---
            if "atom_graph" not in graph_obj.edge_feat:
                if hasattr(graph_obj, "edges"):
                    edges = graph_obj.edges
                    if not isinstance(edges, np.ndarray):
                        edges = np.array(edges)
                    graph_obj.edge_feat["atom_graph"] = edges.astype("int32")
                else:
                    graph_obj.edge_feat["atom_graph"] = np.zeros((0, 2), dtype="int32")

            # --- PATCH 3: Missing 'bond_graph' & Indices ---
            if "bond_graph" not in graph_obj.edge_feat:
                graph_obj.edge_feat["bond_graph"] = np.zeros((0, 2), dtype="int32")
            if "bond_line_graph_index" not in graph_obj.edge_feat:
                graph_obj.edge_feat["bond_line_graph_index"] = np.zeros(
                    (0,), dtype="int32"
                )

            # --- PATCH 4: Missing 'directed2undirected' & 'undirected2directed' ---
            num_edges = 0
            if hasattr(graph_obj, "num_edges"):
                num_edges = graph_obj.num_edges
            elif hasattr(graph_obj, "edges"):
                num_edges = len(graph_obj.edges)

            # Hack: Assume 1-to-1 mapping
            idx_range = np.arange(num_edges, dtype="int32")
            if "directed2undirected" not in graph_obj.edge_feat:
                graph_obj.edge_feat["directed2undirected"] = idx_range

            if "undirected2directed" not in graph_obj.edge_feat:
                graph_obj.edge_feat["undirected2directed"] = idx_range

            # --- PATCH 5: Angle Index (Dummy) ---
            if "angle_graph_index" not in graph_obj.edge_feat:
                graph_obj.edge_feat["angle_graph_index"] = np.zeros((0,), dtype="int32")

            data["graph"] = graph_obj
        else:
            s = self.structures[idx]
            if isinstance(s, str):
                s = self._load_pickle(s)
            z = np.array([site.specie.Z for site in s])
            lattice_matrix = s.lattice.matrix.astype("float32")
            data["structure_array"] = {
                "frac_coords": ConcatData(s.frac_coords.astype("float32")),
                "cart_coords": ConcatData(s.cart_coords.astype("float32")),
                "atom_types": ConcatData(z),
                "lattice": ConcatData(lattice_matrix.reshape(1, 3, 3)),
                "lengths": ConcatData(
                    np.array(s.lattice.abc, dtype="float32").reshape(1, 3)
                ),
                "angles": ConcatData(
                    np.array(s.lattice.angles, dtype="float32").reshape(1, 3)
                ),
                "num_atoms": ConcatData(np.array([len(z)], dtype="int64")),
            }

        for pname in self.property_names:
            v = self.property_data[pname][idx]
            if v is None:
                v = 0.0
            val_arr = np.array(v, dtype="float32")
            if val_arr.ndim == 0:
                val_arr = val_arr.reshape(1)

            # Assign to original key
            data[pname] = val_arr
            # Create alias for CHGNet
            data["energy_per_atom"] = val_arr

        data["id"] = idx
        return self.transforms(data) if self.transforms else data

    def __len__(self):
        return self.num_samples
