from __future__ import annotations

import json  # noqa
import math
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


class OC20S2EFDataset(Dataset):
    """OC20 S2EF parquet dataset handler.

    This loader mirrors the caching workflow of `JarvisDataset`:
    - Downloads shard files if missing
    - Builds and caches `pymatgen.Structure` objects (one pkl per sample)
    - Optionally builds and caches graphs from structures
    - Caches property arrays (one pkl per property)

    Args:
        path (str): Base directory to store/download parquet shards
        urls (Union[str, List[str]]): One or more parquet shard URLs or local file paths
        property_names (Union[str, List[str]]): Labels to extract, e.g.
        build_graph_cfg (Dict, optional): Graph converter config
        transforms (Callable, optional): Sample transforms
        cache_path (str, optional): Explicit cache root
        overwrite (bool): Force rebuild caches
        filter_unvalid (bool): Filter out samples with invalid labels or graphs
    """

    def __init__(
        self,
        path: str,
        urls: Union[str, List[str]],
        property_names: Union[str, List[str]],
        build_graph_cfg: Optional[Dict] = None,
        transforms: Optional[Any] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = list(property_names) if property_names else []

        if isinstance(urls, str):
            urls = [urls]
        self.urls = list(urls)

        os.makedirs(path, exist_ok=True)
        self.shard_dir = osp.join(path, "oc20_s2ef_shards")
        os.makedirs(self.shard_dir, exist_ok=True)

        # Graph converter/cache naming
        if build_graph_cfg is not None:
            graph_converter_name = build_graph_cfg["__class_name__"]
            cutoff_name = str(
                int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5))
            )
        else:
            graph_converter_name = "none"
            cutoff_name = "none"

        if cache_path is None:
            self.cache_path = osp.join(
                path,
                f"oc20_s2ef_cache_{graph_converter_name}_cutoff_{cutoff_name}",
            )
        else:
            self.cache_path = osp.join(
                cache_path,
                f"oc20_s2ef_cache_{graph_converter_name}_cutoff_{cutoff_name}",
            )
        logger.info(f"Cache path: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.build_graph_cfg = build_graph_cfg

        structures_dir = osp.join(self.cache_path, "structures")
        graphs_dir = osp.join(self.cache_path, "graphs")
        props_dir = osp.join(self.cache_path, "properties")
        os.makedirs(structures_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(props_dir, exist_ok=True)

        # 1) Ensure shards exist locally
        local_shards = self._ensure_shards()

        # 2) Check cache status and build if needed
        num_cached_structs = self._count_files(structures_dir)
        missing_props = [
            p
            for p in self.property_names
            if not osp.exists(osp.join(props_dir, f"{p}.pkl"))
        ]
        must_build_structs = self.overwrite or num_cached_structs == 0
        must_build_props = self.overwrite or len(missing_props) > 0

        if must_build_structs or must_build_props:
            if self.overwrite:
                logger.info("overwrite=True, will rebuild structures and properties...")
            else:
                if must_build_structs:
                    logger.info(
                        f"No cached structures found (expected in {structures_dir}), building..."  # noqa
                    )
                if missing_props:
                    logger.info(
                        f"Missing property caches: {missing_props}, building..."
                    )
            if dist.get_rank() == 0:
                # Clean stale caches to avoid length mismatch
                if must_build_structs:
                    try:
                        for f in os.listdir(structures_dir):
                            if f.endswith(".pkl"):
                                os.remove(osp.join(structures_dir, f))
                    except Exception:
                        pass
                if must_build_props:
                    try:
                        for f in os.listdir(props_dir):
                            if f.endswith(".pkl"):
                                os.remove(osp.join(props_dir, f))
                    except Exception:
                        pass
                self._build_structures_and_properties(
                    local_shards, structures_dir, props_dir
                )
            if dist.is_initialized():
                dist.barrier()
        else:
            logger.info(
                f"Using cached structures ({num_cached_structs} files) and properties from {self.cache_path}"  # noqa
            )

        # 3) Optionally build graphs
        if build_graph_cfg is not None:
            num_cached_graphs = self._count_files(graphs_dir)
            must_build_graphs = (
                self.overwrite
                or num_cached_graphs == 0
                or num_cached_graphs != num_cached_structs
            )
            if must_build_graphs:
                if self.overwrite:
                    logger.info(
                        "overwrite=True, rebuilding graphs from cached structures..."
                    )
                elif num_cached_graphs == 0:
                    logger.info(
                        f"No cached graphs found (expected in {graphs_dir}), building..."  # noqa
                    )
                else:
                    logger.info(
                        f"Graph count mismatch (graphs: {num_cached_graphs}, structures: {num_cached_structs}), rebuilding..."  # noqa
                    )
                if dist.get_rank() == 0:
                    # Clean stale graphs first
                    try:
                        for f in os.listdir(graphs_dir):
                            if f.endswith(".pkl"):
                                os.remove(osp.join(graphs_dir, f))
                    except Exception:
                        pass
                    converter = build_graph_converter(build_graph_cfg)
                    self._build_graphs(converter, structures_dir, graphs_dir)
                if dist.is_initialized():
                    dist.barrier()
            else:
                logger.info(
                    f"Using cached graphs ({num_cached_graphs} files) from {graphs_dir}"
                )

        # 4) Load final indices/paths and properties
        self.structures = [
            osp.join(structures_dir, f)
            for f in sorted(os.listdir(structures_dir))
            if f.endswith(".pkl")
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graphs_dir, f)
                for f in sorted(os.listdir(graphs_dir))
                if f.endswith(".pkl")
            ]
        else:
            self.graphs = None

        self.property_data = {
            pname: self._load_pickle(osp.join(props_dir, f"{pname}.pkl"))
            for pname in self.property_names
        }

        if filter_unvalid:
            self._filter_by_properties()
            if len(self.structures) == 0:
                raise RuntimeError(
                    "所有样本在属性筛选后被剔除。\n"
                    "可能原因：\n"
                    "  - 标签列缺失或全为 None/NaN（请查看上方构建日志）\n"
                    "  - 选择了错误的任务/分片或列名别名不匹配\n"
                    "排查建议：\n"
                    "  1) 检查配置的 label_names 与分片列是否一致\n"
                    "  2) 暂时将 filter_unvalid 设为 False 以查看原始样本计数\n"
                    "  3) 仅保留 1 个分片进行快速验证\n"
                )
        if self.graphs is not None:
            self._filter_by_graphs()

        # Consistency clamp (defensive): align lengths if any mismatch remains
        min_len = (
            min(
                [len(self.structures)]
                + ([len(self.graphs)] if self.graphs is not None else [])
                + [len(self.property_data[p]) for p in self.property_names]
            )
            if self.property_names
            else len(self.structures)
        )
        if min_len < len(self.structures):
            self.structures = self.structures[:min_len]
        if self.graphs is not None and min_len < len(self.graphs):
            self.graphs = self.graphs[:min_len]
        for p in list(self.property_names):
            if len(self.property_data[p]) != min_len:
                self.property_data[p] = self.property_data[p][:min_len]

        self.num_samples = len(self.structures)
        logger.info(f"Final OC20S2EFDataset samples: {self.num_samples}")

    def _ensure_shards(self) -> List[str]:
        local_files: List[str] = []
        for url in self.urls:
            filename = osp.basename(url)
            local_path = osp.join(self.shard_dir, filename)
            if (url.startswith("http://") or url.startswith("https://")) and (
                self.overwrite or not osp.exists(local_path)
            ):
                tmp = local_path + ".downloading"
                logger.message(f"Downloading shard: {url}")
                try:
                    urllib.request.urlretrieve(url, tmp)
                    os.replace(tmp, local_path)
                finally:
                    if osp.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
            local_files.append(local_path if osp.exists(local_path) else url)
        return local_files

    def _has_any_files(self, directory: str) -> bool:
        try:
            return any(name.endswith(".pkl") for name in os.listdir(directory))
        except Exception:
            return False

    def _count_files(self, directory: str) -> int:
        try:
            return len(
                [name for name in os.listdir(directory) if name.endswith(".pkl")]
            )
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
        # Initialize property stores
        prop_buffers: Dict[str, List[Any]] = {p: [] for p in self.property_names}

        sample_index = 0
        for shard in shard_paths:
            logger.message(f"Reading shard: {shard}")
            pf = pq.ParquetFile(shard)
            schema_names = set(pf.schema.names)

            # pick available column names
            col_map = {
                "atomic_numbers": ["atomic_numbers", "z"],
                "pos": ["pos", "positions"],
                "cell": [
                    "cell",
                    "lattice",
                    "cell_relaxed",
                    "lattice_vectors",
                    "lattice_mat",
                ],
                "energy": ["energy", "y"],
                "reference_energy": ["reference_energy", "ref_energy", "ref_y"],
                "forces": ["forces", "force"],
                "sid": ["sid", "id"],
                "element": ["element", "elements"],
                "num_atoms_col": ["num_atoms", "nat"],
            }
            chosen: Dict[str, Optional[str]] = {}
            for k, cand in col_map.items():
                chosen[k] = next((c for c in cand if c in schema_names), None)

            # Validate label columns exist (geometry will be synthesized if missing)
            missing_labels: List[str] = []
            for pname in self.property_names:
                if pname == "energy" and (
                    chosen["energy"] is None and chosen["reference_energy"] is None
                ):
                    missing_labels.append("energy (aliases: energy/y/reference_energy)")
                if pname == "forces" and chosen["forces"] is None:
                    missing_labels.append("forces (aliases: forces/force)")
            if missing_labels:
                raise RuntimeError(
                    "OC20 分片缺少任务标签列，无法继续构建。\n"
                    f"  任务标签缺失: {missing_labels}\n"
                    f"  实际可用列: {sorted(schema_names)}\n"
                )

            required = [
                c
                for c in [chosen["atomic_numbers"], chosen["pos"], chosen["cell"]]
                if c is not None
            ]
            optional = [
                c
                for c in [
                    chosen["energy"]
                    if chosen["energy"] is not None
                    else chosen["reference_energy"],
                    chosen["forces"],
                    chosen["sid"],
                    chosen["element"],
                    chosen["num_atoms_col"],
                ]
                if c is not None
            ]
            columns = list(dict.fromkeys(required + optional))

            total_rows = pf.metadata.num_rows if pf.metadata is not None else None
            processed_rows = 0
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=columns)
                data = tbl.to_pydict()

                atoms = data.get(chosen["atomic_numbers"]) or []
                positions = data.get(chosen["pos"]) or []
                cell = data.get(chosen["cell"]) or []
                energies = (
                    data.get(chosen["energy"])
                    if chosen["energy"]
                    else data.get(chosen["reference_energy"])
                    if chosen["reference_energy"]
                    else None
                )
                forces = data.get(chosen["forces"]) if chosen["forces"] else None
                ids = data.get(chosen["sid"]) if chosen["sid"] else None

                elements_col = (
                    data.get(chosen["element"]) if chosen["element"] else None
                )
                num_atoms_col = (
                    data.get(chosen["num_atoms_col"])
                    if chosen["num_atoms_col"]
                    else None
                )

                # Determine row count from available columns
                nrows = 0
                for col in [
                    positions,
                    atoms,
                    cell,
                    energies,
                    forces,
                    ids,
                    elements_col,
                    num_atoms_col,
                ]:
                    if isinstance(col, list) and len(col) > nrows:
                        nrows = len(col)

                for i in range(nrows):
                    try:
                        # atomic numbers
                        if atoms and i < len(atoms) and atoms[i] is not None:
                            atomic_numbers = np.asarray(atoms[i], dtype=int)
                        else:
                            if elements_col is not None and i < len(elements_col):
                                el = elements_col[i]
                                if isinstance(el, (list, tuple)):
                                    atomic_numbers = np.asarray(
                                        [int(Element(x).Z) for x in el], dtype=int
                                    )
                                elif isinstance(el, str):
                                    n_guess = (
                                        int(num_atoms_col[i])
                                        if (
                                            num_atoms_col is not None
                                            and i < len(num_atoms_col)
                                        )
                                        else 1
                                    )
                                    atomic_numbers = np.asarray(
                                        [int(Element(el).Z)] * n_guess, dtype=int
                                    )
                                else:
                                    n_guess = (
                                        int(num_atoms_col[i])
                                        if (
                                            num_atoms_col is not None
                                            and i < len(num_atoms_col)
                                        )
                                        else 1
                                    )
                                    atomic_numbers = np.asarray(
                                        [1] * n_guess, dtype=int
                                    )
                            else:
                                n_guess = (
                                    int(num_atoms_col[i])
                                    if (
                                        num_atoms_col is not None
                                        and i < len(num_atoms_col)
                                    )
                                    else 1
                                )
                                atomic_numbers = np.asarray([1] * n_guess, dtype=int)

                        # positions
                        if (
                            positions
                            and i < len(positions)
                            and positions[i] is not None
                        ):
                            pos = np.asarray(positions[i], dtype=float)
                        else:
                            n = int(atomic_numbers.shape[0])
                            spacing = 2.0
                            side = int(np.ceil(n ** (1.0 / 3.0)))
                            coords = []
                            for xi in range(side):
                                for yi in range(side):
                                    for zi in range(side):
                                        coords.append(
                                            [xi * spacing, yi * spacing, zi * spacing]
                                        )
                                        if len(coords) >= n:
                                            break
                                    if len(coords) >= n:
                                        break
                                if len(coords) >= n:
                                    break
                            pos = np.asarray(coords[:n], dtype=float)

                        # lattice / cell
                        if cell and i < len(cell) and cell[i] is not None:
                            cell_mat = np.asarray(cell[i], dtype=float)
                            lattice = Lattice(cell_mat)
                        else:
                            n = int(atomic_numbers.shape[0])
                            spacing = 2.0
                            side = int(np.ceil(n ** (1.0 / 3.0)))
                            a = max(spacing * (side + 1), 10.0)
                            lattice = Lattice.cubic(a)

                        structure = Structure(
                            lattice,
                            atomic_numbers.tolist(),
                            pos,
                            coords_are_cartesian=True,
                            to_unit_cell=True,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to build structure at row {i}: {e}")
                        continue

                    self._save_pickle(
                        osp.join(structures_dir, f"{sample_index:010d}.pkl"), structure
                    )

                    for pname in self.property_names:
                        if pname == "energy":
                            val = None if energies is None else energies[i]
                        elif pname == "forces":
                            val = (
                                None
                                if forces is None
                                else np.asarray(forces[i], dtype=float)
                            )
                        else:
                            col = data.get(pname)
                            val = None if col is None else col[i]
                        prop_buffers[pname].append(val)

                    sample_index += 1
                    processed_rows += 1

                if total_rows is not None:
                    logger.message(
                        f"Shard progress: row_group {rg+1}/{pf.num_row_groups}, processed {processed_rows}/{total_rows} rows"  # noqa
                    )

        # Save properties
        for pname, arr in prop_buffers.items():
            self._save_pickle(osp.join(props_dir, f"{pname}.pkl"), arr)
        logger.info(
            "Saved structures to '%s' and properties to '%s'",
            structures_dir,
            props_dir,
        )

    def _build_graphs(self, converter, structures_dir: str, graphs_dir: str) -> None:
        files = [f for f in sorted(os.listdir(structures_dir)) if f.endswith(".pkl")]
        total = len(files)
        if total == 0:
            return
        # Chunked parallel conversion using converter(list) which leverages p_map
        chunk_size = 1000
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch_files = files[start:end]
            try:
                structures = [
                    self._load_pickle(osp.join(structures_dir, f)) for f in batch_files
                ]
                graphs = converter(structures)
                for f, g in zip(batch_files, graphs):
                    try:
                        self._save_pickle(osp.join(graphs_dir, f), g)
                    except Exception as ie:
                        logger.warning(f"Failed to save graph for {f}: {ie}")
                logger.message(
                    f"Graph build progress: saved {end}/{total} graphs (chunk {start//chunk_size+1})"  # noqa
                )
            except Exception as e:
                logger.warning(f"Failed to build graphs for chunk {start}:{end}: {e}")

    def _filter_by_properties(self) -> None:
        if not self.property_names:
            return
        # Guard: operate only on the overlapping prefix to avoid IndexError
        candidate_len = min(
            [len(self.structures)]
            + [len(self.property_data[p]) for p in self.property_names]
        )
        keep: List[int] = []
        for i in range(candidate_len):
            ok = True
            for pname in self.property_names:
                v = self.property_data[pname][i]
                if v is None:
                    ok = False
                    break
                if isinstance(v, (float, int)):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        ok = False
                        break
                else:
                    arr = np.asarray(v)
                    if not np.all(np.isfinite(arr)):
                        ok = False
                        break
            if ok:
                keep.append(i)

        # Apply keep mask to all holders
        if keep:
            self.structures = [self.structures[i] for i in keep]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in keep]
            for pname in list(self.property_names):
                self.property_data[pname] = [self.property_data[pname][i] for i in keep]
        else:
            # If nothing kept, clear all to a consistent empty dataset
            self.structures = []
            self.graphs = [] if self.graphs is not None else None
            for pname in list(self.property_names):
                self.property_data[pname] = []
        logger.warning(
            f"Remaining {len(self.structures)} samples after property filtering."
        )

    def _filter_by_graphs(self) -> None:
        keep: List[int] = []
        for i, gpath in enumerate(self.graphs):
            try:
                g = self._load_pickle(gpath)
                if g is not None:
                    keep.append(i)
            except Exception:
                continue
        if len(keep) != len(self.structures):
            self.structures = [self.structures[i] for i in keep]
            self.graphs = [self.graphs[i] for i in keep]
            for pname in list(self.property_data.keys()):
                self.property_data[pname] = [self.property_data[pname][i] for i in keep]
        logger.warning(
            f"Remaining {len(self.structures)} samples after graph filtering."
        )

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
            # Reuse structure_array format used in JarvisDataset for no-graph mode
            atom_types = np.array([site.specie.Z for site in structure])
            lattice_parameters = structure.lattice.parameters
            lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
            angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
            lattice = structure.lattice.matrix.astype("float32")
            data["structure_array"] = {
                "frac_coords": structure.frac_coords.astype("float32"),
                "cart_coords": structure.cart_coords.astype("float32"),
                "atom_types": atom_types,
                "lattice": lattice.reshape(1, 3, 3),
                "lengths": lengths,
                "angles": angles,
                "num_atoms": np.array([len(atom_types)]),
            }

        for pname in self.property_names:
            v = self.property_data[pname][idx]
            if pname == "forces":
                data[pname] = np.asarray(v, dtype="float32")
            else:
                data[pname] = np.asarray([v], dtype="float32")

        data["id"] = idx
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self) -> int:
        return self.num_samples
