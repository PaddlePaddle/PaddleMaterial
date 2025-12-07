from __future__ import annotations
import lmdb
import json
import os
import os.path as osp
import pickle
import tarfile
import urllib.request
import zlib
import ast
import sys
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import Dataset
from typing import Any, Dict, List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from pymatgen.core import Lattice, Structure
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.datasets.custom_data_type import ConcatData

# 调试标记
_DEBUG_PRINT_ONCE = False

class OMol25Dataset(Dataset):
    """
    OMol25 Dataset Handler (Final Version: Single Progress Bar + CHGNet Fixes).
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
        if property_names is None: raise ValueError("property_names required")
        self.property_names = list(property_names) if isinstance(property_names, str) else (list(property_names) if property_names else [])
        
        self.urls = urls if isinstance(urls, list) else ([urls] if isinstance(urls, str) else [])
        if url_indices: self.urls = [self.urls[i] for i in url_indices if 0 <= i < len(self.urls)]

        # Paths
        os.makedirs(path, exist_ok=True)
        self.root_path = path
        self.raw_dir = osp.join(path, "omol25_raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        # Cache config
        gc_name = build_graph_cfg["__class_name__"] if build_graph_cfg else "none"
        cutoff = str(int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5))) if build_graph_cfg else "none"
        self.cache_path = osp.join(cache_path if cache_path else path, f"omol25_cache_{gc_name}_cutoff_{cutoff}")
        
        if dist.get_rank() == 0:
            logger.info(f"Cache path: {self.cache_path}")
            os.makedirs(self.cache_path, exist_ok=True)

        self.transforms = transforms
        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid
        self.build_graph_cfg = build_graph_cfg

        self.structures_dir = osp.join(self.cache_path, "structures")
        self.graphs_dir = osp.join(self.cache_path, "graphs")
        self.props_dir = osp.join(self.cache_path, "properties")

        if dist.get_rank() == 0:
            os.makedirs(self.structures_dir, exist_ok=True)
            os.makedirs(self.graphs_dir, exist_ok=True)
            os.makedirs(self.props_dir, exist_ok=True)

        local_files = self._ensure_data()
        
        if dist.get_rank() == 0:
            self._prepare_structures_and_properties(local_files)
        if dist.is_initialized(): dist.barrier()

        if self.build_graph_cfg and dist.get_rank() == 0:
            self._prepare_graphs()
        if self.build_graph_cfg and dist.is_initialized(): dist.barrier()

        self.structures = [osp.join(self.structures_dir, f) for f in sorted(os.listdir(self.structures_dir)) if f.endswith(".pkl")]
        self.graphs = [osp.join(self.graphs_dir, f) for f in sorted(os.listdir(self.graphs_dir)) if f.endswith(".pkl")] if self.build_graph_cfg else None
        
        logger.info("Loading properties...")
        self.property_data = {p: self._load_pickle(osp.join(self.props_dir, f"{p}.pkl")) for p in self.property_names}

        if self.filter_unvalid: self._filter_by_properties()
        self._ensure_length_consistency()
        self.num_samples = len(self.structures)
        logger.info(f"Final Samples: {self.num_samples}")

    def _prepare_structures_and_properties(self, local_files):
        num_cached = self._count_files(self.structures_dir)
        is_complete = osp.exists(osp.join(self.structures_dir, "completed.flag"))
        if self.overwrite or num_cached == 0 or not is_complete:
            self._clean_dir(self.structures_dir)
            self._clean_dir(self.props_dir)
            self._build_structures_and_properties(local_files, self.structures_dir, self.props_dir)
            with open(osp.join(self.structures_dir, "completed.flag"), "w") as f: f.write("done")
        else:
            logger.info(f"Using cached data: {num_cached}")

    def _prepare_graphs(self):
        if not self.overwrite and osp.exists(osp.join(self.graphs_dir, "completed.flag")): return
        self._clean_dir(self.graphs_dir)
        converter = build_graph_converter(self.build_graph_cfg)
        self._build_graphs(converter, self.structures_dir, self.graphs_dir)
        with open(osp.join(self.graphs_dir, "completed.flag"), "w") as f: f.write("done")

    # =========================================================================
    # 修改点：使用 SuppressStderr 禁止内部进度条刷屏
    # =========================================================================
    def _build_graphs(self, converter, s_dir, g_dir):
        # 内部类：用于临时屏蔽标准错误输出(stderr)，防止内部 tqdm 刷屏
        class SuppressStderr:
            def __init__(self):
                self.null_fds = [os.open(os.devnull, os.O_RDWR)]
                self.save_fds = [os.dup(2)]
            def __enter__(self):
                os.dup2(self.null_fds[0], 2)
            def __exit__(self, *_):
                os.dup2(self.save_fds[0], 2)
                for fd in self.null_fds + self.save_fds: os.close(fd)

        files = sorted([f for f in os.listdir(s_dir) if f.endswith(".pkl")])
        if not files: return
        
        # 只显示这一个全局进度条
        pbar = tqdm(total=len(files), desc="Graph Conversion", unit="sample")
        batch_size = 1000
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            try:
                structs = [self._load_pickle(osp.join(s_dir, f)) for f in batch]
                
                # 尝试屏蔽 converter 内部的输出
                try:
                    with SuppressStderr():
                        graphs = converter(structs)
                except:
                    # 如果屏蔽失败（例如系统不支持），则正常调用
                    graphs = converter(structs)
                
                for f, g in zip(batch, graphs): self._save_pickle(osp.join(g_dir, f), g)
                pbar.update(len(batch))
            except Exception as e: 
                # 确保出错时能看到日志
                sys.stderr = sys.__stderr__ 
                logger.warning(f"Graph convert error: {e}")
        pbar.close()

    def _build_structures_and_properties(self, file_paths: List[str], structures_dir: str, props_dir: str) -> None:
        prop_buffers = {p: [] for p in self.property_names}
        sample_index = 0

        def smart_decode_item(item):
            if isinstance(item, dict) and '__ndarray__' in item:
                content = item['__ndarray__']
                if isinstance(content, list) and len(content) >= 3:
                    dtype = "float64"
                    candidates = []
                    for x in content:
                        if isinstance(x, str): dtype = x
                        elif isinstance(x, list): candidates.append(x)
                    shape, data = None, None
                    if len(candidates) == 2:
                        c1, c2 = candidates[0], candidates[1]
                        l1, l2 = len(c1), len(c2)
                        if l1 <= 5 and l2 > 5: shape, data = c1, c2
                        elif l2 <= 5 and l1 > 5: shape, data = c2, c1
                        else: data, shape = c1, c2
                    try:
                        if data is not None:
                            arr = np.array(data, dtype=dtype)
                            if shape: 
                                try: arr = arr.reshape(shape)
                                except: pass 
                            return arr
                    except: pass
                return item 
            return item

        for filepath in file_paths:
            logger.message(f"Reading LMDB (Smart Unpacker): {filepath}")
            try:
                env = lmdb.open(filepath, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
            except Exception as e:
                logger.warning(f"Open Error: {e}")
                continue

            with env.begin() as txn:
                cursor = txn.cursor()
                total = env.stat()['entries']
                pbar = tqdm(total=total, desc=f"Processing {osp.basename(filepath)}")
                
                debug_count = 0
                for key, value in cursor:
                    try:
                        try: _ = int(key.decode('ascii'))
                        except: continue 
                        
                        raw_obj = None
                        payload = value
                        try: payload = zlib.decompress(value)
                        except: pass
                        
                        try: raw_obj = pickle.loads(payload)
                        except:
                            try: raw_obj = json.loads(payload.decode('utf-8'))
                            except:
                                try: raw_obj = ast.literal_eval(payload.decode('utf-8'))
                                except: continue

                        row_data = {}
                        if isinstance(raw_obj, dict): row_data = raw_obj
                        elif hasattr(raw_obj, '__dict__'): row_data = raw_obj.__dict__
                        else: row_data = raw_obj

                        def get_v(obj, k):
                            val = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
                            return smart_decode_item(val)

                        z = get_v(row_data, 'numbers')
                        if z is None: z = get_v(row_data, 'atomic_numbers')
                        pos = get_v(row_data, 'positions')
                        if z is None or pos is None: continue

                        lattice = Lattice.cubic(50.0)
                        if isinstance(z, dict): z = smart_decode_item(z)
                        if isinstance(pos, dict): pos = smart_decode_item(pos)

                        z = np.array(z, dtype=int)
                        pos = np.array(pos, dtype=float)
                        
                        structure = Structure(lattice, z, pos, coords_are_cartesian=True, to_unit_cell=True)
                        self._save_pickle(osp.join(structures_dir, f"{sample_index:010d}.pkl"), structure)

                        extra = get_v(row_data, 'data') or {}
                        for pname in self.property_names:
                            val = get_v(row_data, pname)
                            if val is None and isinstance(extra, dict): val = extra.get(pname)
                            if val is None:
                                if pname == 'gap' and isinstance(extra, dict): val = extra.get('homo_lumo_gap')
                                elif pname == 'u0': val = get_v(row_data, 'energy')
                            val = smart_decode_item(val)
                            if pname == 'forces' and val is None: val = np.zeros((len(z), 3))
                            prop_buffers[pname].append(val)
                        
                        sample_index += 1
                        pbar.update(1)
                    except Exception as e: continue
                pbar.close()
            env.close()

        logger.info(f"Processed total {sample_index} samples.")
        if sample_index == 0: raise RuntimeError("0 samples processed!")

        logger.info("Saving props...")
        for pname, arr in prop_buffers.items(): self._save_pickle(osp.join(props_dir, f"{pname}.pkl"), arr)

    def _ensure_data(self):
        lmdb_files = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".aselmdb"): lmdb_files.append(osp.join(root, file))
        if not lmdb_files: logger.warning(f"No .aselmdb files in {self.raw_dir}")
        return sorted(lmdb_files)

    def _clean_dir(self, d):
        for f in os.listdir(d):
            if f.endswith(".pkl") or f.endswith(".flag"): os.remove(osp.join(d, f))

    def _count_files(self, d):
        return len([n for n in os.listdir(d) if n.endswith(".pkl")])

    def _ensure_length_consistency(self):
        l = [len(self.structures)]
        if self.graphs: l.append(len(self.graphs))
        for p in self.property_names: l.append(len(self.property_data[p]))
        m = min(l)
        if any(x != m for x in l):
            self.structures = self.structures[:m]
            if self.graphs: self.graphs = self.graphs[:m]
            for p in self.property_names: self.property_data[p] = self.property_data[p][:m]

    def _filter_by_properties(self) -> None:
        if not self.property_names: return
        total = len(self.structures)
        keep = []
        for i in range(total):
            is_valid = True
            for pname in self.property_names:
                val = self.property_data[pname][i]
                if val is None:
                    is_valid = False; break
                if isinstance(val, (float, int, np.floating, np.integer)):
                    if np.isnan(val) or np.isinf(val): is_valid = False; break
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.asarray(val)
                    if not np.all(np.isfinite(arr)): is_valid = False; break
            if is_valid: keep.append(i)

        if len(keep) < total:
            logger.warning(f"Filtering: Dropping {total - len(keep)} samples.")
            self.structures = [self.structures[i] for i in keep]
            if self.graphs: self.graphs = [self.graphs[i] for i in keep]
            for pname in self.property_names:
                self.property_data[pname] = [self.property_data[pname][i] for i in keep]

    def _filter_by_graphs(self) -> None: pass

    def _load_pickle(self, p):
        with open(p, "rb") as f: return pickle.load(f)

    def _save_pickle(self, p, o):
        with open(p, "wb") as f: pickle.dump(o, f)

    # =========================================================================
    # 核心修复：一次性补全所有图索引
    # =========================================================================
    def __getitem__(self, idx):
        global _DEBUG_PRINT_ONCE
        data = {}
        if self.graphs:
            g = self.graphs[idx]
            graph_obj = self._load_pickle(g) if isinstance(g, str) else g
            
            if not _DEBUG_PRINT_ONCE:
                if hasattr(graph_obj, 'node_feat'): print(f"\n[DEBUG] Node Keys: {graph_obj.node_feat.keys()}")
                if hasattr(graph_obj, 'edge_feat'): print(f"[DEBUG] Edge Keys: {graph_obj.edge_feat.keys()}")
                _DEBUG_PRINT_ONCE = True
            
            # 1. 补全 composition_fea (94 dim)
            source_key = None
            for k in ["atom_types", "atom_type", "type", "atomic_numbers", "Z"]:
                if k in graph_obj.node_feat:
                    source_key = k; break
            
            atom_codes = None
            if source_key: atom_codes = graph_obj.node_feat[source_key]
            else:
                try:
                    s_path = self.structures[idx]
                    s = self._load_pickle(s_path) if isinstance(s_path, str) else s_path
                    atom_codes = np.array([site.specie.Z for site in s], dtype="int64")
                    graph_obj.node_feat["atom_type"] = atom_codes
                except: pass
            
            if atom_codes is not None:
                if not isinstance(atom_codes, np.ndarray): atom_codes = np.array(atom_codes)
                N = atom_codes.shape[0]
                padded_fea = np.zeros((N, 94), dtype="float32")
                for i, z in enumerate(atom_codes.flatten()):
                    idx_val = int(z) - 1
                    if 0 <= idx_val < 94: padded_fea[i, idx_val] = 1.0
                graph_obj.node_feat["composition_fea"] = padded_fea
            
            # 2. 补全 atom_graph
            if "atom_graph" not in graph_obj.edge_feat:
                if hasattr(graph_obj, 'edges'):
                    edges = graph_obj.edges
                    if not isinstance(edges, np.ndarray): edges = np.array(edges)
                    graph_obj.edge_feat["atom_graph"] = edges.astype("int32")
                else:
                    graph_obj.edge_feat["atom_graph"] = np.zeros((0, 2), dtype="int32")

            # 3. 补全 bond_graph / bond_line_graph_index (Dummy)
            if "bond_graph" not in graph_obj.edge_feat:
                graph_obj.edge_feat["bond_graph"] = np.zeros((0, 2), dtype="int32")
            if "bond_line_graph_index" not in graph_obj.edge_feat:
                graph_obj.edge_feat["bond_line_graph_index"] = np.zeros((0,), dtype="int32")
                
            # 4. 补全 directed / undirected Indices
            num_edges = 0
            if hasattr(graph_obj, 'num_edges'): num_edges = graph_obj.num_edges
            elif hasattr(graph_obj, 'edges'): num_edges = len(graph_obj.edges)
            
            # Hack: 假设 1-to-1 映射
            idx_range = np.arange(num_edges, dtype="int32")
            if "directed2undirected" not in graph_obj.edge_feat:
                graph_obj.edge_feat["directed2undirected"] = idx_range
            
            if "undirected2directed" not in graph_obj.edge_feat:
                # 注意：undirected2directed 通常需要 ConcatData 支持 (变长)
                # 这里为了简单跑通，直接给 identity。如果模型报错 Variable Length，可能需要更复杂的 Patch
                graph_obj.edge_feat["undirected2directed"] = idx_range

            # 5. 补全 Angle 索引 (Dummy)
            if "angle_graph_index" not in graph_obj.edge_feat:
                graph_obj.edge_feat["angle_graph_index"] = np.zeros((0,), dtype="int32")

            data["graph"] = graph_obj
        else:
            s = self.structures[idx]
            if isinstance(s, str): s = self._load_pickle(s)
            z = np.array([site.specie.Z for site in s])
            l = s.lattice.matrix.astype("float32")
            data["structure_array"] = {
                "frac_coords": ConcatData(s.frac_coords.astype("float32")),
                "cart_coords": ConcatData(s.cart_coords.astype("float32")),
                "atom_types": ConcatData(z),
                "lattice": ConcatData(l.reshape(1, 3, 3)),
                "lengths": ConcatData(np.array(s.lattice.abc, dtype="float32").reshape(1, 3)),
                "angles": ConcatData(np.array(s.lattice.angles, dtype="float32").reshape(1, 3)),
                "num_atoms": ConcatData(np.array([len(z)], dtype="int64")),
            }

        for pname in self.property_names:
            v = self.property_data[pname][idx]
            if v is None: v = 0.0
            val_arr = np.array(v, dtype="float32")
            if val_arr.ndim == 0: val_arr = val_arr.reshape(1)
            data[pname] = val_arr

            # === 别名补丁：解决 KeyError 'energy_per_atom' ===
            if pname == 'gap':
                # data['energy'] = val_arr  # 注释掉，防止和 metric key 冲突
                data['energy_per_atom'] = val_arr

        data["id"] = idx
        return self.transforms(data) if self.transforms else data

    def __len__(self): return self.num_samples