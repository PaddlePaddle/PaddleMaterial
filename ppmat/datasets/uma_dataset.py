# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import bisect
import glob
import inspect
import json
import os
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import paddle
from paddle.io import Dataset

from ppmat.models import build_graph_converter


def _ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]


def _to_tensor(x, dtype: str | None = None) -> paddle.Tensor:
    if paddle.is_tensor(x):
        return x.astype(dtype) if dtype is not None else x
    t = paddle.to_tensor(x)
    return t.astype(dtype) if dtype is not None else t


def _find_in_atoms(atoms, key: str):
    if key in atoms.info:
        return atoms.info[key]
    if key in atoms.arrays:
        return atoms.arrays[key]
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", None)
    if isinstance(results, dict) and key in results:
        return results[key]
    return None


@dataclass
class _SampleId:
    path: str
    frame_index: int | None = None


def _collect_paths(src: str | list[str], pattern: str) -> list[str]:
    if isinstance(src, str):
        src_list = [src]
    else:
        src_list = list(src)
    paths: list[str] = []
    for item in src_list:
        if os.path.isfile(item):
            paths.append(item)
        elif os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, pattern))))
        else:
            paths.extend(sorted(glob.glob(item)))
    if not paths:
        raise ValueError(f"No input files found from src={src!r}, pattern={pattern!r}")
    return sorted(paths)


def _collect_aselmdb_paths(src: str | list[str], pattern: str) -> list[str]:
    """Collect aselmdb files from file/dir/glob inputs."""
    if isinstance(src, str):
        src_list = [src]
    else:
        src_list = list(src)
    paths: list[str] = []
    for item in src_list:
        if os.path.isfile(item):
            if item.endswith(".aselmdb"):
                paths.append(item)
        elif os.path.isdir(item):
            paths.extend(
                sorted(
                    glob.glob(os.path.join(item, "**", pattern), recursive=True),
                )
            )
        else:
            paths.extend(sorted(glob.glob(item)))
    paths = [p for p in paths if p.endswith(".aselmdb")]
    if not paths:
        raise ValueError(
            f"No *.aselmdb files found from src={src!r}, pattern={pattern!r}"
        )
    return sorted(paths)


def _to_stress_1x9(stress_value: Any, dtype: str = "float32") -> paddle.Tensor:
    arr = np.asarray(stress_value)
    if arr.shape == (3, 3):
        arr = arr.reshape(1, 9)
    elif arr.shape == (6,):
        # ASE voigt ordering: (xx, yy, zz, yz, xz, xy)
        xx, yy, zz, yz, xz, xy = arr.tolist()
        arr = np.array(
            [[xx, xy, xz, xy, yy, yz, xz, yz, zz]],
            dtype=np.float64,
        )
    elif arr.ndim == 1 and arr.size == 9:
        arr = arr.reshape(1, 9)
    elif arr.ndim == 2 and arr.shape == (1, 9):
        pass
    else:
        raise ValueError(f"Unsupported stress shape: {arr.shape}")
    return _to_tensor(arr, dtype=dtype)


def _build_dataset_from_config(dataset_cfg: dict[str, Any]) -> Dataset:
    dataset_cfg = dict(dataset_cfg)
    cls_name = dataset_cfg.pop("__class_name__")
    init_params = dataset_cfg.pop("__init_params__", {})
    return eval(cls_name)(**init_params)


def uma_data_list_to_batch(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not data_list:
        raise ValueError("Cannot collate empty batch.")

    out: dict[str, Any] = {}
    node_keys = ["pos", "atomic_numbers", "fixed", "tags", "forces"]
    graph_keys = ["cell", "pbc", "natoms", "charge", "spin", "energy", "stress"]

    for key in node_keys + graph_keys:
        values = [d[key] for d in data_list if key in d]
        if values:
            out[key] = paddle.concat(values, axis=0)

    batch_idx = []
    sid_list: list[str] = []
    dataset_list: list[str] = []
    dataset_name_list: list[str] = []
    edge_index_list = []
    cell_offsets_list = []
    nedges_list = []
    node_offset = 0

    has_edge = all(("edge_index" in d and "cell_offsets" in d and "nedges" in d) for d in data_list)
    for i, d in enumerate(data_list):
        natoms_i = int(d["natoms"].reshape([-1])[0].item())
        batch_idx.append(paddle.full([natoms_i], i, dtype="int64"))
        sid_list.extend([str(x) for x in _ensure_list(d.get("sid", [f"sample_{i}"]))])
        dataset_list.extend([str(x) for x in _ensure_list(d.get("dataset", ["uma"]))])
        dataset_name_list.extend(
            [str(x) for x in _ensure_list(d.get("dataset_name", d.get("dataset", ["uma"])))]
        )

        if has_edge:
            edge_index_list.append(d["edge_index"] + node_offset)
            cell_offsets_list.append(d["cell_offsets"])
            nedges_list.append(d["nedges"])
        node_offset += natoms_i

    out["batch"] = paddle.concat(batch_idx, axis=0)
    out["sid"] = sid_list
    out["dataset"] = dataset_list
    out["dataset_name"] = dataset_name_list

    if has_edge:
        out["edge_index"] = paddle.concat(edge_index_list, axis=1)
        out["cell_offsets"] = paddle.concat(cell_offsets_list, axis=0)
        out["nedges"] = paddle.concat(nedges_list, axis=0)
    return out


class UMASingleCollator:
    """UMA collator for single-dataset training."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return uma_data_list_to_batch(batch)


class UMASingleDataset(Dataset):
    """ASE-backed single dataset for UMA.

    Each sample is converted to UMA-required fields so that the model can use
    on-the-fly graph generation (`otf_graph=True`) without extra preprocessing.
    """

    def __init__(
        self,
        src: str | list[str],
        pattern: str = "*",
        ase_read_args: dict[str, Any] | None = None,
        dataset_name: str = "uma.single",
        task_name: str | None = None,
        dtype: str = "float32",
        include_energy: bool = True,
        include_forces: bool = True,
        include_stress: bool = False,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        charge_key: str = "charge",
        spin_key: str = "spin",
        default_charge: int = 0,
        default_spin: int = 0,
        force_pbc: bool | None = None,
        build_graph_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.dataset_name = dataset_name
        self.task_name = task_name or dataset_name
        self.include_energy = include_energy
        self.include_forces = include_forces
        self.include_stress = include_stress
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.charge_key = charge_key
        self.spin_key = spin_key
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.force_pbc = force_pbc
        self.graph_converter = (
            build_graph_converter(build_graph_cfg)
            if build_graph_cfg is not None
            else None
        )
        self.ase_read_args = dict(ase_read_args or {})
        self.paths = _collect_paths(src, pattern)
        self.sample_ids = self._build_sample_ids()

    def _build_sample_ids(self) -> list[_SampleId]:
        import ase.io

        index_cfg = self.ase_read_args.get("index", ":")
        read_args_no_index = {k: v for k, v in self.ase_read_args.items() if k != "index"}
        sample_ids: list[_SampleId] = []
        for path in self.paths:
            frames = ase.io.read(path, index=index_cfg, **read_args_no_index)
            if isinstance(frames, list):
                for i in range(len(frames)):
                    sample_ids.append(_SampleId(path=path, frame_index=i))
            else:
                frame_idx = index_cfg if isinstance(index_cfg, int) else None
                sample_ids.append(_SampleId(path=path, frame_index=frame_idx))
        if not sample_ids:
            raise ValueError("No samples found after ASE parsing.")
        return sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _read_atoms(self, sample_id: _SampleId):
        import ase.io

        read_args_no_index = {k: v for k, v in self.ase_read_args.items() if k != "index"}
        idx = sample_id.frame_index if sample_id.frame_index is not None else self.ase_read_args.get("index", 0)
        atoms = ase.io.read(sample_id.path, index=idx, **read_args_no_index)
        if isinstance(atoms, list):
            if len(atoms) != 1:
                raise ValueError(
                    f"Expected one frame when loading sample {sample_id}, got {len(atoms)}."
                )
            atoms = atoms[0]
        return atoms

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_id = self.sample_ids[idx]
        atoms = self._read_atoms(sample_id)

        pos = _to_tensor(atoms.get_positions(), self.dtype)
        atomic_numbers = _to_tensor(atoms.get_atomic_numbers(), "int64")
        natoms = _to_tensor([len(atoms)], "int64")
        cell = _to_tensor(np.asarray(atoms.cell).reshape(1, 3, 3), self.dtype)
        if self.force_pbc is None:
            pbc_arr = np.asarray(atoms.get_pbc(), dtype=np.bool_).reshape(1, 3)
        else:
            pbc_arr = np.asarray([self.force_pbc, self.force_pbc, self.force_pbc], dtype=np.bool_).reshape(1, 3)
        pbc = _to_tensor(pbc_arr, "bool")

        tags_arr = atoms.get_tags() if hasattr(atoms, "get_tags") else np.zeros(len(atoms), dtype=np.int64)
        tags = _to_tensor(tags_arr, "int64")
        fixed = paddle.zeros([len(atoms)], dtype="int64")
        charge = _to_tensor([atoms.info.get(self.charge_key, self.default_charge)], self.dtype)
        spin = _to_tensor([atoms.info.get(self.spin_key, self.default_spin)], self.dtype)

        sample: dict[str, Any] = {
            "pos": pos,
            "atomic_numbers": atomic_numbers,
            "cell": cell,
            "pbc": pbc,
            "natoms": natoms,
            "charge": charge,
            "spin": spin,
            "fixed": fixed,
            "tags": tags,
            "sid": [f"{sample_id.path}#{sample_id.frame_index if sample_id.frame_index is not None else 0}"],
            "dataset": [self.task_name],
            "dataset_name": [self.dataset_name],
        }

        if self.include_energy:
            energy_value = _find_in_atoms(atoms, self.energy_key)
            if energy_value is None:
                try:
                    energy_value = atoms.get_potential_energy(apply_constraint=False)
                except Exception:
                    energy_value = np.nan
            sample["energy"] = _to_tensor(np.asarray(energy_value).reshape(1), self.dtype)

        if self.include_forces:
            forces_value = _find_in_atoms(atoms, self.forces_key)
            if forces_value is None:
                try:
                    forces_value = atoms.get_forces(apply_constraint=False)
                except Exception:
                    forces_value = np.full((len(atoms), 3), np.nan, dtype=np.float32)
            sample["forces"] = _to_tensor(forces_value, self.dtype).reshape([-1, 3])

        if self.include_stress:
            stress_value = _find_in_atoms(atoms, self.stress_key)
            if stress_value is None:
                try:
                    stress_value = atoms.get_stress(voigt=False)
                except Exception:
                    stress_value = np.full((3, 3), np.nan, dtype=np.float32)
            sample["stress"] = _to_stress_1x9(stress_value, dtype=self.dtype)
        if self.graph_converter is not None:
            sample.update(self.graph_converter(sample))
        return sample


class UMAAseDBDataset(Dataset):
    """Read UMA samples directly from ASE DB/ASELMDB files.

    This is the preferred dataset adapter for OMat24-style folders that contain
    `*.aselmdb` plus `metadata.npz`.
    """

    def __init__(
        self,
        src: str | list[str],
        pattern: str = "*.aselmdb",
        connect_args: dict[str, Any] | None = None,
        select_args: dict[str, Any] | None = None,
        dataset_name: str = "omat",
        task_name: str | None = None,
        dtype: str = "float32",
        include_energy: bool = True,
        include_forces: bool = True,
        include_stress: bool = False,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        charge_key: str = "charge",
        spin_key: str = "spin",
        default_charge: int = 0,
        default_spin: int = 0,
        force_pbc: bool | None = None,
        build_graph_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.task_name = task_name or dataset_name
        self.dtype = dtype
        self.include_energy = include_energy
        self.include_forces = include_forces
        self.include_stress = include_stress
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.charge_key = charge_key
        self.spin_key = spin_key
        self.default_charge = default_charge
        self.default_spin = default_spin
        self.force_pbc = force_pbc
        self.graph_converter = (
            build_graph_converter(build_graph_cfg)
            if build_graph_cfg is not None
            else None
        )
        self.connect_args = dict(connect_args or {})
        self.select_args = dict(select_args or {})
        global_offset = int(self.select_args.pop("offset", 0))
        global_limit = self.select_args.pop("limit", None)
        if global_limit is not None:
            global_limit = int(global_limit)

        self.db_paths = _collect_aselmdb_paths(src, pattern)
        self.dbs = [self._connect_db(path, self.connect_args) for path in self.db_paths]
        self.db_row_ids = []
        self._flat_index: list[tuple[int, int]] = []
        remaining_offset = global_offset
        remaining_limit = global_limit
        for db_idx, db in enumerate(self.dbs):
            if remaining_limit is not None and remaining_limit <= 0:
                self.db_row_ids.append([])
                continue

            db_len = self._count_row_ids(db, self.select_args)
            if remaining_offset >= db_len:
                remaining_offset -= db_len
                self.db_row_ids.append([])
                continue

            local_limit = remaining_limit
            row_ids = self._collect_row_ids(
                db,
                self.select_args,
                offset=remaining_offset,
                limit=local_limit,
            )
            remaining_offset = 0
            if remaining_limit is not None:
                remaining_limit -= len(row_ids)
            self.db_row_ids.append(row_ids)
            for row_id in row_ids:
                self._flat_index.append((db_idx, int(row_id)))
        if not self._flat_index:
            raise ValueError("No records found in provided ASELMDB files.")

    @staticmethod
    def _decode_json_value(value: Any) -> Any:
        if isinstance(value, dict):
            if "__ndarray__" in value:
                payload = value["__ndarray__"]
                if isinstance(payload, list) and len(payload) >= 3:
                    shape, dtype, data = payload[0], payload[1], payload[2]
                    arr = np.asarray(data, dtype=dtype)
                    try:
                        arr = arr.reshape(shape)
                    except Exception:
                        pass
                    return arr
            return {k: UMAAseDBDataset._decode_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [UMAAseDBDataset._decode_json_value(v) for v in value]
        return value

    @staticmethod
    def _open_lmdb_env(path: str):
        try:
            import lmdb
        except Exception as e:
            raise ImportError(
                "UMAAseDBDataset LMDB fallback requires `lmdb` package."
            ) from e
        return lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    @staticmethod
    def _connect_db(path: str, connect_args: dict[str, Any]):
        try:
            import ase.db
        except Exception as e:
            raise ImportError(
                "UMAAseDBDataset requires `ase` package. Install with `pip install ase`."
            ) from e

        db_args = dict(connect_args)
        if "aselmdb" in path:
            db_args.setdefault("readonly", True)
            db_args.setdefault("use_lock_file", False)
        connect_sig = inspect.signature(ase.db.connect)
        supported_kwargs = set(connect_sig.parameters.keys())
        if "readonly" in db_args and "readonly" not in supported_kwargs:
            # Older ASE releases do not expose `readonly`.
            # IMPORTANT: do not set `append=False` here. For unknown extensions,
            # ASE will remove the existing file when append=False.
            db_args.pop("readonly", None)
        db_args = {k: v for k, v in db_args.items() if k in supported_kwargs}
        try:
            return ase.db.connect(path, **db_args)
        except Exception as e:
            # `aselmdb` backend may be unavailable in older ASE builds.
            if "aselmdb" in path and "unknown database type" in str(e).lower():
                env = UMAAseDBDataset._open_lmdb_env(path)
                return {"_uma_kind": "lmdb_json", "env": env, "path": path}
            raise

    @staticmethod
    def _count_row_ids(db, select_args: dict[str, Any]) -> int:
        if isinstance(db, dict) and db.get("_uma_kind") == "lmdb_json":
            with db["env"].begin() as txn:
                raw_len = txn.get(b"length")
                if raw_len is not None:
                    try:
                        return int(raw_len.decode("utf-8") if isinstance(raw_len, bytes) else raw_len)
                    except Exception:
                        pass
                count = 0
                cursor = txn.cursor()
                for key, _ in cursor:
                    try:
                        key_text = key.decode("utf-8")
                    except Exception:
                        continue
                    if key_text.isdigit():
                        count += 1
                return count
        if hasattr(db, "ids") and not select_args:
            return len(db.ids)
        if hasattr(db, "count"):
            try:
                return int(db.count(**select_args))
            except Exception:
                pass
        return sum(1 for _ in db.select(**select_args))

    @staticmethod
    def _collect_row_ids(
        db,
        select_args: dict[str, Any],
        offset: int = 0,
        limit: int | None = None,
    ) -> list[int]:
        if isinstance(db, dict) and db.get("_uma_kind") == "lmdb_json":
            ids: list[int] = []
            with db["env"].begin() as txn:
                raw_len = txn.get(b"length")
                if raw_len is not None:
                    try:
                        if isinstance(raw_len, bytes):
                            total = int(raw_len.decode("utf-8"))
                        else:
                            total = int(raw_len)
                        ids = list(range(1, total + 1))
                    except Exception:
                        ids = []
                if not ids:
                    cursor = txn.cursor()
                    for key, _ in cursor:
                        try:
                            key_text = key.decode("utf-8")
                        except Exception:
                            continue
                        if key_text.isdigit():
                            ids.append(int(key_text))
            ids = sorted(ids)
            if limit is not None:
                limit = int(limit)
                ids = ids[offset : offset + limit]
            elif offset > 0:
                ids = ids[offset:]
            return ids

        if hasattr(db, "ids") and not select_args:
            ids = [int(x) for x in db.ids]
            if limit is not None:
                return ids[offset : offset + int(limit)]
            if offset > 0:
                return ids[offset:]
            return ids
        local_select_args = dict(select_args)
        if offset > 0:
            local_select_args["offset"] = offset
        if limit is not None:
            local_select_args["limit"] = int(limit)
        return [int(row.id) for row in db.select(**local_select_args)]

    def __len__(self) -> int:
        return len(self._flat_index)

    def _get_row(self, db_idx: int, row_id: int):
        db = self.dbs[db_idx]
        if isinstance(db, dict) and db.get("_uma_kind") == "lmdb_json":
            key = str(row_id).encode("ascii")
            with db["env"].begin() as txn:
                raw = txn.get(key)
            if raw is None:
                raise KeyError(f"Missing LMDB row id={row_id} in {db.get('path')}")
            try:
                payload = zlib.decompress(raw)
            except Exception:
                payload = raw
            row = json.loads(payload.decode("utf-8"))
            return self._decode_json_value(row)
        if hasattr(db, "_get_row"):
            return db._get_row(row_id)
        return db.get(row_id)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        db_idx, row_id = self._flat_index[idx]
        row = self._get_row(db_idx, row_id)
        if isinstance(row, dict):
            numbers = np.asarray(row.get("numbers"), dtype=np.int64).reshape([-1])
            positions = np.asarray(row.get("positions"), dtype=np.float64).reshape(
                [-1, 3]
            )
            cell_arr = np.asarray(
                row.get("cell", np.eye(3, dtype=np.float64)), dtype=np.float64
            ).reshape([3, 3])
            pbc_source = row.get("pbc", [True, True, True])
            row_info = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
            sid = row_info.get("sid", f"{self.db_paths[db_idx]}#{row_id}")
            tags_arr = np.asarray(
                row.get("tags", np.zeros(len(numbers), dtype=np.int64)),
                dtype=np.int64,
            ).reshape([-1])
            charge_value = row_info.get(self.charge_key, self.default_charge)
            spin_value = row_info.get(self.spin_key, self.default_spin)
            energy_value = row.get(self.energy_key, row_info.get(self.energy_key, None))
            forces_value = row.get(self.forces_key, row_info.get(self.forces_key, None))
            stress_value = row.get(self.stress_key, row_info.get(self.stress_key, None))
        else:
            atoms = row.toatoms()
            row_data = getattr(row, "data", None)
            if isinstance(row_data, dict):
                atoms.info.update(row_data)
            numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            cell_arr = np.asarray(atoms.cell)
            pbc_source = atoms.get_pbc()
            sid = atoms.info.get("sid", f"{self.db_paths[db_idx]}#{row_id}")
            tags_arr = (
                atoms.get_tags()
                if hasattr(atoms, "get_tags")
                else np.zeros(len(atoms), dtype=np.int64)
            )
            charge_value = atoms.info.get(self.charge_key, self.default_charge)
            spin_value = atoms.info.get(self.spin_key, self.default_spin)
            energy_value = _find_in_atoms(atoms, self.energy_key)
            forces_value = _find_in_atoms(atoms, self.forces_key)
            stress_value = _find_in_atoms(atoms, self.stress_key)

        pos = _to_tensor(positions, self.dtype)
        atomic_numbers = _to_tensor(numbers, "int64")
        natoms = _to_tensor([len(numbers)], "int64")
        cell = _to_tensor(np.asarray(cell_arr).reshape(1, 3, 3), self.dtype)
        if self.force_pbc is None:
            pbc_arr = np.asarray(pbc_source, dtype=np.bool_).reshape(1, 3)
        else:
            pbc_arr = np.asarray(
                [self.force_pbc, self.force_pbc, self.force_pbc], dtype=np.bool_
            ).reshape(1, 3)
        pbc = _to_tensor(pbc_arr, "bool")
        tags = _to_tensor(tags_arr, "int64")
        fixed = paddle.zeros([len(numbers)], dtype="int64")
        charge = _to_tensor([charge_value], self.dtype)
        spin = _to_tensor([spin_value], self.dtype)

        sample: dict[str, Any] = {
            "pos": pos,
            "atomic_numbers": atomic_numbers,
            "cell": cell,
            "pbc": pbc,
            "natoms": natoms,
            "charge": charge,
            "spin": spin,
            "fixed": fixed,
            "tags": tags,
            "sid": [str(sid)],
            "dataset": [self.task_name],
            "dataset_name": [self.dataset_name],
        }

        if self.include_energy:
            if energy_value is None:
                energy_value = np.nan
            sample["energy"] = _to_tensor(np.asarray(energy_value).reshape(1), self.dtype)

        if self.include_forces:
            if forces_value is None:
                forces_value = np.full((len(numbers), 3), np.nan, dtype=np.float32)
            sample["forces"] = _to_tensor(forces_value, self.dtype).reshape([-1, 3])

        if self.include_stress:
            if stress_value is None:
                stress_value = np.full((3, 3), np.nan, dtype=np.float32)
            sample["stress"] = _to_stress_1x9(stress_value, dtype=self.dtype)

        if self.graph_converter is not None:
            sample.update(self.graph_converter(sample))
        return sample

    def __del__(self):
        dbs = getattr(self, "dbs", None)
        if not dbs:
            return
        for db in dbs:
            if isinstance(db, dict) and db.get("_uma_kind") == "lmdb_json":
                env = db.get("env")
                if env is not None and hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
                continue
            if hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    pass


class UMAMultiDataset(Dataset):
    """Concatenate UMA datasets while preserving per-sample task/domain names."""

    def __init__(self, datasets: list[dict[str, Any] | Dataset]) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("UMAMultiDataset requires at least one dataset config.")
        self.datasets = [
            _build_dataset_from_config(ds) if isinstance(ds, dict) else ds
            for ds in datasets
        ]
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
        if total == 0:
            raise ValueError("UMAMultiDataset cannot wrap empty datasets.")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        prev_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][idx - prev_size]
