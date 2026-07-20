"""Offline normalization utility for the legacy GMTNet dielectric pickle.

The legacy source pickle stores torch tensors and therefore cannot be loaded by
the runtime-only PaddleMaterials dataset stack. This module is intentionally an
offline preparation tool: importing it never imports torch, while legacy
conversion imports torch lazily only after the conversion command is selected.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import hashlib
import importlib.abc
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
EXPECTED_SOURCE_SHA256 = (
    "5a2198f51f4a7f9aa26fa6be60ed65db0ecc0a13646d63399e897b8168dcbb4d"
)
EXPECTED_NORMALIZED_SHA256 = (
    "eb0b9516c937575afe3f20a0f88953724abfcde6de07c4af15248468598b349f"
)
EXPECTED_RECORD_COUNT = 4713
EXPECTED_SPLIT_SEED = 32
EXPECTED_SPLIT_SIZES = {"train": 3770, "val": 471, "test": 472}
TOOL_PATH = "ppmat/datasets/prepare_gmtnet_dataset.py"
REQUIRED_RECORD_KEYS = {
    "data_index",
    "JARVIS_ID",
    "structure",
    "equivalent_atoms",
    "feature_mask",
    "matrix_equal",
    "dielectric",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Legacy GMTNet conversion requires PyTorch only for offline conversion."
        ) from exc
    return torch


def _require_torch_for_split():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GMTNet split generation requires PyTorch only for offline reproducibility."
        ) from exc
    return torch


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_to_builtin(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "Structure dictionary contains a non-serializable object: "
        f"{type(value).__module__}.{type(value).__name__}"
    )


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _audit_object_graph(value: Any) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    disallowed: Counter[str] = Counter()
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, (str, int, float, bool)) or item is None:
            type_counts[_type_name(item)] += 1
            return
        if isinstance(item, np.ndarray):
            type_counts[_type_name(item)] += 1
            if item.dtype == object:
                disallowed["numpy.ndarray[object]"] += 1
            return
        if isinstance(item, np.generic):
            type_counts[_type_name(item)] += 1
            disallowed[_type_name(item)] += 1
            return

        item_id = id(item)
        if item_id in visited:
            return
        visited.add(item_id)

        if isinstance(item, dict):
            type_counts[_type_name(item)] += 1
            for key, child in item.items():
                visit(key)
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            type_counts[_type_name(item)] += 1
            for child in item:
                visit(child)
            return

        type_counts[_type_name(item)] += 1
        disallowed[_type_name(item)] += 1

    visit(value)
    return {
        "type_counts": dict(sorted(type_counts.items())),
        "disallowed_type_counts": dict(sorted(disallowed.items())),
        "disallowed_object_count": int(sum(disallowed.values())),
    }


def _normalize_record(record: dict[str, Any], data_index: int, torch) -> dict[str, Any]:
    required_legacy_keys = {"JARVIS_ID", "p_input", "feature_mask", "matrix_equal", "dielectric"}
    missing = required_legacy_keys.difference(record)
    if missing:
        raise KeyError(f"Record {data_index} is missing legacy keys: {sorted(missing)}")

    p_input = record["p_input"]
    if not isinstance(p_input, dict):
        raise TypeError(f"Record {data_index} p_input must be a dict.")
    if "structure" not in p_input or "equivalent_atoms" not in p_input:
        raise KeyError(f"Record {data_index} p_input is missing structure/equivalent_atoms.")

    structure = p_input["structure"]
    structure_dict = _to_builtin(structure.as_dict())
    equivalent_atoms = np.asarray(p_input["equivalent_atoms"], dtype=np.int32)
    if equivalent_atoms.ndim != 1:
        raise ValueError(f"Record {data_index} equivalent_atoms must be one-dimensional.")
    if equivalent_atoms.shape[0] != len(structure):
        raise ValueError(
            f"Record {data_index} equivalent_atoms length does not match structure atoms."
        )

    feature_mask = record["feature_mask"]
    matrix_equal = record["matrix_equal"]
    if not isinstance(feature_mask, torch.Tensor):
        raise TypeError(f"Record {data_index} feature_mask must be a torch.Tensor.")
    if not isinstance(matrix_equal, torch.Tensor):
        raise TypeError(f"Record {data_index} matrix_equal must be a torch.Tensor.")
    feature_mask = feature_mask.detach().cpu().numpy().astype(np.float32, copy=False)
    matrix_equal = matrix_equal.detach().cpu().numpy().astype(np.bool_, copy=False)
    dielectric = np.asarray(record["dielectric"], dtype=np.float64)

    if feature_mask.shape != (32, 32):
        raise ValueError(f"Record {data_index} feature_mask shape is {feature_mask.shape}.")
    if matrix_equal.shape != (9, 9):
        raise ValueError(f"Record {data_index} matrix_equal shape is {matrix_equal.shape}.")
    if dielectric.shape != (3, 3):
        raise ValueError(f"Record {data_index} dielectric shape is {dielectric.shape}.")

    normalized = {
        "data_index": int(data_index),
        "JARVIS_ID": str(record["JARVIS_ID"]),
        "structure": structure_dict,
        "equivalent_atoms": equivalent_atoms,
        "feature_mask": feature_mask,
        "matrix_equal": matrix_equal,
        "dielectric": dielectric,
    }
    audit = _audit_object_graph(normalized)
    if audit["disallowed_object_count"] != 0:
        raise TypeError(f"Record {data_index} has disallowed normalized objects: {audit}")
    return normalized


def _array_max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.dtype == np.bool_ or right.dtype == np.bool_:
        return 0.0 if np.array_equal(left, right) else float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _summary_add(summary: dict[str, dict[str, Any]], name: str, array: np.ndarray) -> None:
    entry = summary.setdefault(name, {"count": 0, "dtypes": set(), "shapes": set()})
    entry["count"] += 1
    entry["dtypes"].add(str(array.dtype))
    entry["shapes"].add(tuple(array.shape))


def _json_ready_summary(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "count": value["count"],
            "dtypes": sorted(value["dtypes"]),
            "shapes": [list(shape) for shape in sorted(value["shapes"], key=str)],
        }
        for name, value in sorted(summary.items())
    }


def _validate_full_dataset(legacy_records, normalized_data: dict[str, Any]) -> dict[str, Any]:
    from pymatgen.core import Structure

    normalized_records = normalized_data.get("records")
    if normalized_data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Normalized schema_version is invalid.")
    if normalized_data.get("num_records") != EXPECTED_RECORD_COUNT:
        raise ValueError("Normalized num_records is invalid.")
    if not isinstance(normalized_records, list) or len(normalized_records) != EXPECTED_RECORD_COUNT:
        raise ValueError("Normalized records length is invalid.")

    field_pass_counts = Counter()
    field_summary: dict[str, dict[str, Any]] = {}
    max_diffs = {
        "equivalent_atoms": 0.0,
        "feature_mask": 0.0,
        "matrix_equal": 0.0,
        "dielectric": 0.0,
        "lattice": 0.0,
        "frac_coords": 0.0,
    }
    first_failure = None
    structure_verified = 0

    for index, (legacy, normalized) in enumerate(zip(legacy_records, normalized_records)):
        try:
            if set(normalized) != REQUIRED_RECORD_KEYS:
                raise ValueError(f"unexpected normalized keys: {sorted(normalized)}")
            if normalized["data_index"] != index:
                raise ValueError("data_index mismatch")
            field_pass_counts["data_index"] += 1
            if normalized["JARVIS_ID"] != legacy["JARVIS_ID"]:
                raise ValueError("JARVIS_ID mismatch")
            field_pass_counts["JARVIS_ID"] += 1

            expected_arrays = {
                "equivalent_atoms": np.asarray(
                    legacy["p_input"]["equivalent_atoms"], dtype=np.int32
                ),
                "feature_mask": legacy["feature_mask"].detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                "matrix_equal": legacy["matrix_equal"].detach()
                .cpu()
                .numpy()
                .astype(np.bool_, copy=False),
                "dielectric": np.asarray(legacy["dielectric"], dtype=np.float64),
            }
            expected_shapes = {
                "equivalent_atoms": expected_arrays["equivalent_atoms"].shape,
                "feature_mask": (32, 32),
                "matrix_equal": (9, 9),
                "dielectric": (3, 3),
            }
            expected_dtypes = {
                "equivalent_atoms": np.dtype(np.int32),
                "feature_mask": np.dtype(np.float32),
                "matrix_equal": np.dtype(np.bool_),
                "dielectric": np.dtype(np.float64),
            }
            for field, expected in expected_arrays.items():
                actual = normalized[field]
                if not isinstance(actual, np.ndarray):
                    raise TypeError(f"{field} is not numpy.ndarray")
                if actual.dtype != expected_dtypes[field]:
                    raise ValueError(f"{field} dtype mismatch: {actual.dtype}")
                if actual.shape != expected_shapes[field]:
                    raise ValueError(f"{field} shape mismatch: {actual.shape}")
                difference = _array_max_abs_diff(expected, actual)
                max_diffs[field] = max(max_diffs[field], difference)
                if not np.array_equal(expected, actual):
                    raise ValueError(f"{field} values differ")
                _summary_add(field_summary, field, actual)
                field_pass_counts[field] += 1

            restored_structure = Structure.from_dict(normalized["structure"])
            original_structure = legacy["p_input"]["structure"]
            if len(restored_structure) != len(original_structure):
                raise ValueError("structure atom count differs")
            original_elements = [str(site.specie) for site in original_structure]
            restored_elements = [str(site.specie) for site in restored_structure]
            if original_elements != restored_elements:
                raise ValueError("structure element order differs")
            lattice_diff = _array_max_abs_diff(
                np.asarray(original_structure.lattice.matrix),
                np.asarray(restored_structure.lattice.matrix),
            )
            coordinate_diff = _array_max_abs_diff(
                np.asarray(original_structure.frac_coords),
                np.asarray(restored_structure.frac_coords),
            )
            max_diffs["lattice"] = max(max_diffs["lattice"], lattice_diff)
            max_diffs["frac_coords"] = max(max_diffs["frac_coords"], coordinate_diff)
            if lattice_diff != 0.0 or coordinate_diff != 0.0:
                raise ValueError("structure round-trip has nonzero numerical difference")
            structure_verified += 1
            field_pass_counts["structure"] += 1
        except Exception as exc:
            first_failure = {
                "data_index": index,
                "field": str(exc),
                "exception_type": type(exc).__name__,
            }
            break

    if first_failure is not None:
        raise RuntimeError(f"Full dataset validation failed: {first_failure}")
    if structure_verified != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Structure round-trip validation count is incomplete.")
    if any(value != EXPECTED_RECORD_COUNT for value in field_pass_counts.values()):
        raise RuntimeError(f"Full field validation count is incomplete: {field_pass_counts}")
    return {
        "validated_record_count": EXPECTED_RECORD_COUNT,
        "field_pass_counts": dict(sorted(field_pass_counts.items())),
        "field_dtype_summary": _json_ready_summary(field_summary),
        "field_shape_summary": _json_ready_summary(field_summary),
        "structure_roundtrip_verified_count": structure_verified,
        "full_field_equality_verified_count": EXPECTED_RECORD_COUNT,
        "first_failure": None,
        "max_abs_diffs": max_diffs,
    }


def _validate_graph_samples(legacy_records, normalized_data: dict[str, Any]) -> list[dict[str, Any]]:
    from pymatgen.core import Structure

    from ppmat.models.gmtnet.gmtnet_graph_converter import GMTNetGraphConverter

    converter = GMTNetGraphConverter(
        cutoff=4.0,
        max_neighbors=16,
        atom_features="cgcnn",
        use_canonize=True,
        reduce_cell=False,
    )
    expected_samples = {
        747: "JVASP-33315",
        1423: "JVASP-8034",
        1322: "JVASP-34317",
    }
    rows = []
    for data_index, jarvis_id in expected_samples.items():
        legacy = legacy_records[data_index]
        normalized = normalized_data["records"][data_index]
        if legacy["JARVIS_ID"] != jarvis_id or normalized["JARVIS_ID"] != jarvis_id:
            raise ValueError(f"Fixed sample identity mismatch at {data_index}.")
        original_graph = converter(
            legacy["p_input"]["structure"], legacy["p_input"]["equivalent_atoms"]
        )
        restored_graph = converter(
            Structure.from_dict(normalized["structure"]), normalized["equivalent_atoms"]
        )
        original_x = original_graph.x.numpy()
        restored_x = restored_graph.x.numpy()
        original_edge_index = original_graph.edge_index.numpy()
        restored_edge_index = restored_graph.edge_index.numpy()
        original_edge_attr = original_graph.edge_attr.numpy()
        restored_edge_attr = restored_graph.edge_attr.numpy()
        edge_abs_diff = np.abs(original_edge_attr - restored_edge_attr)
        row = {
            "data_index": data_index,
            "JARVIS_ID": jarvis_id,
            "num_nodes": int(original_x.shape[0]),
            "num_edges": int(original_edge_index.shape[1]),
            "x_shape": list(original_x.shape),
            "x_dtype": str(original_x.dtype),
            "edge_index_shape": list(original_edge_index.shape),
            "edge_index_dtype": str(original_edge_index.dtype),
            "edge_attr_shape": list(original_edge_attr.shape),
            "edge_attr_dtype": str(original_edge_attr.dtype),
            "x_equal": bool(np.array_equal(original_x, restored_x)),
            "edge_index_equal": bool(
                np.array_equal(original_edge_index, restored_edge_index)
            ),
            "edge_attr_max_abs_diff": float(edge_abs_diff.max()),
            "edge_attr_mean_abs_diff": float(edge_abs_diff.mean()),
            "has_nan_or_inf": bool(
                not np.isfinite(original_x).all()
                or not np.isfinite(restored_x).all()
                or not np.isfinite(original_edge_attr).all()
                or not np.isfinite(restored_edge_attr).all()
            ),
        }
        if (
            not row["x_equal"]
            or not row["edge_index_equal"]
            or row["edge_attr_max_abs_diff"] != 0.0
            or row["edge_attr_mean_abs_diff"] != 0.0
            or row["has_nan_or_inf"]
        ):
            raise RuntimeError(f"Graph validation failed: {row}")
        rows.append(row)
    return rows


def _atomic_dump_pickle(value: Any, output_path: Path) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_dump_json(value: dict[str, Any], output_path: Path) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _split_indices_sha256(indices: list[int]) -> str:
    encoded = json.dumps(indices, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_split_indices(indices: Any, name: str) -> list[int]:
    expected_size = EXPECTED_SPLIT_SIZES[name]
    if not isinstance(indices, list) or len(indices) != expected_size:
        raise RuntimeError(f"{name}_indices length does not match {expected_size}.")
    if any(type(index) is not int for index in indices):
        raise RuntimeError(f"{name}_indices must contain only JSON integers.")
    if any(index < 0 or index >= EXPECTED_RECORD_COUNT for index in indices):
        raise RuntimeError(f"{name}_indices contains an out-of-range index.")
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"{name}_indices contains duplicate indices.")
    return indices


def _validate_split_json(split_data: Any, normalized_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(split_data, dict):
        raise RuntimeError("Split JSON root must be an object.")
    expected_fields = {
        "schema_version",
        "source_original_dataset_sha256",
        "normalized_schema_version",
        "normalized_dataset_sha256",
        "num_records",
        "seed",
        "generation_method",
        "torch_version",
        "split_sizes",
        "split_indices_sha256",
        "train_indices",
        "val_indices",
        "test_indices",
    }
    if set(split_data) != expected_fields:
        raise RuntimeError("Split JSON fields do not match the required schema.")
    if split_data["schema_version"] != 1:
        raise RuntimeError("Split schema_version is invalid.")
    if split_data["source_original_dataset_sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Split source_original_dataset_sha256 is invalid.")
    if split_data["normalized_schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("Split normalized_schema_version is invalid.")
    if split_data["normalized_dataset_sha256"] != EXPECTED_NORMALIZED_SHA256:
        raise RuntimeError("Split normalized_dataset_sha256 is invalid.")
    if split_data["num_records"] != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Split num_records is invalid.")
    if split_data["seed"] != EXPECTED_SPLIT_SEED:
        raise RuntimeError("Split seed is invalid.")
    if split_data["generation_method"] != "torch.utils.data.random_split":
        raise RuntimeError("Split generation_method is invalid.")
    if not isinstance(split_data["torch_version"], str) or not split_data["torch_version"]:
        raise RuntimeError("Split torch_version is invalid.")
    if split_data["split_sizes"] != EXPECTED_SPLIT_SIZES:
        raise RuntimeError("Split sizes are invalid.")

    split_indices = {
        name: _validate_split_indices(split_data[f"{name}_indices"], name)
        for name in EXPECTED_SPLIT_SIZES
    }
    all_indices = [index for indices in split_indices.values() for index in indices]
    if len(all_indices) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Split index total is invalid.")
    if len(set(all_indices)) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Split indices overlap across partitions.")
    if set(all_indices) != set(range(EXPECTED_RECORD_COUNT)):
        raise RuntimeError("Split indices do not cover the normalized dataset.")
    if split_indices["test"][:3] != [747, 1423, 1322]:
        raise RuntimeError("Split test_indices fixed prefix is invalid.")

    hashes = split_data["split_indices_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(EXPECTED_SPLIT_SIZES):
        raise RuntimeError("Split split_indices_sha256 fields are invalid.")
    for name, indices in split_indices.items():
        if hashes[name] != _split_indices_sha256(indices):
            raise RuntimeError(f"Split {name}_indices SHA256 does not match.")

    records = normalized_data.get("records")
    expected_jarvis_ids = {747: "JVASP-33315", 1423: "JVASP-8034", 1322: "JVASP-34317"}
    for index, expected_jarvis_id in expected_jarvis_ids.items():
        if records[index].get("JARVIS_ID") != expected_jarvis_id:
            raise RuntimeError(f"Normalized fixed record {index} JARVIS_ID is invalid.")
    return {
        "split_indices_sha256": hashes,
        "split_sizes": split_data["split_sizes"],
        "test_indices_prefix": split_indices["test"][:3],
    }


def _load_split_json(split_path: Path, normalized_data: dict[str, Any]) -> dict[str, Any]:
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    return _validate_split_json(split_data, normalized_data)


def _load_split_generation_inputs(
    input_path: Path, manifest_path: Path
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError("Normalized input does not exist.")
    if not manifest_path.is_file():
        raise FileNotFoundError("Normalization manifest does not exist.")
    normalized_sha256 = _sha256(input_path)
    if normalized_sha256 != EXPECTED_NORMALIZED_SHA256:
        raise RuntimeError("Normalized input SHA256 does not match the frozen value.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("normalized_dataset_sha256") != normalized_sha256:
        raise RuntimeError("Manifest normalized_dataset_sha256 does not match input.")
    if manifest.get("source_original_dataset_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Manifest source_original_dataset_sha256 is invalid.")
    if manifest.get("normalized_schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Manifest normalized_schema_version is invalid.")
    if manifest.get("num_records") != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Manifest num_records is invalid.")
    with input_path.open("rb") as handle:
        normalized_data = pickle.load(handle)
    if normalized_data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Normalized input schema_version is invalid.")
    if normalized_data.get("num_records") != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Normalized input num_records is invalid.")
    if normalized_data.get("source_original_dataset_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Normalized input source_original_dataset_sha256 is invalid.")
    records = normalized_data.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Normalized input records are invalid.")
    return normalized_data


def _generate_split(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.split_output).resolve()
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite an existing split JSON.")
    if args.split_seed != EXPECTED_SPLIT_SEED:
        raise ValueError(f"--split-seed must be {EXPECTED_SPLIT_SEED}.")
    split_sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    if split_sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Split sizes must be {EXPECTED_SPLIT_SIZES}.")
    normalized_data = _load_split_generation_inputs(input_path, manifest_path)
    torch = _require_torch_for_split()
    generator = torch.Generator()
    generator.manual_seed(args.split_seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        range(EXPECTED_RECORD_COUNT),
        [args.train_size, args.val_size, args.test_size],
        generator=generator,
    )
    split_indices = {
        "train": list(train_subset.indices),
        "val": list(val_subset.indices),
        "test": list(test_subset.indices),
    }
    split_data = {
        "schema_version": 1,
        "source_original_dataset_sha256": EXPECTED_SOURCE_SHA256,
        "normalized_schema_version": SCHEMA_VERSION,
        "normalized_dataset_sha256": EXPECTED_NORMALIZED_SHA256,
        "num_records": EXPECTED_RECORD_COUNT,
        "seed": args.split_seed,
        "generation_method": "torch.utils.data.random_split",
        "torch_version": str(torch.__version__),
        "split_sizes": split_sizes,
        "split_indices_sha256": {
            name: _split_indices_sha256(indices) for name, indices in split_indices.items()
        },
        "train_indices": split_indices["train"],
        "val_indices": split_indices["val"],
        "test_indices": split_indices["test"],
    }
    _validate_split_json(split_data, normalized_data)
    temporary_output = None
    output_created = False
    try:
        temporary_output = _atomic_dump_json(split_data, output_path)
        _load_split_json(temporary_output, normalized_data)
        os.replace(temporary_output, output_path)
        output_created = True
        temporary_output = None
    except Exception:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if output_created:
            output_path.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "ok", "split_output": str(output_path)}, sort_keys=True))


def _run_no_torch_pickle_load(path: Path) -> subprocess.CompletedProcess:
    script = r'''
import importlib.abc
import pickle
import sys

class TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"blocked torch import: {fullname}", name=fullname)
        return None

if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
    raise RuntimeError("torch was preloaded")
sys.meta_path.insert(0, TorchBlocker())
with open(sys.argv[1], "rb") as handle:
    data = pickle.load(handle)
if data["schema_version"] != 1 or data["num_records"] != 4713 or len(data["records"]) != 4713:
    raise RuntimeError("normalized header validation failed")
if data["records"][747]["JARVIS_ID"] != "JVASP-33315":
    raise RuntimeError("fixed sample identity validation failed")
if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
    raise RuntimeError("torch was imported")
print("NO_TORCH_PICKLE_LOAD_OK")
'''
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _run_no_torch_verify_only(path: Path, manifest_path: Path) -> subprocess.CompletedProcess:
    script = r'''
import importlib.abc
import runpy
import sys

class TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"blocked torch import: {fullname}", name=fullname)
        return None

if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
    raise RuntimeError("torch was preloaded")
sys.meta_path.insert(0, TorchBlocker())
sys.argv = [
    "ppmat.datasets.prepare_gmtnet_dataset",
    "--input",
    sys.argv[1],
    "--manifest",
    sys.argv[2],
    "--verify-only",
]
runpy.run_module("ppmat.datasets.prepare_gmtnet_dataset", run_name="__main__")
'''
    return subprocess.run(
        [sys.executable, "-c", script, str(path), str(manifest_path)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _write_reports(report_dir: Path, full_report: dict[str, Any], graph_rows, object_audit) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "full_field_validation.json").write_text(
        json.dumps(full_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    structure_report = {
        "structure_roundtrip_verified_count": full_report[
            "structure_roundtrip_verified_count"
        ],
        "max_abs_diffs": {
            "lattice": full_report["max_abs_diffs"]["lattice"],
            "frac_coords": full_report["max_abs_diffs"]["frac_coords"],
        },
    }
    (report_dir / "structure_roundtrip_summary.json").write_text(
        json.dumps(structure_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (report_dir / "graph_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(graph_rows[0]))
        writer.writeheader()
        writer.writerows(graph_rows)
    (report_dir / "object_type_audit.json").write_text(
        json.dumps(object_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_normalized(
    input_path: Path, manifest_path: Path, split_path: Path | None = None
) -> dict[str, Any]:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("verify-only must not run with torch imported.")
    with input_path.open("rb") as handle:
        normalized_data = pickle.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    object_audit = _audit_object_graph(normalized_data)
    if object_audit["disallowed_object_count"] != 0:
        raise RuntimeError(f"Normalized object graph is unsafe: {object_audit}")
    normalized_sha256 = _sha256(input_path)
    if manifest.get("normalized_dataset_sha256") != normalized_sha256:
        raise RuntimeError("Manifest normalized_dataset_sha256 does not match input.")
    if manifest.get("normalized_file_size") != input_path.stat().st_size:
        raise RuntimeError("Manifest normalized_file_size does not match input.")
    if normalized_data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Normalized schema_version does not match.")
    if normalized_data.get("num_records") != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Normalized num_records does not match.")
    records = normalized_data.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise RuntimeError("Normalized records are invalid.")
    if records[747].get("JARVIS_ID") != "JVASP-33315":
        raise RuntimeError("Normalized fixed sample identity is invalid.")
    result = {
        "normalized_dataset_sha256": normalized_sha256,
        "normalized_file_size": input_path.stat().st_size,
        "object_audit": object_audit,
    }
    if split_path is not None:
        result["split_validation"] = _load_split_json(split_path, normalized_data)
    return result


def _convert(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else None
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing normalized output or manifest.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Legacy input does not exist: {input_path}")
    if output_path.parent != manifest_path.parent:
        raise ValueError("Output and manifest must be placed in the same directory.")

    torch = _require_torch()
    source_sha256 = _sha256(input_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Legacy source SHA256 mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}."
        )
    source_size = input_path.stat().st_size
    temporary_output = None
    temporary_manifest = None
    output_created = False
    manifest_created = False
    try:
        with input_path.open("rb") as handle:
            legacy_records = pickle.load(handle)
        if not isinstance(legacy_records, list) or len(legacy_records) != EXPECTED_RECORD_COUNT:
            raise RuntimeError("Legacy input record count does not match 4713.")

        records = [_normalize_record(record, index, torch) for index, record in enumerate(legacy_records)]
        normalized_data = {
            "schema_version": SCHEMA_VERSION,
            "source_original_dataset_sha256": source_sha256,
            "num_records": EXPECTED_RECORD_COUNT,
            "records": records,
        }
        object_audit = _audit_object_graph(normalized_data)
        if object_audit["disallowed_object_count"] != 0:
            raise RuntimeError(f"Normalized object graph is unsafe: {object_audit}")

        temporary_output = _atomic_dump_pickle(normalized_data, output_path)
        with temporary_output.open("rb") as handle:
            reloaded_normalized_data = pickle.load(handle)
        full_report = _validate_full_dataset(legacy_records, reloaded_normalized_data)
        graph_rows = _validate_graph_samples(legacy_records, reloaded_normalized_data)
        no_torch_load = _run_no_torch_pickle_load(temporary_output)
        if no_torch_load.returncode != 0:
            raise RuntimeError(
                "No-torch pickle load validation failed:\n"
                f"stdout={no_torch_load.stdout}\nstderr={no_torch_load.stderr}"
            )

        normalized_sha256 = _sha256(temporary_output)
        normalized_size = temporary_output.stat().st_size
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "normalized_schema_version": SCHEMA_VERSION,
            "source_original_dataset_sha256": source_sha256,
            "normalized_dataset_sha256": normalized_sha256,
            "source_file_size": source_size,
            "normalized_file_size": normalized_size,
            "num_records": EXPECTED_RECORD_COUNT,
            "field_names": sorted(REQUIRED_RECORD_KEYS),
            "field_dtype_summary": full_report["field_dtype_summary"],
            "field_shape_summary": full_report["field_shape_summary"],
            "structure_roundtrip_verified_count": full_report[
                "structure_roundtrip_verified_count"
            ],
            "full_field_equality_verified_count": full_report[
                "full_field_equality_verified_count"
            ],
            "graph_validation_samples": graph_rows,
            "graph_validation_passed": True,
            "no_torch_pickle_load_passed": True,
            "no_torch_verify_only_passed": False,
            "disallowed_object_count": object_audit["disallowed_object_count"],
            "conversion_tool_path": TOOL_PATH,
            "conversion_tool_sha256": _sha256(Path(__file__).resolve()),
            "created_at_utc": datetime_module.datetime.now(
                datetime_module.timezone.utc
            ).isoformat(),
        }
        temporary_manifest = _atomic_dump_json(manifest, manifest_path)
        no_torch_verify = _run_no_torch_verify_only(temporary_output, temporary_manifest)
        if no_torch_verify.returncode != 0:
            raise RuntimeError(
                "No-torch verify-only validation failed:\n"
                f"stdout={no_torch_verify.stdout}\nstderr={no_torch_verify.stderr}"
            )
        manifest["no_torch_verify_only_passed"] = True
        temporary_manifest.unlink(missing_ok=True)
        temporary_manifest = _atomic_dump_json(manifest, manifest_path)

        if _sha256(input_path) != source_sha256:
            raise RuntimeError("Legacy source changed during conversion.")
        os.replace(temporary_output, output_path)
        output_created = True
        temporary_output = None
        os.replace(temporary_manifest, manifest_path)
        manifest_created = True
        temporary_manifest = None
        if report_dir is not None:
            _write_reports(report_dir, full_report, graph_rows, object_audit)
            (report_dir / "no_torch_load_validation.log").write_text(
                "pickle.load:\n"
                + no_torch_load.stdout
                + no_torch_load.stderr
                + "\nverify-only:\n"
                + no_torch_verify.stdout
                + no_torch_verify.stderr,
                encoding="utf-8",
            )
        print(json.dumps({"status": "ok", "manifest": manifest}, sort_keys=True))
    except Exception:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        if output_created:
            output_path.unlink(missing_ok=True)
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize the legacy GMTNet dielectric pickle without runtime PyTorch."
    )
    parser.add_argument("--input", required=True, help="Legacy or normalized pickle path.")
    parser.add_argument("--output", help="Normalized pickle output path for conversion.")
    parser.add_argument("--manifest", required=True, help="Normalization manifest JSON path.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify a normalized pickle and manifest without importing PyTorch.",
    )
    mode_group.add_argument(
        "--generate-split-only",
        action="store_true",
        help="Generate the fixed historical GMTNet split JSON.",
    )
    parser.add_argument(
        "--split-output",
        help="Split JSON output path for --generate-split-only.",
    )
    parser.add_argument(
        "--split-json",
        help="Optional split JSON to validate with --verify-only.",
    )
    parser.add_argument("--split-seed", type=int, default=EXPECTED_SPLIT_SEED)
    parser.add_argument("--train-size", type=int, default=EXPECTED_SPLIT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=EXPECTED_SPLIT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=EXPECTED_SPLIT_SIZES["test"])
    parser.add_argument(
        "--report-dir",
        help="Optional directory for conversion validation reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            if args.output is not None or args.split_output is not None:
                parser.error("--output and --split-output are not valid together with --verify-only.")
            result = _verify_normalized(
                Path(args.input).resolve(),
                Path(args.manifest).resolve(),
                Path(args.split_json).resolve() if args.split_json else None,
            )
            print(json.dumps({"status": "ok", "verify_only": result}, sort_keys=True))
        elif args.generate_split_only:
            if args.output is not None or args.report_dir is not None or args.split_json is not None:
                parser.error(
                    "--output, --report-dir, and --split-json are not valid with "
                    "--generate-split-only."
                )
            if args.split_output is None:
                parser.error("--split-output is required with --generate-split-only.")
            _generate_split(args)
        else:
            if args.output is None:
                parser.error("--output is required unless --verify-only is used.")
            if args.split_output is not None or args.split_json is not None:
                parser.error(
                    "--split-output and --split-json require --generate-split-only or "
                    "--verify-only."
                )
            _convert(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
