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
"""GMTNet dielectric dataset backed by the normalized frozen pickle."""

from __future__ import annotations

import hashlib
import json
import operator
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import paddle
from paddle.io import Dataset
from pymatgen.core import Structure

from ppmat.models.gmtnet.gmtnet_graph_converter import GMTNetGraphConverter

_RECORD_COUNT = 4713
_SPLIT_SIZES = {"train": 3770, "val": 471, "test": 472}
_CONVERTER_DEFAULTS = {
    "cutoff": 4.0,
    "max_neighbors": 16,
    "atom_features": "cgcnn",
    "use_canonize": True,
    "reduce_cell": False,
}


@dataclass(frozen=True)
class _CachedPayload:
    """Validated immutable-by-convention normalized data cache entry."""

    payload: Mapping[str, Any]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_indices_sha256(indices: list[int]) -> str:
    encoded = json.dumps(indices, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _validate_record(record: Any, index: int) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"Record {index} must be a mapping.")
    required_fields = {
        "data_index",
        "JARVIS_ID",
        "structure",
        "equivalent_atoms",
        "feature_mask",
        "matrix_equal",
        "dielectric",
    }
    missing = required_fields - set(record)
    if missing:
        raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")
    if record["data_index"] != index:
        raise ValueError(f"Record {index} data_index does not match its position.")
    if not isinstance(record["JARVIS_ID"], str) or not record["JARVIS_ID"]:
        raise ValueError(f"Record {index} JARVIS_ID must be a non-empty string.")
    if type(record["structure"]) is not dict:
        raise ValueError(f"Record {index} structure must be a plain dict.")
    field_shapes = {
        "equivalent_atoms": (None,),
        "feature_mask": (32, 32),
        "matrix_equal": (9, 9),
        "dielectric": (3, 3),
    }
    for field_name, expected_shape in field_shapes.items():
        value = record[field_name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Record {index} {field_name} must be a numpy array.")
        if expected_shape == (None,):
            if value.ndim != 1:
                raise ValueError(
                    f"Record {index} equivalent_atoms must have shape [N]."
                )
        elif value.shape != expected_shape:
            raise ValueError(
                f"Record {index} {field_name} has shape {value.shape}, expected {expected_shape}."
            )


def _validate_payload(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Normalized pickle root must be a mapping.")
    if payload.get("schema_version") != 1:
        raise ValueError("Normalized pickle schema_version must be 1.")
    if payload.get("num_records") != _RECORD_COUNT:
        raise ValueError(f"Normalized pickle num_records must be {_RECORD_COUNT}.")
    source_hash = payload.get("source_original_dataset_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError("Normalized pickle source_original_dataset_sha256 is missing.")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != _RECORD_COUNT:
        raise ValueError(
            f"Normalized pickle records must contain {_RECORD_COUNT} entries."
        )
    for index, record in enumerate(records):
        _validate_record(record, index)
    return payload


def _validate_split_json(
    split_path: Path,
    normalized_sha256: str,
    normalized_payload: Mapping[str, Any],
    verify_sha256: bool,
) -> dict[str, tuple[int, ...]]:
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    required_fields = {
        "schema_version",
        "normalized_schema_version",
        "source_original_dataset_sha256",
        "normalized_dataset_sha256",
        "num_records",
        "seed",
        "split_sizes",
        "split_indices_sha256",
        "train_indices",
        "val_indices",
        "test_indices",
    }
    if not isinstance(split_data, dict) or not required_fields <= set(split_data):
        raise ValueError("Split JSON has an invalid schema.")
    if split_data["schema_version"] != 1:
        raise ValueError("Split JSON schema_version must be 1.")
    if split_data["normalized_schema_version"] != 1:
        raise ValueError("Split JSON normalized_schema_version must be 1.")
    if split_data["num_records"] != _RECORD_COUNT:
        raise ValueError(f"Split JSON num_records must be {_RECORD_COUNT}.")
    if split_data["seed"] != 32:
        raise ValueError("Split JSON seed must be 32.")
    if split_data["split_sizes"] != _SPLIT_SIZES:
        raise ValueError("Split JSON split_sizes are invalid.")
    if verify_sha256 and split_data["normalized_dataset_sha256"] != normalized_sha256:
        raise ValueError(
            "Split JSON normalized_dataset_sha256 does not match data_path."
        )
    if (
        split_data["source_original_dataset_sha256"]
        != normalized_payload["source_original_dataset_sha256"]
    ):
        raise ValueError(
            "Split JSON source_original_dataset_sha256 does not match normalized data."
        )
    hashes = split_data["split_indices_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(_SPLIT_SIZES):
        raise ValueError("Split JSON split_indices_sha256 is invalid.")
    split_indices: dict[str, tuple[int, ...]] = {}
    all_indices: list[int] = []
    for split_name, expected_size in _SPLIT_SIZES.items():
        indices = split_data[f"{split_name}_indices"]
        if not isinstance(indices, list) or len(indices) != expected_size:
            raise ValueError(f"Split JSON {split_name}_indices length is invalid.")
        if any(type(index) is not int for index in indices):
            raise ValueError(f"Split JSON {split_name}_indices must contain integers.")
        if any(index < 0 or index >= _RECORD_COUNT for index in indices):
            raise ValueError(
                f"Split JSON {split_name}_indices contains an out-of-range index."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"Split JSON {split_name}_indices contains duplicate indices."
            )
        if hashes[split_name] != _split_indices_sha256(indices):
            raise ValueError(f"Split JSON {split_name}_indices SHA256 does not match.")
        split_indices[split_name] = tuple(indices)
        all_indices.extend(indices)
    if len(set(all_indices)) != _RECORD_COUNT or set(all_indices) != set(
        range(_RECORD_COUNT)
    ):
        raise ValueError(
            "Split JSON indices overlap or do not cover the normalized data."
        )
    if split_indices["test"][:3] != (747, 1423, 1322):
        raise ValueError("Split JSON test_indices fixed prefix is invalid.")
    return split_indices


def _validate_smoke_split_json(
    split_path: Path,
    canonical_split_path: Path,
    normalized_sha256: str,
    normalized_payload: Mapping[str, Any],
    verify_sha256: bool,
) -> dict[str, tuple[int, ...]]:
    canonical_indices = _validate_split_json(
        canonical_split_path,
        normalized_sha256,
        normalized_payload,
        verify_sha256,
    )
    with canonical_split_path.open("r", encoding="utf-8") as handle:
        canonical_data = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    if not isinstance(split_data, dict):
        raise ValueError("Smoke split JSON must be a mapping.")
    if split_data.get("smoke_only") is not True:
        raise ValueError("Smoke split JSON smoke_only must be true.")
    if split_data.get("parent_split_sha256") != _sha256(canonical_split_path):
        raise ValueError(
            "Smoke split JSON parent_split_sha256 does not match canonical split."
        )
    for field_name in (
        "schema_version",
        "normalized_schema_version",
        "source_original_dataset_sha256",
        "normalized_dataset_sha256",
        "num_records",
        "seed",
    ):
        if split_data.get(field_name) != canonical_data.get(field_name):
            raise ValueError(
                f"Smoke split JSON {field_name} does not match canonical split."
            )
    if verify_sha256 and split_data["normalized_dataset_sha256"] != normalized_sha256:
        raise ValueError(
            "Smoke split JSON normalized_dataset_sha256 does not match data_path."
        )
    hashes = split_data.get("split_indices_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_SPLIT_SIZES):
        raise ValueError("Smoke split JSON split_indices_sha256 is invalid.")
    split_sizes = split_data.get("split_sizes")
    if not isinstance(split_sizes, dict) or set(split_sizes) != set(_SPLIT_SIZES):
        raise ValueError("Smoke split JSON split_sizes is invalid.")
    smoke_indices: dict[str, tuple[int, ...]] = {}
    all_indices: list[int] = []
    for split_name in _SPLIT_SIZES:
        indices = split_data.get(f"{split_name}_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(
                f"Smoke split JSON {split_name}_indices must be a non-empty list."
            )
        if any(type(index) is not int for index in indices):
            raise ValueError(
                f"Smoke split JSON {split_name}_indices must contain integers."
            )
        if any(index < 0 or index >= _RECORD_COUNT for index in indices):
            raise ValueError(
                f"Smoke split JSON {split_name}_indices contains an out-of-range index."
            )
        if split_sizes[split_name] != len(indices):
            raise ValueError(
                f"Smoke split JSON split_sizes[{split_name}] does not match indices."
            )
        if hashes[split_name] != _split_indices_sha256(indices):
            raise ValueError(
                f"Smoke split JSON {split_name}_indices SHA256 does not match."
            )
        if (
            len(indices) > len(canonical_indices[split_name])
            or tuple(indices) != canonical_indices[split_name][: len(indices)]
        ):
            raise ValueError(
                f"Smoke split JSON {split_name}_indices must be a canonical ordered prefix."
            )
        smoke_indices[split_name] = tuple(indices)
        all_indices.extend(indices)
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Smoke split JSON indices overlap across splits.")
    return smoke_indices


def _converter_params(build_graph_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if build_graph_cfg is None:
        return dict(_CONVERTER_DEFAULTS)
    config = dict(build_graph_cfg)
    if "__class_name__" in config or "__init_params__" in config:
        class_name = config.pop("__class_name__", None)
        init_params = config.pop("__init_params__", None)
        if config or class_name not in {
            "GMTNetGraphConverter",
            "ppmat.models.gmtnet.gmtnet_graph_converter.GMTNetGraphConverter",
        }:
            raise ValueError("build_graph_cfg must describe GMTNetGraphConverter.")
        if not isinstance(init_params, Mapping):
            raise ValueError("build_graph_cfg __init_params__ must be a mapping.")
        config = dict(init_params)
    unsupported = set(config) - set(_CONVERTER_DEFAULTS)
    if unsupported:
        raise ValueError(
            f"build_graph_cfg has unsupported GMTNetGraphConverter parameters: {sorted(unsupported)}"
        )
    return {**_CONVERTER_DEFAULTS, **config}


class GMTNetDielectricDataset(Dataset):
    """Construct frozen GMTNet dielectric samples for one fixed split."""

    _payload_cache: ClassVar[dict[tuple[Path, int, int], _CachedPayload]] = {}
    _pickle_load_count: ClassVar[int] = 0

    def __init__(
        self,
        data_path: str | Path,
        split: str,
        build_graph_cfg: Mapping[str, Any] | None = None,
        split_path: str | Path | None = None,
        verify_sha256: bool = True,
        allow_smoke_split: bool = False,
        canonical_split_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(allow_smoke_split, bool):
            raise TypeError("allow_smoke_split must be a bool.")
        if split not in _SPLIT_SIZES:
            raise ValueError("split must be one of: train, val, test.")
        self.data_path = _require_regular_file(Path(data_path), "data_path")
        resolved_split_path = (
            Path(split_path)
            if split_path is not None
            else Path(__file__)
            .resolve()
            .with_name("gmtnet_dielectric_split_seed32.json")
        )
        self.split_path = _require_regular_file(resolved_split_path, "split_path")
        cache_entry = self._load_payload(self.data_path)
        self._payload = cache_entry.payload
        if allow_smoke_split:
            if canonical_split_path is None:
                raise ValueError(
                    "canonical_split_path is required when allow_smoke_split is true."
                )
            try:
                resolved_canonical_path = _require_regular_file(
                    Path(canonical_split_path), "canonical_split_path"
                )
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(f"canonical_split_path is invalid: {error}") from error
            self._split_indices = _validate_smoke_split_json(
                self.split_path,
                resolved_canonical_path,
                cache_entry.sha256,
                self._payload,
                verify_sha256,
            )[split]
        else:
            with self.split_path.open("r", encoding="utf-8") as handle:
                split_data = json.load(handle)
            if isinstance(split_data, dict) and split_data.get("smoke_only") is True:
                raise ValueError(
                    "Smoke split JSON requires allow_smoke_split to be true."
                )
            self._split_indices = _validate_split_json(
                self.split_path,
                cache_entry.sha256,
                self._payload,
                verify_sha256,
            )[split]
        self.split = split
        self.graph_converter = GMTNetGraphConverter(
            **_converter_params(build_graph_cfg)
        )

    @classmethod
    def _load_payload(cls, data_path: Path) -> _CachedPayload:
        stat = data_path.stat()
        cache_key = (data_path, stat.st_size, stat.st_mtime_ns)
        cached = cls._payload_cache.get(cache_key)
        if cached is not None:
            return cached
        sha256 = _sha256(data_path)
        with data_path.open("rb") as handle:
            payload = _validate_payload(pickle.load(handle))
        cls._pickle_load_count += 1
        cache_entry = _CachedPayload(payload=payload, sha256=sha256)
        cls._payload_cache[cache_key] = cache_entry
        return cache_entry

    @classmethod
    def _clear_cache_for_testing(cls) -> None:
        cls._payload_cache.clear()
        cls._pickle_load_count = 0

    def __len__(self) -> int:
        return len(self._split_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        local_index = operator.index(index)
        if local_index < 0:
            local_index += len(self)
        if local_index < 0 or local_index >= len(self):
            raise IndexError("GMTNetDielectricDataset index out of range.")
        data_index = self._split_indices[local_index]
        record = self._payload["records"][data_index]
        structure = Structure.from_dict(record["structure"])
        graph = self.graph_converter(structure, record["equivalent_atoms"])
        return {
            "graph": graph,
            "feature_mask": paddle.to_tensor(record["feature_mask"], dtype="float32"),
            "matrix_equal": paddle.to_tensor(record["matrix_equal"], dtype="bool"),
            "dielectric": paddle.to_tensor(record["dielectric"], dtype="float32"),
            "id": record["JARVIS_ID"],
            "data_index": record["data_index"],
        }


def gmtnet_dielectric_collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate one GMTNet graph while preserving its single-graph representation."""

    if not batch:
        raise ValueError("GMTNet dielectric collate requires a non-empty batch.")
    if len(batch) != 1:
        raise ValueError("GMTNet dielectric collate currently requires batch_size=1.")
    sample = batch[0]
    return {
        "graph": sample["graph"],
        "feature_mask": sample["feature_mask"].unsqueeze(0),
        "matrix_equal": sample["matrix_equal"].unsqueeze(0),
        "dielectric": sample["dielectric"].unsqueeze(0),
        "id": [sample["id"]],
        "data_index": paddle.to_tensor([sample["data_index"]], dtype="int64"),
    }
