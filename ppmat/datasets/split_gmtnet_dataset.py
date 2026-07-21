"""Generate or verify the fixed-seed GMTNet dielectric split.

This offline utility only manages the canonical index split. It neither
downloads nor converts datasets, and it never overwrites a split unless
``--force`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = (
    "5a2198f51f4a7f9aa26fa6be60ed65db0ecc0a13646d63399e897b8168dcbb4d"
)
EXPECTED_NORMALIZED_SHA256 = (
    "eb0b9516c937575afe3f20a0f88953724abfcde6de07c4af15248468598b349f"
)
EXPECTED_RECORD_COUNT = 4713
EXPECTED_SPLIT_SEED = 32
EXPECTED_SPLIT_SIZES = {"train": 3770, "val": 471, "test": 472}
_SPLIT_NAMES = tuple(EXPECTED_SPLIT_SIZES)


def _split_indices_sha256(indices: list[int]) -> str:
    """Return the canonical hash for one ordered index list."""
    encoded = json.dumps(indices, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_indices(indices: Any, split_name: str) -> list[int]:
    """Validate one fixed-size partition without changing its order."""
    expected_size = EXPECTED_SPLIT_SIZES[split_name]
    if not isinstance(indices, list) or len(indices) != expected_size:
        raise ValueError(f"{split_name}_indices length must be {expected_size}.")
    if any(type(index) is not int for index in indices):
        raise ValueError(f"{split_name}_indices must contain JSON integers only.")
    if any(index < 0 or index >= EXPECTED_RECORD_COUNT for index in indices):
        raise ValueError(f"{split_name}_indices contains an out-of-range index.")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{split_name}_indices contains duplicate indices.")
    return indices


def validate_split_data(split_data: Any) -> dict[str, list[int]]:
    """Validate the canonical GMTNet split schema and ordered partitions."""
    required_fields = {
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
    if not isinstance(split_data, dict) or set(split_data) != required_fields:
        raise ValueError("Split JSON fields do not match the canonical schema.")
    expected_values = {
        "schema_version": 1,
        "source_original_dataset_sha256": EXPECTED_SOURCE_SHA256,
        "normalized_schema_version": 1,
        "normalized_dataset_sha256": EXPECTED_NORMALIZED_SHA256,
        "num_records": EXPECTED_RECORD_COUNT,
        "seed": EXPECTED_SPLIT_SEED,
        "generation_method": "torch.utils.data.random_split",
        "split_sizes": EXPECTED_SPLIT_SIZES,
    }
    for field_name, expected_value in expected_values.items():
        if split_data[field_name] != expected_value:
            raise ValueError(f"Split JSON {field_name} is invalid.")
    if not isinstance(split_data["torch_version"], str) or not split_data["torch_version"]:
        raise ValueError("Split JSON torch_version is invalid.")

    split_indices = {
        split_name: _validate_indices(
            split_data[f"{split_name}_indices"], split_name
        )
        for split_name in _SPLIT_NAMES
    }
    all_indices = [index for indices in split_indices.values() for index in indices]
    if len(all_indices) != EXPECTED_RECORD_COUNT or len(set(all_indices)) != len(
        all_indices
    ):
        raise ValueError("Split partitions must be disjoint and cover every record.")
    if set(all_indices) != set(range(EXPECTED_RECORD_COUNT)):
        raise ValueError("Split partitions do not cover the complete dataset.")
    if split_indices["test"][:3] != [747, 1423, 1322]:
        raise ValueError("Split JSON test_indices fixed prefix is invalid.")

    hashes = split_data["split_indices_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(_SPLIT_NAMES):
        raise ValueError("Split JSON split_indices_sha256 fields are invalid.")
    for split_name, indices in split_indices.items():
        if hashes[split_name] != _split_indices_sha256(indices):
            raise ValueError(f"Split JSON {split_name}_indices SHA256 is invalid.")
    return split_indices


def load_and_validate_split(split_path: str | Path) -> dict[str, list[int]]:
    """Load and validate a split JSON file."""
    path = Path(split_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"split_path must be a regular file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return validate_split_data(json.load(handle))


def _require_torch_for_generation():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Generating the historical random_split order requires PyTorch. "
            "Use --verify to validate an existing split without PyTorch."
        ) from error
    return torch


def generate_split(seed: int = EXPECTED_SPLIT_SEED) -> dict[str, Any]:
    """Reproduce the official PyTorch ``random_split`` index order."""
    if seed != EXPECTED_SPLIT_SEED:
        raise ValueError(f"seed must be {EXPECTED_SPLIT_SEED}.")
    torch = _require_torch_for_generation()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        range(EXPECTED_RECORD_COUNT),
        list(EXPECTED_SPLIT_SIZES.values()),
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
        "normalized_schema_version": 1,
        "normalized_dataset_sha256": EXPECTED_NORMALIZED_SHA256,
        "num_records": EXPECTED_RECORD_COUNT,
        "seed": seed,
        "generation_method": "torch.utils.data.random_split",
        "torch_version": str(torch.__version__),
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "split_indices_sha256": {
            split_name: _split_indices_sha256(indices)
            for split_name, indices in split_indices.items()
        },
        "train_indices": split_indices["train"],
        "val_indices": split_indices["val"],
        "test_indices": split_indices["test"],
    }
    validate_split_data(split_data)
    return split_data


def _write_split(split_data: dict[str, Any], output_path: Path, force: bool) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing split JSON: {output_path}. Use --force."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(split_data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the fixed-seed GMTNet dielectric split."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", metavar="SPLIT_JSON", help="Validate a split JSON.")
    action.add_argument("--output", metavar="SPLIT_JSON", help="Write a generated split JSON.")
    parser.add_argument("--seed", type=int, default=EXPECTED_SPLIT_SEED)
    parser.add_argument(
        "--record-count",
        type=int,
        default=EXPECTED_RECORD_COUNT,
        help=f"Must be {EXPECTED_RECORD_COUNT}; validates the source record count.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow overwriting --output."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the split CLI."""
    args = _build_parser().parse_args(argv)
    try:
        if args.record_count != EXPECTED_RECORD_COUNT:
            raise ValueError(
                f"record_count must be {EXPECTED_RECORD_COUNT}; "
                "the source dataset is incomplete or incompatible."
            )
        if args.verify:
            split_indices = load_and_validate_split(args.verify)
            print(
                json.dumps(
                    {"status": "ok", "sizes": {key: len(value) for key, value in split_indices.items()}},
                    sort_keys=True,
                )
            )
        else:
            _write_split(generate_split(args.seed), Path(args.output), args.force)
            print(json.dumps({"status": "ok", "output": str(Path(args.output).resolve())}))
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
