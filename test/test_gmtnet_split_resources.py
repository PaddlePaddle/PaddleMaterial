import hashlib
import json
from importlib import resources
from pathlib import Path

import yaml

from ppmat.datasets import gmtnet_dataset
from ppmat.datasets.gmtnet_dataset import (
    GMTNetDielectricDataset,
    _CachedPayload,
    _default_split_resource,
)
from ppmat.datasets.split_gmtnet_dataset import (
    EXPECTED_SOURCE_SHA256,
    load_and_validate_split,
)


EXPECTED_SHA256 = "8d426234d0a89d1794cccb3560c3b9d397f186253703f8b9e2cb86777b2fd4df"


def test_default_gmtnet_split_resource_is_canonical():
    resource = _default_split_resource()
    with resources.as_file(resource) as split_path:
        assert hashlib.sha256(split_path.read_bytes()).hexdigest() == EXPECTED_SHA256
        split_indices = load_and_validate_split(split_path)

    assert {name: len(indices) for name, indices in split_indices.items()} == {
        "train": 3770,
        "val": 471,
        "test": 472,
    }
    assert split_indices["test"][:3] == [747, 1423, 1322]


def test_explicit_gmtnet_split_path_keeps_order():
    resource = _default_split_resource()
    with resources.as_file(resource) as split_path:
        split_indices = load_and_validate_split(split_path)
        split_data = json.loads(split_path.read_text(encoding="utf-8"))

    for split_name in ("train", "val", "test"):
        assert split_indices[split_name] == split_data[f"{split_name}_indices"]


def test_dataset_uses_resource_fallback_without_current_directory(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "normalized.pkl"
    data_path.write_bytes(b"placeholder")
    payload = {"source_original_dataset_sha256": EXPECTED_SOURCE_SHA256}
    cache_entry = _CachedPayload(payload=payload, sha256="not-checked")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        GMTNetDielectricDataset,
        "_load_payload",
        classmethod(lambda cls, path: cache_entry),
    )
    monkeypatch.setattr(gmtnet_dataset, "GMTNetGraphConverter", lambda **kwargs: kwargs)

    dataset = GMTNetDielectricDataset(
        data_path=data_path,
        split="test",
        verify_sha256=False,
    )

    assert dataset.split_path.name == "split_gmtnet_dielectric_seed32.json"
    assert dataset._split_indices[:3] == (747, 1423, 1322)


def test_gmtnet_yaml_uses_dataset_resource_fallback():
    config_path = Path("property_prediction/configs/gmtnet/gmtnet_jarvis_dielectric.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for split_name in ("train", "val", "test"):
        dataset_params = config["Dataset"][split_name]["dataset"]["__init_params__"]
        assert "split_path" not in dataset_params
