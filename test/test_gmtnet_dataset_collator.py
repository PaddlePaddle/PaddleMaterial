from pathlib import Path

import paddle
import yaml

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.datasets.gmtnet_dataset import GMTNetDielectricDataset


def _sample(data_index: int) -> dict:
    return {
        "graph": Data(
            x=paddle.to_tensor([[1.0, 0.0]], dtype="float32"),
            edge_index=paddle.to_tensor([[0], [0]], dtype="int64"),
            edge_attr=paddle.to_tensor([[0.0, 0.0, 1.0]], dtype="float32"),
        ),
        "feature_mask": paddle.eye(32, dtype="float32"),
        "matrix_equal": paddle.eye(9, dtype="float32").astype("bool"),
        "dielectric": paddle.eye(3, dtype="float32"),
        "id": f"sample-{data_index}",
        "data_index": paddle.to_tensor(data_index, dtype="int64"),
    }


def test_gmtnet_dataset_declares_dielectric_property_name():
    assert GMTNetDielectricDataset.property_names == ("dielectric",)


def test_default_collator_batches_gmtnet_sample_without_special_collator():
    batch = DefaultCollator()([_sample(7)])

    assert type(batch["graph"]).__name__ == "Batch"
    assert list(batch["graph"].x.shape) == [1, 2]
    assert list(batch["graph"].edge_index.shape) == [2, 1]
    assert list(batch["feature_mask"].shape) == [1, 32, 32]
    assert list(batch["matrix_equal"].shape) == [1, 9, 9]
    assert list(batch["dielectric"].shape) == [1, 3, 3]
    assert list(batch["data_index"].shape) == [1]
    assert batch["data_index"].dtype == paddle.int64
    assert batch["id"] == ["sample-7"]


def test_default_collator_batches_two_gmtnet_samples():
    batch = DefaultCollator()([_sample(7), _sample(9)])

    assert type(batch["graph"]).__name__ == "Batch"
    assert list(batch["graph"].x.shape) == [2, 2]
    assert list(batch["graph"].batch.shape) == [2]
    assert list(batch["feature_mask"].shape) == [2, 32, 32]
    assert list(batch["matrix_equal"].shape) == [2, 9, 9]
    assert list(batch["dielectric"].shape) == [2, 3, 3]
    assert batch["data_index"].numpy().tolist() == [7, 9]


def test_gmtnet_yaml_uses_default_collator_only():
    config_path = Path("property_prediction/configs/gmtnet/gmtnet_jarvis_dielectric.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for split_name in ("train", "val", "test"):
        assert config["Dataset"][split_name]["loader"]["collate_fn"] == "DefaultCollator"
