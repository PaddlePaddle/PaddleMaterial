import os

import numpy as np
import pytest

from ppmat.datasets.liflow_dataset import LiFlowDataset

DATA = os.path.join(
    os.path.dirname(__file__), "fixtures", "liflow", "dataset_mini"
)


def make(path, cache, **kwargs):
    options = {
        "path": path,
        "index_file": "train_800K.csv",
        "time_delay_steps": 100,
        "seed": 42,
        "random_time": False,
        "cache_path": cache,
    }
    options.update(kwargs)
    return LiFlowDataset(**options)


def values(sample):
    out = {}
    for key in ("positions_1", "positions_2", "prior", "target", "elements",
                "atomic_numbers", "edge_index", "shifts", "time", "temp",
                "num_atoms", "frame_start", "frame_end"):
        value = sample[key]
        if hasattr(value, "numpy"):  # ConcatNumpyWarper
            value = value.numpy()
        out[key] = np.asarray(value)
    return out


def test_sample_schema_and_determinism(tmp_path):
    ds1 = make(DATA, str(tmp_path / "c1"))
    ds2 = make(DATA, str(tmp_path / "c2"))
    s1, s2 = values(ds1[0]), values(ds2[0])
    assert len(ds1) == 1
    assert s1["positions_1"].shape == (14, 3)
    assert s1["positions_2"].shape == (14, 3)
    assert s1["prior"].shape == (14, 3)
    assert s1["edge_index"].shape[0] == 2 and s1["edge_index"].shape[1] > 0
    assert s1["shifts"].shape == (s1["edge_index"].shape[1], 3)
    for key in ("positions_1", "positions_2", "prior", "target", "elements",
                "edge_index", "shifts", "time", "temp"):
        np.testing.assert_array_equal(s1[key], s2[key], err_msg=key)
    assert int(s1["frame_start"]) == 0
    assert int(s1["frame_end"]) == 100
    assert float(s1["time"]) == 0.5


def test_shifts_are_cartesian_lattice_images(tmp_path):
    sample = values(make(DATA, str(tmp_path / "c"))[0])
    lattice = np.diag([6.0, 6.0, 6.0])
    cart = sample["shifts"] / np.array([6.0, 6.0, 6.0])
    np.testing.assert_allclose(cart, np.round(cart), atol=1e-6)
    assert np.abs(cart).max() >= 1  # at least one true periodic image edge


def test_different_seed_changes_prior(tmp_path):
    a = values(make(DATA, str(tmp_path / "a"), seed=1)[0])
    b = values(make(DATA, str(tmp_path / "b"), seed=2)[0])
    assert not np.array_equal(a["prior"], b["prior"])


def test_split_counts_and_composition_weights(tmp_path):
    train = make(
        DATA,
        str(tmp_path / "train"),
        train_valid_split=True,
        dataset_split="train",
        num_valid_samples=0,
        num_train_samples=1,
        sample_weight_comp=True,
    )
    valid = make(
        DATA,
        str(tmp_path / "valid"),
        train_valid_split=True,
        dataset_split="valid",
        num_valid_samples=1,
    )
    assert len(train) == 1
    assert len(valid) == 1
    assert train.sample_weights.shape == (1,)
    assert valid.sample_weights.shape == (1,)


def test_adaptive_prior_amplifies_lithium(tmp_path):
    ds = make(DATA, str(tmp_path / "c"), seed=7, random_time=True)
    samples = [values(ds[0]) for _ in range(300)]
    stack = np.stack([sample["prior"] for sample in samples])  # [300, 14, 3]
    atomic_numbers = samples[0]["atomic_numbers"]
    li = stack[:, atomic_numbers == 3, :].std()
    frame = stack[:, atomic_numbers != 3, :].std()
    assert li > 5.0 * frame
