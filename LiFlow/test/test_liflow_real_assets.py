import csv
import os

import numpy as np
import paddle
import pytest

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.models.liflow import LiFlow
from ppmat.predictor.integrator_predictor import IntegratorPredictor


DATA_ROOT = r"D:\实验\liflow_official_data\data\universal"
P_CHECKPOINT = r"D:\实验\liflow_reference\ckpt\converted\P_universal\propagator.pdparams"
C_CHECKPOINT = r"D:\实验\liflow_reference\ckpt\converted\C_universal\corrector.pdparams"


def _external_assets_available():
    if os.name != "nt":
        return False
    required = [
        os.path.join(DATA_ROOT, "test_800K.csv"),
        os.path.join(DATA_ROOT, "element_index.npy"),
        os.path.join(DATA_ROOT, "atomic_numbers.npy"),
        os.path.join(DATA_ROOT, "lattice.npy"),
        os.path.join(DATA_ROOT, "positions_800K.npz"),
        P_CHECKPOINT,
        C_CHECKPOINT,
    ]
    return all(os.path.isfile(path) for path in required)


pytestmark = pytest.mark.skipif(
    not _external_assets_available(),
    reason="LiFlow external real-data/checkpoint assets are unavailable",
)


def _model():
    return LiFlow(
        num_features=64,
        num_radial_basis=20,
        num_layers=3,
        num_elements=77,
        r_max=5.0,
        r_offset=0.5,
        ref_temp=1000.0,
        prediction_mode="velocity",
    )


def _batch(sample):
    collated = DefaultCollator()([sample])
    batch = {key: getattr(collated, key) for key in collated.keys}
    for key, value in list(batch.items()):
        if isinstance(value, np.ndarray):
            dtype = "int64" if np.issubdtype(value.dtype, np.integer) else "float32"
            batch[key] = paddle.to_tensor(value, dtype=dtype)
    return batch


def _dataset(tmp_path):
    with open(os.path.join(DATA_ROOT, "test_800K.csv"), newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    index_file = tmp_path / "test_800K_one_sample.csv"
    with open(index_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    return LiFlowDataset(
        path=DATA_ROOT,
        index_file=str(index_file),
        time_delay_steps=100,
        seed=42,
        random_time=False,
        cache_path=str(tmp_path / "liflow_real_cache"),
    )


def test_real_dataset_schema_element_mapping_and_periodic_graph(tmp_path):
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    required = {
        "positions_1", "positions_2", "prior", "target", "elements",
        "atomic_numbers", "lattice", "edge_index", "shifts", "time",
        "temp", "num_atoms", "frame_start", "frame_end",
    }
    assert required.issubset(set(sample.keys))
    atomic_numbers = np.asarray(sample["atomic_numbers"])
    elements = np.asarray(sample["elements"])
    np.testing.assert_array_equal(elements, dataset.element_index[atomic_numbers])
    assert sample["positions_1"].shape == sample["positions_2"].shape
    assert sample["positions_1"].shape[1] == 3
    assert sample["edge_index"].shape[0] == 2
    assert sample["edge_index"].shape[1] > 0
    assert sample["shifts"].shape == (sample["edge_index"].shape[1], 3)
    assert np.isfinite(np.asarray(sample["shifts"])).all()
    assert int(sample["frame_end"]) - int(sample["frame_start"]) == 100


def test_real_checkpoints_load_liflow_forward_and_integrate(tmp_path):
    propagator = _model()
    corrector = _model()
    predictor = IntegratorPredictor(
        propagator=propagator,
        corrector=corrector,
        propagator_checkpoint_path=P_CHECKPOINT,
        corrector_checkpoint_path=C_CHECKPOINT,
    )
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    batch = _batch(sample)
    output = propagator(batch)
    prediction = output["pred_dict"]["target"]
    assert prediction.shape == [int(sample["num_atoms"]), 3]
    assert bool(paddle.all(paddle.isfinite(prediction)))

    trajectory = predictor.run(
        np.asarray(sample["positions_1"], dtype=np.float32),
        steps=2,
        flow_steps=1,
        solver="euler",
        seed=42,
        prior="Normal",
        prior_scale=0.0,
        lattice=np.asarray(sample["lattice"], dtype=np.float32),
        elements=np.asarray(sample["elements"], dtype=np.int64),
        temp=float(sample["temp"]),
        fix_com=False,
        use_corrector=False,
    )
    assert trajectory.shape == (3, int(sample["num_atoms"]), 3)
    assert np.isfinite(trajectory).all()
