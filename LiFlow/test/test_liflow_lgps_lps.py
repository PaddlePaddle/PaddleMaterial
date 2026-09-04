import csv
import json
import os

import numpy as np
import paddle
import pytest

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.models.liflow import LiFlow
from ppmat.predictor.integrator_predictor import IntegratorPredictor


ROOT = os.path.dirname(os.path.dirname(__file__))
SPECIAL_CKPT = os.path.join(ROOT, "test", "fixtures", "liflow", "reference_outputs", "special_ckpt")
LGPS_DATA = os.environ.get("LIFLOW_LGPS_DATA", r"D:\实验\liflow_official_data\data\LGPS")
LPS_DATA = os.environ.get("LIFLOW_LPS_DATA", r"D:\实验\liflow_official_data\data\LPS")




def _checkpoint_path(name, filename):
    return os.path.join(SPECIAL_CKPT, name, filename)


def _require_lgps_assets():
    index_path = os.path.join(LGPS_DATA, "train_25ps.csv")
    required = [
        index_path,
        os.path.join(LGPS_DATA, "element_index.npy"),
        os.path.join(LGPS_DATA, "atomic_numbers.npy"),
        os.path.join(LGPS_DATA, "lattice.npy"),
    ]
    if not all(os.path.isfile(path) for path in required):
        pytest.skip("LGPS 真实数据资产不存在；请设置 LIFLOW_LGPS_DATA 指向完整数据目录")
    with open(index_path, newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    archive = os.path.join(LGPS_DATA, f"positions_{int(float(row['temp']))}K.npz")
    if not os.path.isfile(archive):
        pytest.skip(f"LGPS 轨迹资产不存在: {archive}")


def _require_checkpoints(names):
    paths = []
    for name, filename in names:
        path = _checkpoint_path(name, filename)
        if not os.path.isfile(path):
            pytest.skip(f"LiFlow 外部 checkpoint 资产不存在: {path}")
        paths.append(path)
    return paths


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


def _lgps_dataset(tmp_path):
    _require_lgps_assets()
    with open(os.path.join(LGPS_DATA, "train_25ps.csv"), newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    index_file = tmp_path / "train_25ps_one_sample.csv"
    with open(index_file, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    return LiFlowDataset(
        path=LGPS_DATA,
        index_file=str(index_file),
        time_delay_steps=67,
        seed=42,
        random_time=False,
        cache_path=str(tmp_path / "liflow_lgps_cache"),
    )


def test_lgps_dataset_mapping_and_periodic_graph(tmp_path):
    dataset = _lgps_dataset(tmp_path)
    assert len(dataset) == 1
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
    assert sample["edge_index"].shape[0] == 2
    assert sample["edge_index"].shape[1] > 0
    assert sample["shifts"].shape == (sample["edge_index"].shape[1], 3)
    assert np.isfinite(np.asarray(sample["shifts"])).all()
    assert int(sample["frame_end"]) - int(sample["frame_start"]) == 67


def test_lgps_checkpoints_forward_and_short_trajectories(tmp_path):
    dataset = _lgps_dataset(tmp_path)
    sample = dataset[0]
    batch = _batch(sample)
    propagator = _model()
    corrector_01 = _model()
    corrector_02 = _model()
    _require_checkpoints([
        ("P_LGPS", "propagator.pdparams"),
        ("C_LGPS_0.1", "corrector.pdparams"),
        ("C_LGPS_0.2", "corrector.pdparams"),
    ])
    models = {
        "P_LGPS": (propagator, _checkpoint_path("P_LGPS", "propagator.pdparams")),
        "C_LGPS_0.1": (corrector_01, _checkpoint_path("C_LGPS_0.1", "corrector.pdparams")),
        "C_LGPS_0.2": (corrector_02, _checkpoint_path("C_LGPS_0.2", "corrector.pdparams")),
    }
    for name, (model, checkpoint) in models.items():
        IntegratorPredictor._load_checkpoint(model, checkpoint, name)
        output = model(batch)["pred_dict"]["target"]
        assert output.shape == [len(sample["elements"]), 3]
        assert bool(paddle.all(paddle.isfinite(output))), name

    predictor_01 = IntegratorPredictor(propagator=_model(), corrector=_model())
    IntegratorPredictor._load_checkpoint(
        predictor_01.propagator, _checkpoint_path("P_LGPS", "propagator.pdparams"), "propagator"
    )
    IntegratorPredictor._load_checkpoint(
        predictor_01.corrector, _checkpoint_path("C_LGPS_0.1", "corrector.pdparams"), "corrector"
    )
    predictor_02 = IntegratorPredictor(propagator=_model(), corrector=_model())
    IntegratorPredictor._load_checkpoint(
        predictor_02.propagator, _checkpoint_path("P_LGPS", "propagator.pdparams"), "propagator"
    )
    IntegratorPredictor._load_checkpoint(
        predictor_02.corrector, _checkpoint_path("C_LGPS_0.2", "corrector.pdparams"), "corrector"
    )

    common = dict(
        initial_positions=np.asarray(sample["positions_1"], dtype=np.float32),
        flow_steps=10,
        solver="euler",
        seed=42,
        prior="Normal",
        prior_scale=0.0,
        lattice=np.asarray(sample["lattice"], dtype=np.float32),
        elements=np.asarray(sample["elements"], dtype=np.int64),
        temp=float(sample["temp"]),
        fix_com=False,
    )
    p_only = predictor_01.run(steps=1, use_corrector=False, **common)
    p_plus_c = predictor_02.run(steps=1, use_corrector=True, **common)
    assert p_only.shape == (2, len(sample["elements"]), 3)
    assert p_plus_c.shape == (2, len(sample["elements"]), 3)
    assert np.isfinite(p_only).all()
    assert np.isfinite(p_plus_c).all()


def test_special_checkpoint_audits_are_clean():
    audit_paths = [os.path.join(SPECIAL_CKPT, name, "audit.json") for name in (
        "P_LGPS", "C_LGPS_0.1", "C_LGPS_0.2", "P_LPS", "C_LPS"
    )]
    if not all(os.path.isfile(path) for path in audit_paths):
        pytest.skip("LiFlow checkpoint 审计资产不存在")
    for path in audit_paths:
        name = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as file:
            audit = json.load(file)
        assert audit["missing_keys"] == [], name
        assert audit["unexpected_keys"] == [], name
        assert audit["shape_mismatch"] == [], name


def test_lps_checkpoints_load_without_real_data():
    _require_checkpoints([
        ("P_LPS", "propagator.pdparams"),
        ("C_LPS", "corrector.pdparams"),
    ])
    IntegratorPredictor(
        propagator=_model(),
        corrector=_model(),
        propagator_checkpoint_path=_checkpoint_path("P_LPS", "propagator.pdparams"),
        corrector_checkpoint_path=_checkpoint_path("C_LPS", "corrector.pdparams"),
    )


def test_lps_real_data_end_to_end_is_explicitly_skipped():
    pytest.skip("LPS 只有 checkpoint、没有官方数据资产；禁止伪造数据或替用 Universal 数据")
