# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""PyTorch/Paddle forward alignment tests against frozen reference goldens.

Loads reference_outputs/mini_state_dict.npz into a same-hyperparameter Paddle
DualPaiNN (transposing [in,out] Paddle Linear weights to [out,in] torch layout),
then compares network outputs on the three deterministic fixtures.
"""

import json
import os

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

from ppmat.models.liflow.dual_painn import DualPaiNN
from ppmat.models.liflow.flow_module import LiFlow
from ppmat.utils import save_load

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "liflow")
REF = os.path.join(FIX, "reference_outputs")

TAGS = ["14atoms", "nonperiodic", "cutoff_edge"]


def _tensor(array, dtype):
    arr = np.asarray(array)
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    return paddle.to_tensor(arr)


def load_aligned_model():
    with open(os.path.join(REF, "mini_meta.json")) as f:
        meta = json.load(f)
    model = DualPaiNN(
        num_features=meta["num_features"],
        num_radial_basis=meta["num_radial_basis"],
        num_layers=meta["num_layers"],
        num_elements=meta["num_elements"],
        r_max=meta["r_max"],
        r_offset=meta["r_offset"],
        ref_temp=meta["ref_temp"],
    )
    src = np.load(os.path.join(REF, "mini_state_dict.npz"))
    target = dict(model.state_dict())
    missing = [k for k in src.files if k not in target]
    extra = [k for k in target if k not in src.files]
    assert not missing, f"missing keys: {missing}"
    assert not extra, f"unexpected keys: {extra}"
    payload = {}
    for key in src.files:
        arr = np.asarray(src[key])
        tshape = tuple(target[key].shape)
        if key == "atom_embedding.weight":
            value = arr  # Paddle/torch Embedding share [num_elements, features]
        elif arr.ndim == 2:
            # Paddle Linear weight layout is [in, out]; torch is [out, in].
            value = arr.T.copy()
        else:
            value = arr  # 1-D/0-D biases and buffers
        value = value.reshape(tshape)
        payload[key] = paddle.to_tensor(value)
    model.set_state_dict(payload)
    return model


@pytest.fixture(scope="module")
def model():
    return load_aligned_model()


@pytest.mark.parametrize("tag", TAGS)
def test_forward_matches_reference(model, tag):
    inp = np.load(os.path.join(REF, f"mini_{tag}_input.npz"))
    feed = {
        "positions_1": _tensor(inp["positions_1"], "float32"),
        "positions_2": _tensor(inp["positions_2"], "float32"),
        "elements": _tensor(inp["elements"], "int64"),
        "edge_index": _tensor(inp["edge_index"], "int64"),
        "shifts": _tensor(inp["shifts"], "float32"),
        "time": _tensor(inp["time"], "float32"),
        "temp": _tensor(inp["temp"], "float32"),
    }
    model.eval()
    with paddle.no_grad():
        out = model(feed).numpy()
    golden = np.load(os.path.join(REF, f"mini_{tag}_output.npy"))
    assert out.shape == golden.shape
    diff = np.abs(out - golden)
    print(f"{tag}: max_abs_diff={diff.max():.3e} mean_abs={diff.mean():.3e}")
    np.testing.assert_allclose(out, golden, rtol=2e-5, atol=2e-5)


def test_state_dict_keys_match_reference(model):
    # parameter + buffer names must align 1:1 with the torch golden export
    src = np.load(os.path.join(REF, "mini_state_dict.npz"))
    assert set(src.files) == set(model.state_dict().keys())


@pytest.mark.parametrize("tag", TAGS)
def test_loss_gradients_match_reference(model, tag):
    inp = np.load(os.path.join(REF, f"mini_{tag}_input.npz"))
    feed = {
        "positions_1": _tensor(inp["positions_1"], "float32"),
        "positions_2": _tensor(inp["positions_2"], "float32"),
        "elements": _tensor(inp["elements"], "int64"),
        "edge_index": _tensor(inp["edge_index"], "int64"),
        "shifts": _tensor(inp["shifts"], "float32"),
        "time": _tensor(inp["time"], "float32"),
        "temp": _tensor(inp["temp"], "float32"),
    }
    for parameter in model.parameters():
        parameter.clear_gradient()
    loss = paddle.mean(paddle.sum(model(feed) ** 2, axis=-1))
    loss.backward()
    golden = np.load(os.path.join(REF, f"mini_{tag}_loss_gradients.npz"))
    assert float(np.abs(loss.numpy() - golden["loss"]).max()) <= 1e-6
    max_diff = 0.0
    for name, parameter in model.named_parameters():
        actual = parameter.grad.numpy() if parameter.grad is not None else np.zeros(parameter.shape, dtype="float32")
        expected = golden[name]
        if expected.ndim == 2 and name != "atom_embedding.weight":
            expected = expected.T
        diff = np.abs(actual - expected)
        max_diff = max(max_diff, float(diff.max()))
    print(f"{tag}: loss_gradients max_abs_diff={max_diff:.3e}")
    assert max_diff <= 1e-6


def _load_train_case(mode):
    ref = np.load(os.path.join(REF, f"mini_train_{mode}.npz"))
    inputs = []
    for tag in TAGS:
        raw = np.load(os.path.join(REF, f"mini_{tag}_input.npz"))
        inputs.append({key: raw[key] for key in raw.files})
    return ref, inputs


def _flow_model(mode):
    model = LiFlow(num_features=8, num_radial_basis=5, num_layers=1,
                   num_elements=40, r_max=5.0, r_offset=0.5,
                   ref_temp=1000.0, prediction_mode=mode)
    src = np.load(os.path.join(REF, "mini_state_dict.npz"))
    payload = {}
    for key in src.files:
        value = np.asarray(src[key])
        if value.ndim == 2 and key != "atom_embedding.weight":
            value = value.T.copy()
        payload[f"network.{key}"] = paddle.to_tensor(value.reshape(model.state_dict()[f"network.{key}"].shape))
    model.set_state_dict(payload)
    return model


def _batch(raw, time):
    return {
        "positions_1": _tensor(raw["positions_1"], "float32"),
        "positions_2": _tensor(raw["positions_2"], "float32"),
        "prior": _tensor(raw["prior"], "float32"),
        "elements": _tensor(raw["elements"], "int64"),
        "edge_index": _tensor(raw["edge_index"], "int64"),
        "shifts": _tensor(raw["shifts"], "float32"),
        "time": paddle.full([raw["positions_1"].shape[0]], float(time), dtype="float32"),
        "temp": _tensor(raw["temp"], "float32"),
        "batch": _tensor(raw["batch"], "int64"),
    }


def _train_steps(model, optimizer, cases, times, mode, start, stop):
    losses = []
    for index in range(start, stop):
        batch = _batch(cases[index % len(cases)], times[index])
        loss = model(batch)["loss_dict"]["loss"]
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        losses.append(float(loss.numpy()))
    return losses


@pytest.mark.parametrize("mode", ["velocity", "data"])
def test_two_epoch_training_matches_reference_and_resume(tmp_path, mode):
    reference, cases = _load_train_case(mode)
    model = _flow_model(mode)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    actual = _train_steps(model, optimizer, cases, reference["times"], mode, 0, 6)
    np.testing.assert_allclose(actual, reference["losses"], rtol=2e-5, atol=2e-5)
    for key in model.state_dict():
        expected = reference[f"epoch_2::{key.removeprefix('network.')}"]
        actual_value = model.state_dict()[key].numpy()
        if expected.ndim == 2 and not key.endswith("atom_embedding.weight"):
            expected = expected.T
        np.testing.assert_allclose(actual_value, expected, rtol=2e-5, atol=2e-5, err_msg=key)

    resumed = _flow_model(mode)
    resumed_optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=resumed.parameters())
    _train_steps(resumed, resumed_optimizer, cases, reference["times"], mode, 0, 3)
    save_load.save_checkpoint(
        resumed, resumed_optimizer, {"global_step": 3}, output_dir=str(tmp_path), prefix=mode
    )
    restored = _flow_model(mode)
    restored_optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=restored.parameters())
    state = save_load.load_checkpoint(str(tmp_path / "checkpoints" / mode), restored, restored_optimizer)
    assert state["global_step"] == 3
    _train_steps(restored, restored_optimizer, cases, reference["times"], mode, 3, 6)
    for key in model.state_dict():
        np.testing.assert_allclose(restored.state_dict()[key].numpy(), model.state_dict()[key].numpy(), rtol=2e-5, atol=2e-5, err_msg=key)
