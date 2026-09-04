import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

from ppmat.predictor.base import BasePredictor
from ppmat.predictor.integrator_predictor import IntegratorPredictor


class LightweightModel:
    def __init__(self, output, prediction_mode="velocity", use_scale=False, scale_model=None):
        self.output = output
        self.prediction_mode = prediction_mode
        self.use_scale = use_scale
        self.scale_model = scale_model
        self.calls = 0
        self.batches = []

    def eval(self):
        return self

    def predict(self, batch):
        self.calls += 1
        self.batches.append(batch)
        if callable(self.output):
            return self.output(self.calls, batch)
        return np.asarray(self.output, dtype=np.float32)


def make_predictor(propagator, corrector=None):
    return IntegratorPredictor(propagator=propagator, corrector=corrector)


def test_uses_real_constructor_and_inherits_base_predictor():
    predictor = make_predictor(LightweightModel(np.zeros((2, 3), dtype=np.float32)))
    assert isinstance(predictor, BasePredictor)
    assert predictor.propagator is predictor.model


def test_trajectory_shape_seed_and_different_prior():
    model = LightweightModel(np.zeros((4, 3), dtype=np.float32))
    predictor = make_predictor(model)
    initial = np.zeros((4, 3), dtype=np.float32)
    first = predictor.run(initial, steps=2, seed=7)
    first_prior = model.batches[-1]["prior"].numpy()
    second = predictor.run(initial, steps=2, seed=7)
    second_prior = model.batches[-1]["prior"].numpy()
    other = predictor.run(initial, steps=2, seed=8)
    other_prior = model.batches[-1]["prior"].numpy()
    assert first.shape == (3, 4, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], other[0])
    np.testing.assert_array_equal(first[0], second[0])
    assert np.array_equal(first_prior, second_prior)
    assert not np.array_equal(first_prior, other_prior)


def test_euler_uses_propagator_and_corrector_every():
    propagator = LightweightModel(np.ones((2, 3), dtype=np.float32))
    corrector = LightweightModel(np.full((2, 3), 2, dtype=np.float32))
    trajectory = make_predictor(propagator, corrector).run(
        np.zeros((2, 3), dtype=np.float32), steps=3, flow_steps=1,
        corrector_every=2, prior_scale=0.0, corrector_prior_scale=0.0,
        fix_com=False,
    )
    assert propagator.calls == 3
    assert corrector.calls == 1
    np.testing.assert_allclose(trajectory[-1] - trajectory[0], 5.0)


def test_corrector_none_and_euler_steps_plus_one():
    model = LightweightModel(np.ones((2, 3), dtype=np.float32))
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=0, flow_steps=2,
        corrector_every=1,
    )
    assert trajectory.shape == (1, 2, 3)
    assert model.calls == 0


def test_heun_calls_propagator_twice_per_step():
    model = LightweightModel(np.ones((2, 3), dtype=np.float32))
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=1, flow_steps=2,
        solver="heun", prior_scale=0.0, fix_com=False,
    )
    assert model.calls == 4
    np.testing.assert_allclose(trajectory[-1] - trajectory[0], 1.0)


def test_data_prediction_mode_and_data_argument():
    model = LightweightModel(np.full((2, 3), 3, dtype=np.float32), prediction_mode="data")
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=1, flow_steps=2, data=True, prior_scale=0.0, fix_com=False,
    )
    assert model.calls == 2
    np.testing.assert_allclose(
        trajectory[-1] - trajectory[0], np.full((2, 3), 3, dtype=np.float32),
    )


def test_batches_include_prior_and_seed_changes_it():
    model = LightweightModel(np.zeros((2, 3), dtype=np.float32))
    predictor = make_predictor(model)
    initial = np.zeros((2, 3), dtype=np.float32)
    predictor.run(initial, steps=1, seed=11, prior="Normal")
    first_prior = model.batches[-1]["prior"].numpy()
    predictor.run(initial, steps=1, seed=12, prior="Normal")
    second_prior = model.batches[-1]["prior"].numpy()
    assert not np.array_equal(first_prior, second_prior)


def test_learned_scale_multiplies_prior():
    def scale_model(batch):
        return np.zeros(2, dtype=np.float32)

    model = LightweightModel(
        np.zeros((2, 3), dtype=np.float32),
        use_scale=True,
        scale_model=scale_model,
    )
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=1, flow_steps=1,
        seed=5, fix_com=False,
    )
    np.testing.assert_allclose(trajectory[-1] - trajectory[0], 0.0, atol=1e-6)


def test_mass_weighted_center_of_mass_is_preserved():
    model = LightweightModel(np.array([[2, 0, 0], [0, 0, 0]], dtype=np.float32))
    initial = np.zeros((2, 3), dtype=np.float32)
    masses = np.array([1, 3], dtype=np.float32)
    trajectory = make_predictor(model).run(initial, steps=1, flow_steps=1, masses=masses)
    np.testing.assert_allclose((trajectory[-1] * masses[:, None]).sum(axis=0), 0.0, atol=1e-6)


@pytest.mark.parametrize("bad_output", [np.full((2, 3), np.nan), np.full((2, 3), 2001)])
def test_nan_or_large_displacement_retries(bad_output):
    def output(call, batch):
        return np.zeros((2, 3), dtype=np.float32) if call > 1 else bad_output

    model = LightweightModel(output)
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=1, flow_steps=1, seed=3,
        num_retries=1, fix_com=False,
    )
    assert model.calls == 2
    assert trajectory.shape == (2, 2, 3)


def test_exhausted_retries_raise_without_random_fallback():
    model = LightweightModel(np.full((2, 3), np.nan, dtype=np.float32))
    with pytest.raises(FloatingPointError, match="NaN"):
        make_predictor(model).run(
            np.zeros((2, 3), dtype=np.float32), steps=1, flow_steps=1, max_retries=2,
        )
    assert model.calls == 3


def test_uniform_scale_normal_zero_scale_gives_zero_noise():
    model = LightweightModel(np.zeros((2, 3), dtype=np.float32))
    trajectory = make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=2, flow_steps=1, seed=4,
        prior="UniformScaleNormal", prior_scale=0.0, fix_com=False,
    )
    np.testing.assert_allclose(trajectory[-1] - trajectory[0], 0.0, atol=1e-6)


def test_adaptive_prior_scales_lithium_vs_frame():
    model = LightweightModel(np.zeros((2, 3), dtype=np.float32))
    make_predictor(model).run(
        np.zeros((2, 3), dtype=np.float32), steps=400, flow_steps=1, seed=6,
        prior="AdaptiveMaxwellBoltzmann", fix_com=False,
        atomic_numbers=np.array([3, 1]), masses=np.ones(2),
        scale_Li_index=1, scale_frame_index=0,
        prior_matrix=[[1.0, 10.0], [1.0, 5.0]],
    )
    prior = np.stack([b["prior"].numpy() for b in model.batches])  # [400, 2, 3]
    li_std = prior[:, 0, :].std()
    frame_std = prior[:, 1, :].std()
    assert li_std > frame_std
    np.testing.assert_allclose(li_std / frame_std, 10.0, rtol=0.2)


def test_periodic_graph_shifts_are_cartesian():
    lattice = 10.0 * np.eye(3, dtype=np.float32)
    positions = np.array([[0, 0, 0], [8, 0, 0]], dtype=np.float32)
    edge, shifts = IntegratorPredictor._periodic_graph(positions, lattice, cutoff=5.0)
    assert edge.shape[0] == 2 and edge.shape[1] > 0
    assert shifts.shape == (edge.shape[1], 3)
    # shifts must already be Cartesian: integer multiples of the lattice vectors
    cart = np.round(shifts / np.array([10.0, 10.0, 10.0]))
    np.testing.assert_array_equal(cart, shifts / np.array([10.0, 10.0, 10.0]))


def test_missing_checkpoint_is_reported():
    model = LightweightModel(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="propagator checkpoint not found"):
        IntegratorPredictor(
            propagator=model,
            propagator_checkpoint_path="definitely-missing.pdparams",
        )


def test_missing_propagator_is_rejected_in_constructor():
    with pytest.raises(ValueError, match="propagator model or model_name/config_path"):
        IntegratorPredictor()
