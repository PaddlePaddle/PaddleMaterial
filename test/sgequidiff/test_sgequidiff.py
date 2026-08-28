# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGEQuiDiff integration tests.
"""

from pathlib import Path

import numpy as np
import paddle
import pytest

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.models.sgequidiff.diffusion_model import EquivariantDiffusionModel
from ppmat.models.sgequidiff.sgequidiff import SGEQuiDiff
from ppmat.models.sgequidiff.vocabs import build_embedding_tools
from ppmat.models.sgequidiff.wyckoff_geometry import build_wyckoff_geometry


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=2):
    """Create a synthetic batch dict with valid P1 (space group 1) data."""
    total = batch_size * atoms_per_crystal
    return {
        "frac_coords": paddle.rand([total, 3]),
        "element_indices": paddle.to_tensor([1, 2] * batch_size, dtype=paddle.int64),
        "wyckoff_indices": paddle.to_tensor([0] * total, dtype=paddle.int64),
        "space_group_indices": paddle.to_tensor([0] * batch_size, dtype=paddle.int64),
        "n_atoms_per_asu": paddle.to_tensor(
            [atoms_per_crystal] * batch_size, dtype=paddle.int64
        ),
        "wyckoff_shape_indices": paddle.to_tensor([0] * total, dtype=paddle.int64),
        "lattice_lengths": paddle.to_tensor(
            [[5.0, 5.0, 5.0]] * batch_size, dtype=paddle.float32
        ),
        "lattice_angles": paddle.to_tensor(
            [[90.0, 90.0, 90.0]] * batch_size, dtype=paddle.float32
        ),
    }


def _make_mixed_space_group_batch(batch_size=3):
    """Synthetic batch with multiple space groups (P1, P2, Pm)."""
    sg_indices = paddle.to_tensor([0, 3, 5], dtype=paddle.int64)[:batch_size]
    n_crystals = sg_indices.shape[0]
    n_atoms = 2
    total = n_crystals * n_atoms
    return {
        "space_group_indices": sg_indices,
        "lattice_lengths": paddle.full([n_crystals, 3], 5.0),
        "lattice_angles": paddle.full([n_crystals, 3], 90.0),
        "n_atoms_per_asu": paddle.full([n_crystals], n_atoms, dtype=paddle.int64),
        "element_indices": paddle.to_tensor([1, 2] * n_crystals, dtype=paddle.int64),
        "wyckoff_indices": paddle.zeros([total], dtype=paddle.int64),
        "wyckoff_shape_indices": paddle.zeros([total], dtype=paddle.int64),
        "frac_coords": paddle.rand([total, 3]),
    }


def _make_sgequidiff():
    """Build a lightweight SGEQuiDiff for fast integration tests."""
    return SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=2,
        noise_scheduler_num_monte_carlo_samples=2,
        num_lattice_translations=1,
    )


@pytest.fixture(scope="module")
def model():
    """Lightweight MLP EquivariantDiffusionModel, shared across the class.

    Kept small (10 timesteps, tiny embeddings) so the forward/loss checks
    run quickly while still exercising the real score-matching path.
    """
    wyckoff_geometry = build_wyckoff_geometry()
    embedding_tools = build_embedding_tools()
    return EquivariantDiffusionModel(
        model_type="mlp",
        num_timesteps=10,
        num_lattice_translations=1,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=32,
        num_plane_wave_freqs=16,
        wyckoff_geometry=wyckoff_geometry,
        embedding_tools=embedding_tools,
    )


class TestModelBuildForward:
    """Build + forward + loss validity of the coordinate diffusion model.

    Overall question: can the model be constructed from a light config and
    complete one training forward pass that yields a valid scalar loss?
    """

    def test_build_from_config_produces_valid_output(self, model):
        """Forward pass returns a scalar, finite, positive loss."""
        batch_data = _make_synthetic_batch()
        output = model(batch_data)

        assert "loss_dict" in output
        loss = output["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()
        assert float(loss) > 0.0


class TestSGEQuiDiffEndToEnd:
    """SGEQuiDiff end-to-end: build -> sample / forward.

    Overall questions: does unconditional sampling produce a complete,
    internally-consistent crystal result, and does the full model forward
    produce a valid loss?
    """

    @pytest.fixture(scope="class")
    def sampler(self):
        paddle.seed(0)
        return _make_sgequidiff()

    def test_sample_returns_valid_result_format(self, sampler):
        """Unconditional sampling yields crystals with self-consistent fields.

        Each generated structure must report a positive atom count whose
        atom_types / frac_coords match, plus 3 lattice lengths and angles.
        """
        batch_data = {
            "structure_array": {
                "num_atoms": paddle.to_tensor([2], dtype=paddle.int64),
            }
        }
        out = sampler.sample(batch_data)
        assert "result" in out
        for crystal in out["result"]:
            assert crystal["num_atoms"] > 0
            assert len(crystal["atom_types"]) == crystal["num_atoms"]
            assert len(crystal["frac_coords"]) == crystal["num_atoms"]
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            for pt in crystal["frac_coords"]:
                assert len(pt) == 3

    def test_forward_returns_loss_dict(self, sampler):
        """Full SGEQuiDiff forward pass returns a scalar, finite loss."""
        batch_data = _make_synthetic_batch()
        out = sampler(batch_data)
        assert "loss_dict" in out
        loss = out["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()

    def test_sample_keeps_hydrogen_element(self, sampler, monkeypatch):
        """Regression: element_indices=0 (H) must survive the sample() filter.

        sample() maps 0-indexed element_indices through chemical_symbols,
        which is 1-indexed (chemical_symbols[0] == "X" placeholder). An
        off-by-one here drops H from every generated crystal.
        """
        from ppmat.models.sgequidiff.asu_crystal import ASUCrystal

        crystal = ASUCrystal(
            space_group_number=paddle.to_tensor(1, dtype=paddle.int64),
            conventional_lattice_lengths=paddle.to_tensor(
                [[5.0, 5.0, 5.0]], dtype=paddle.float32
            ),
            conventional_lattice_angles=paddle.to_tensor(
                [[90.0, 90.0, 90.0]], dtype=paddle.float32
            ),
            element_indices=paddle.to_tensor([0, 1, 2], dtype=paddle.int64),
            wyckoff_indices=paddle.to_tensor([0, 0, 0], dtype=paddle.int64),
            conventional_frac_coords=paddle.to_tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
                dtype=paddle.float32,
            ),
        )

        def _fake_sample_crystal(
            batch_size, diffusion_snr=0.4, temperature=1.0, **kwargs
        ):
            return [crystal]

        monkeypatch.setattr(sampler, "sample_crystal", _fake_sample_crystal)
        out = sampler.sample(
            {
                "structure_array": {
                    "num_atoms": paddle.to_tensor([1], dtype=paddle.int64)
                }
            }
        )
        assert len(out["result"]) == 1
        atom_types = out["result"][0]["atom_types"]
        assert 1 in atom_types, f"H (Z=1) was filtered out: {atom_types}"
        assert set(atom_types) == {1, 2, 3}


class TestFullTrainingObjective:
    """Full training objective: MLE on discrete parts + score matching.

    Overall questions: can the model be trained end-to-end (forward +
    backward with gradients flowing to all four submodules), does the loss
    actually decrease, and do sampled structures stay physically valid?
    """

    @pytest.fixture(scope="class")
    def sampler(self):
        paddle.seed(0)
        return _make_sgequidiff()

    def test_mixed_space_group_batch_trains(self, sampler):
        """Training on mixed space groups: forward + backward propagate
        gradients into every trainable submodule."""
        sampler.train()
        batch = _make_mixed_space_group_batch()
        out = sampler(batch)
        loss = out["loss_dict"]["loss"]
        assert not paddle.isnan(loss).item()
        pred = out["pred_dict"]
        for key in (
            "space_group_log_prob",
            "lattice_log_prob",
            "elements_log_prob",
            "wyckoffs_log_prob",
            "termination_log_prob",
            "score_matching_loss",
        ):
            assert key in pred, f"missing artifact: {key}"
        loss.backward()
        for name, p in sampler.named_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                assert name.split(".")[0] in (
                    "atom_coord_diffusion_model",
                    "space_group_sampler",
                    "lattice_sampler",
                    "wyckoff_and_element_sampler",
                )

    def test_training_loss_decreases_over_steps(self):
        """Gradient descent drives the total training loss down."""
        batch = _make_mixed_space_group_batch()
        n_steps = 40
        window = 5
        for seed in (0, 1, 2):
            paddle.seed(seed)
            sampler = _make_sgequidiff()
            sampler.train()
            opt = paddle.optimizer.Adam(
                parameters=sampler.parameters(), learning_rate=1e-3
            )
            losses = []
            for _ in range(n_steps):
                out = sampler(batch)
                loss = out["loss_dict"]["loss"]
                losses.append(float(loss))
                opt.clear_grad()
                loss.backward()
                opt.step()
            initial = sum(losses[:window]) / window
            best = min(
                sum(losses[i : i + window]) / window
                for i in range(n_steps - window + 1)
            )
            assert best < initial, (
                f"seed {seed}: loss did not improve "
                f"(initial={initial:.2f}, best={best:.2f})"
            )

    def test_sampled_structure_validity(self, sampler):
        """Sampled crystals keep elements, coords and lattice params in range.

        Guards the sampling path against generating out-of-domain values
        (elements outside the encoding table, coords outside the unit cell,
        or lattice params outside the dataset bounds).
        """
        from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE

        out = sampler.sample(
            {"structure_array": {"num_atoms": paddle.to_tensor([2, 3], dtype="int64")}}
        )
        assert len(out["result"]) > 0
        for crystal in out["result"]:
            n = crystal["num_atoms"]
            assert n > 0
            assert len(crystal["atom_types"]) == n
            assert len(crystal["frac_coords"]) == n
            for elem in crystal["atom_types"]:
                assert 1 <= elem <= ELEMENT_ENCODING_SIZE, f"elem out of range: {elem}"
            for coords in crystal["frac_coords"]:
                assert len(coords) == 3
                assert all(
                    0.0 <= c < 1.0 for c in coords
                ), f"coords out of cell: {coords}"
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            assert all(2.0 <= length <= 133.0 for length in crystal["lengths"])
            assert all(60.0 <= a <= 135.0 for a in crystal["angles"])


def _write_synthetic_mp20_npz(root_dir, split="train", num_crystals=8):
    """Write a small mp_20-style npz (packed + indices) for the dataset chain test.

    Field layout follows AsymmetricUnitDataset._parse_flat with NE=98:
    [n, sg, comp(98), lengths(3), angles(3), elems(n), wyckoffs(n),
     frac_coords(3n), wyckoff_shape(n)].
    """
    from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE as NE

    rng = np.random.default_rng(0)
    packed_arrays = []
    indices = []
    offset = 0
    for i in range(num_crystals):
        n = int(rng.integers(1, 5))
        comp = np.zeros(NE, dtype=np.float32)
        comp[0] = 1.0
        lengths = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        angles = np.array([90.0, 90.0, 90.0], dtype=np.float32)
        elems = np.full(n, 1 + int(rng.integers(0, 2)), dtype=np.float32)
        wycks = np.zeros(n, dtype=np.float32)  # P1 has a single wyckoff site
        fracs = rng.random([n, 3]).astype(np.float32)
        wsi = np.zeros(n, dtype=np.float32)
        flat = np.concatenate(
            [
                np.array([n], dtype=np.float32),
                np.array([1.0], dtype=np.float32),  # space group 1 (P1)
                comp,
                lengths,
                angles,
                elems,
                wycks,
                fracs.reshape(-1),
                wsi,
            ]
        ).astype(np.float32)
        packed_arrays.append(flat)
        offset += flat.shape[0]
        # indices holds the split points for the first num_crystals - 1
        # crystals only (matching the real mp_20 npz layout, where
        # np.split(packed, indices) yields exactly num_crystals segments).
        if i < num_crystals - 1:
            indices.append(offset)
    root = Path(root_dir) / "mp_20"
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / f"{split}.npz",
        packed=np.concatenate(packed_arrays),
        indices=np.array(indices, dtype=np.int64),
    )
    return root


def test_dataset_collate_to_model_forward(tmp_path):
    """Data pipeline: npz -> build_dataloader -> collate -> model forward.

    Overall question: does a realistic dataset batch flow from disk through
    AsymmetricUnitDataset / DefaultCollator into SGEQuiDiff.forward without
    key/shape mismatches?
    """
    from ppmat.datasets import build_dataloader

    _write_synthetic_mp20_npz(tmp_path, num_crystals=8)
    loader = build_dataloader(
        {
            "dataset": {
                "__class_name__": "AsymmetricUnitDataset",
                "__init_params__": {
                    "name": "mp_20",
                    "split": "train",
                    "data_directory": str(tmp_path),
                },
            },
            "loader": {"num_workers": 0, "use_shared_memory": False},
            "sampler": {
                "__class_name__": "BatchSampler",
                "__init_params__": {
                    "batch_size": 4,
                    "shuffle": False,
                    "drop_last": False,
                },
            },
        }
    )
    batch = next(iter(loader))
    assert "n_atoms_per_asu" in batch

    sampler = _make_sgequidiff()
    sampler.train()
    out = sampler(batch)
    loss = out["loss_dict"]["loss"]
    assert loss.ndim == 0
    assert not paddle.isnan(loss).item()
    assert float(loss) > 0.0

    # Contract-8: label_dict must expose label keys identical to batch_data.
    label_keys = (
        "space_group_indices",
        "lattice_lengths",
        "lattice_angles",
        "n_atoms_per_asu",
        "element_indices",
        "wyckoff_indices",
        "wyckoff_shape_indices",
        "frac_coords",
    )
    assert "label_dict" in out
    for key in label_keys:
        assert key in out["label_dict"], f"missing label key: {key}"
        assert bool(
            paddle.equal(out["label_dict"][key], batch[key]).all().item()
        ), f"label_dict[{key}] mismatch with batch label"


def test_sgequidiff_metric(tmp_path):
    """Evaluation pipeline: generation-quality metric over synthetic structures.

    Overall question: does SGEQuiDiffMetric return the complete expected
    metric set (validity / uniqueness / novelty / coverage / distances)
    with values in valid ranges? Uses self-contained synthetic CIFs written
    to ``tmp_path``, so the test runs without external data files.
    """
    import pandas as pd
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    from ppmat.metrics.sgequidiff_metric import SGEQuiDiffMetric
    from ppmat.metrics.utils import get_crys_from_cif

    structures = [
        Structure(
            Lattice.from_parameters(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            Lattice.from_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
            ["C", "C"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        Structure(
            Lattice.from_parameters(3.0, 3.0, 3.0, 90.0, 90.0, 90.0),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        Structure(
            Lattice.from_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0),
            ["Fe", "Fe"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
    ]
    cifs = [structure.to(fmt="cif") for structure in structures]
    gt_csv = tmp_path / "gt.csv"
    pd.DataFrame({"cif": cifs}).to_csv(gt_csv, index=False)

    pred_dicts = [get_crys_from_cif(cif).dict for cif in cifs]
    metric = SGEQuiDiffMetric(gt_file_path=str(gt_csv))
    result = metric(pred_dicts)

    for key in (
        "validity",
        "uniqueness",
        "novelty",
        "cov_recall",
        "cov_precision",
        "amsd_recall",
        "amsd_precision",
        "amcd_recall",
        "amcd_precision",
    ):
        assert key in result, f"missing metric: {key}"
    assert 0.0 <= result["validity"] <= 1.0
    assert 0.0 <= result["uniqueness"] <= 1.0
    assert 0.0 <= result["novelty"] <= 1.0

    # Streaming contract: per-step accumulation must match the batch interface.
    assert isinstance(metric, StreamingMetricBase)

    stream_metric = SGEQuiDiffMetric(gt_file_path=str(gt_csv))
    assert stream_metric.compute_epoch(stage="sample") == {}
    assert stream_metric.compute_epoch(stage="eval") == {}
    stream_metric.update_step(
        result={"samples": {"result": pred_dicts[:4]}}, batch=None, stage="sample"
    )
    stream_metric.update_step(
        result={"result": pred_dicts[4:]}, batch=None, stage="sample"
    )
    stream_metric.update_step(result={"result": pred_dicts}, batch=None, stage="eval")
    streamed = stream_metric.compute_epoch(stage="sample")
    for key, value in result.items():
        assert streamed[key] == pytest.approx(value)
    stream_metric.reset()
    assert stream_metric.compute_epoch(stage="sample") == {}
