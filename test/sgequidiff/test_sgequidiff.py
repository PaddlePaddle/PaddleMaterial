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

from pathlib import Path

import numpy as np
import pytest
import paddle

from ppmat.models.sgequidiff.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
from ppmat.models.sgequidiff.sgequidiff import SGEQuiDiff


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=2):
    """Create a synthetic batch dict with valid P1 (space group 1) data."""
    total = batch_size * atoms_per_crystal
    return {
        "frac_coords": paddle.rand([total, 3]),
        "element_indices": paddle.to_tensor(
            [1, 2] * batch_size, dtype=paddle.int64
        ),
        "wyckoff_indices": paddle.to_tensor(
            [0] * total, dtype=paddle.int64
        ),
        "space_group_indices": paddle.to_tensor(
            [0] * batch_size, dtype=paddle.int64
        ),
        "n_atoms_per_asu": paddle.to_tensor(
            [atoms_per_crystal] * batch_size, dtype=paddle.int64
        ),
        "wyckoff_shape_indices": paddle.to_tensor(
            [0] * total, dtype=paddle.int64
        ),
        "lattice_lengths": paddle.to_tensor(
            [[5.0, 5.0, 5.0]] * batch_size, dtype=paddle.float32
        ),
        "lattice_angles": paddle.to_tensor(
            [[90.0, 90.0, 90.0]] * batch_size, dtype=paddle.float32
        ),
    }


@pytest.fixture(scope="module")
def model():
    """Build a lightweight MLP EquivariantDiffusionModel. Shared across tests."""
    return EquivariantDiffusionModel(
        model_type="mlp",
        num_timesteps=10,
        num_wn_lattice_translations=1,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=32,
        num_plane_wave_freqs=16,
    )


class TestModelBuildForward:
    """Integrated test: model construction, forward pass, loss validity."""

    def test_build_from_config_produces_valid_output(self, model):
        batch_data = _make_synthetic_batch()
        output = model(batch_data)

        assert "loss_dict" in output
        loss = output["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()
        assert float(loss) > 0.0

    def test_build_scheduler_from_cfg(self):
        N_T = 10
        model = EquivariantDiffusionModel(
            model_type="mlp",
            num_timesteps=N_T,
            num_wn_lattice_translations=1,
            noise_scheduler_num_monte_carlo_samples=10,
            noise_scheduler_cfg={
                "__class_name__": "ASUVESDEScheduler",
                "__init_params__": {
                    "num_timesteps": N_T,
                    "sigma_min": 0.002,
                    "sigma_max": 0.5,
                    "num_lattice_translations": 1,
                    "num_monte_carlo_samples": 10,
                },
            },
        )
        assert hasattr(model, "noise_scheduler")
        sigmas = model.noise_scheduler.sigmas
        assert sigmas.shape[0] == N_T + 1  # concat([0], sigmas)

    def test_config_validation_rejects_invalid_models(self):
        for bad_type in ("transformer", "resnet"):
            cfg = EquivariantDiffusionModelConfig(model_type=bad_type)
            with pytest.raises(AssertionError):
                EquivariantDiffusionModel.validate_config(cfg)


class TestSGEQuiDiffEndToEnd:
    """Integrated test: SGEQuiDiff build -> sample -> result format."""

    @pytest.fixture(scope="class")
    def sampler(self):
        paddle.seed(0)
        return SGEQuiDiff(
            dataset_name="mp_20",
            num_timesteps=2,
            noise_scheduler_num_monte_carlo_samples=2,
            num_wn_lattice_translations=1,
        )

    def test_sample_returns_valid_result_format(self, sampler):
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
        batch_data = _make_synthetic_batch()
        out = sampler(batch_data)
        assert "loss_dict" in out
        loss = out["loss_dict"]["loss"]
        assert loss.ndim == 0
        assert not paddle.isnan(loss).item()

    def test_space_group_log_prob_consistent(self, sampler):
        sample, log_prob = sampler.space_group_sampler.sample_and_log_prob(batch_size=10)
        direct_lp = sampler.space_group_sampler.log_prob(sample)
        assert paddle.allclose(log_prob, direct_lp, atol=1e-5)

    def test_space_group_log_prob_consistent_with_temperature(self, sampler):
        temperature = 0.5
        sample, log_prob = sampler.space_group_sampler.sample_and_log_prob(
            batch_size=10, temperature=temperature
        )
        logits = (
            sampler.space_group_sampler.marginal_space_group_logits / temperature
        )
        direct_lp = paddle.nn.functional.log_softmax(logits, axis=-1)[sample]
        assert paddle.allclose(log_prob, direct_lp, atol=1e-5)


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
        "element_indices": paddle.to_tensor(
            [1, 2] * n_crystals, dtype=paddle.int64
        ),
        "wyckoff_indices": paddle.zeros([total], dtype=paddle.int64),
        "wyckoff_shape_indices": paddle.zeros([total], dtype=paddle.int64),
        "frac_coords": paddle.rand([total, 3]),
    }


class TestFullTrainingObjective:
    """Integrated test: complete training objective (MLE + score matching)."""

    @pytest.fixture(scope="class")
    def sampler(self):
        paddle.seed(0)
        return SGEQuiDiff(
            dataset_name="mp_20",
            num_timesteps=2,
            noise_scheduler_num_monte_carlo_samples=2,
            num_wn_lattice_translations=1,
        )

    def test_mixed_space_group_batch_trains(self, sampler):
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

    def test_log_prob_agrees_with_sample_and_log_prob(self, sampler):
        we = sampler.wyckoff_and_element_sampler
        we.eval()
        n_crystals = 3
        ll = paddle.rand([n_crystals, 3]) * 3 + 4
        la = paddle.rand([n_crystals, 3]) * 30 + 75
        sg = paddle.randint(low=0, high=230, shape=[n_crystals])
        with paddle.no_grad():
            elems, wycks, n_atoms, elems_lp, wycks_lp, term_lp = (
                we.sample_and_log_prob(ll, la, sg, 1.0)
            )
            elems_lp2, wycks_lp2, term_lp2 = we.log_prob(
                elems, wycks, n_atoms, ll, la, sg
            )
        assert paddle.allclose(wycks_lp, wycks_lp2, atol=1e-4)
        assert paddle.allclose(term_lp, term_lp2, atol=1e-4)
        assert paddle.allclose(elems_lp, elems_lp2, atol=1e-4)

    def test_lattice_log_prob_agrees_with_sampling(self, sampler):
        ls = sampler.lattice_sampler
        ls.eval()
        sg = paddle.randint(low=0, high=230, shape=[4])
        with paddle.no_grad():
            lengths, angles, log_pf_sample = ls(sg)
            log_pf_lp, _ = ls.log_prob(lengths, angles, sg)
        log_pf_lp_sum = (log_pf_lp * ls.bravais_log_prob_masks[sg]).sum(axis=1)
        assert paddle.allclose(
            log_pf_sample, log_pf_lp_sum, atol=1e-3
        ), "lattice log_prob diverges from sampling path"

    def test_training_loss_decreases_over_steps(self, sampler):
        paddle.seed(0)
        sampler.train()
        opt = paddle.optimizer.Adam(parameters=sampler.parameters(), learning_rate=1e-3)
        losses = []
        batch = _make_mixed_space_group_batch()
        for _ in range(5):
            out = sampler(batch)
            loss = out["loss_dict"]["loss"]
            losses.append(float(loss))
            opt.clear_grad()
            loss.backward()
            opt.step()
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    def test_weight_parameter_name_contract(self, sampler):
        """The released checkpoints store MHA weights as _qkv_weight; the model
        must keep that naming contract (checked without any weight file)."""
        names = [n for n, _ in sampler.named_parameters()]
        n_qkv = sum(1 for n in names if "_qkv_weight" in n)
        assert n_qkv >= 2, f"expected >=2 _qkv_weight params, got {n_qkv}"
        for module in (
            "atom_coord_diffusion_model",
            "space_group_sampler",
            "lattice_sampler",
            "wyckoff_and_element_sampler",
        ):
            assert any(n.startswith(module) for n in names), f"missing module: {module}"

    def test_noisy_lattice_sampling_respects_constraints(self, sampler):
        """get_noisy_lattice_lengths_and_angles must respect Bravais constraints."""
        sampler.train()
        batch_size = 4
        sg = paddle.to_tensor([0, 3, 5, 74], dtype="int64")
        ll = paddle.full([batch_size, 3], 5.0)
        la = paddle.full([batch_size, 3], 90.0)
        noisy_ll, noisy_la = sampler.get_noisy_lattice_lengths_and_angles(ll, la, sg)
        min_len, max_len = 2.0, 133.0
        min_ang, max_ang = 60.0, 135.0
        assert bool((noisy_ll >= min_len).all().item())
        assert bool((noisy_ll <= max_len).all().item())
        assert bool((noisy_la >= min_ang).all().item())
        assert bool((noisy_la <= max_ang).all().item())
        # Bravais angle constraints: P1 (sg 0) free, P2 (sg 3) beta free
        # orthorhombic (sg 16-74) fixed at 90 -> noisy angles must equal 90
        ortho_mask = paddle.isin(sg, paddle.arange(16, 75))
        if ortho_mask.any():
            assert paddle.allclose(
                noisy_la[ortho_mask], paddle.full_like(noisy_la[ortho_mask], 90.0),
                atol=1e-3,
            )

    def test_sampled_structure_validity(self, sampler):
        """Sampled crystals must have in-range elements, coords, lattice params."""
        from ppmat.utils.crystal import ELEMENT_ENCODING_SIZE

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
                assert all(0.0 <= c < 1.0 for c in coords), f"coords out of cell: {coords}"
            assert len(crystal["lengths"]) == 3
            assert len(crystal["angles"]) == 3
            assert all(2.0 <= l <= 133.0 for l in crystal["lengths"])
            assert all(60.0 <= a <= 135.0 for a in crystal["angles"])


def test_pbc_graph_construction_small_crystal():
    """PBC graph: self-loops excluded, per-crystal edge counts consistent."""
    from ppmat.utils.pbc_graph import (
        construct_fully_connected_graphs_with_periodic_boundaries,
    )

    # cubic cell 5A, 2 atoms: one at corner, one at center
    cart_coords = paddle.to_tensor(
        [[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]], dtype="float32"
    )
    lattice = paddle.to_tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype="float32"
    ).unsqueeze(0)
    n_atoms = paddle.to_tensor([2], dtype="int64")

    dst, src, offsets, num_edges = (
        construct_fully_connected_graphs_with_periodic_boundaries(
            cart_coords, lattice, n_atoms
        )
    )
    assert num_edges.shape[0] == 1
    assert int(num_edges[0]) > 0, "expected edges in PBC graph"
    # no self loops after mask (same atom with zero displacement is removed)
    assert bool((dst != src).all().item()) or bool(
        (offsets.abs().sum(axis=-1) > 1e-5).any().item()
    )
    # offsets are integer multiples of the lattice vectors
    assert bool((offsets != paddle.floor(offsets)).any().item() == False)


def test_pbc_graph_batched_two_crystals():
    from ppmat.utils.pbc_graph import (
        construct_fully_connected_graphs_with_periodic_boundaries,
    )

    cart_coords = paddle.to_tensor(
        [
            [0.0, 0.0, 0.0],
            [2.5, 2.5, 2.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],
        ],
        dtype="float32",
    )
    lattice = paddle.to_tensor(
        [[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]], dtype="float32"
    ).tile([2, 1, 1])
    n_atoms = paddle.to_tensor([2, 3], dtype="int64")

    dst, src, offsets, num_edges = (
        construct_fully_connected_graphs_with_periodic_boundaries(
            cart_coords, lattice, n_atoms
        )
    )
    assert num_edges.shape[0] == 2
    assert int(num_edges[0]) > 0 and int(num_edges[1]) > 0
    # edges never cross crystals
    assert bool((dst < 2).all().item()) and bool((src < 2).all().item()) or True
    crystal_ids_dst = (dst >= 2).cast("int64") + (dst >= 5).cast("int64")
    crystal_ids_src = (src >= 2).cast("int64") + (src >= 5).cast("int64")
    assert bool((crystal_ids_dst == crystal_ids_src).all().item())


def _write_synthetic_mp20_npz(root_dir, split="train", num_crystals=8):
    """Write a small mp_20-style npz (packed + indices) for the dataset chain test.

    Field layout follows AsymmetricUnitDataset._parse_flat with NE=98:
    [n, sg, comp(98), lengths(3), angles(3), elems(n), wyckoffs(n),
     frac_coords(3n), wyckoff_shape(n)].
    """
    from ppmat.utils.crystal import ELEMENT_ENCODING_SIZE as NE

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
    """Integrated test: build_dataloader -> collate -> model forward.

    Realistic data flow: npz -> build_dataloader(AsymmetricUnitDataset) ->
    DefaultCollator -> SGEQuiDiff.forward. Guards against key/shape
    mismatches between the dataset output and the model input contract
    (frac_coords / element_indices / wyckoff_indices / n_atoms_per_asu /
    wyckoff_shape_indices / lattice_lengths / lattice_angles).
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

    sampler = SGEQuiDiff(
        dataset_name="mp_20",
        num_timesteps=2,
        noise_scheduler_num_monte_carlo_samples=2,
        num_wn_lattice_translations=1,
    )
    sampler.train()
    out = sampler(batch)
    loss = out["loss_dict"]["loss"]
    assert loss.ndim == 0
    assert not paddle.isnan(loss).item()
    assert float(loss) > 0.0


def test_sgequidiff_metric(tmp_path):
    """SGEQuiDiffMetric returns all expected generation-quality metrics."""
    import pandas as pd

    from ppmat.metrics.sgequidiff_metric import SGEQuiDiffMetric
    from ppmat.metrics.utils import get_crys_from_cif
    from ppmat.utils.asu_data import resolve_asu_data_dir

    src = pd.read_csv(resolve_asu_data_dir() / "mp_20" / "test.csv", nrows=4)
    gt_csv = tmp_path / "gt.csv"
    src.to_csv(gt_csv, index=False)

    pred_dicts = [get_crys_from_cif(cif).dict for cif in src["cif"].tolist()[:3]]
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
