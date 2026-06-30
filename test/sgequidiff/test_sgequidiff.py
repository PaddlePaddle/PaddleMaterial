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

import pytest
import paddle

from ppmat.models.sgequidiff.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
from ppmat.models.sgequidiff.crystal_data import (
    ASUCrystal,
    is_inside,
    uniformly_sample_point_in_convex_shape,
)
from ppmat.models.sgequidiff.wrappers import SpaceGroupSampler


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
    cfg = EquivariantDiffusionModelConfig(
        model_type="mlp",
        num_timesteps=10,
        num_wn_lattice_translations=1,
        noise_scheduler_num_monte_carlo_samples=10,
        time_emb_dim=32,
        num_plane_wave_freqs=16,
    )
    return EquivariantDiffusionModel(cfg)


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
        cfg = EquivariantDiffusionModelConfig(
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
        model = EquivariantDiffusionModel(cfg)
        assert hasattr(model, "noise_scheduler")
        sigmas = model.noise_scheduler.sigmas
        assert sigmas.shape[0] == N_T + 1  # concat([0], sigmas)

    def test_config_validation_rejects_invalid_models(self):
        for bad_type in ("transformer", "resnet"):
            cfg = EquivariantDiffusionModelConfig(model_type=bad_type)
            with pytest.raises(AssertionError):
                EquivariantDiffusionModel.validate_config(cfg)


class TestCrystalDataPipeline:
    """Integrated test: ASUCrystal construction, hull check, convex sampling."""

    def test_crystal_roundtrip_immutable(self):
        crystal = ASUCrystal(
            space_group_number=paddle.to_tensor(1, dtype=paddle.int64),
            conventional_lattice_lengths=paddle.to_tensor([5.0, 5.0, 5.0]),
            conventional_lattice_angles=paddle.to_tensor([90.0, 90.0, 90.0]),
            element_indices=paddle.to_tensor([1, 2], dtype=paddle.int64),
            wyckoff_indices=paddle.to_tensor([0, 0], dtype=paddle.int64),
            conventional_frac_coords=paddle.to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        )
        assert crystal.num_atoms == 2

        imm = crystal.to_ImmutableASUCrystal()
        assert imm.num_atoms == 2
        back = imm.to_ASUCrystal()
        assert back.num_atoms == 2

    def test_is_inside_unit_cube(self):
        hull = paddle.to_tensor([
            [1, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, -1, 0, 0],
            [0, 0, 1, -1],
            [0, 0, -1, 0],
        ], dtype=paddle.float32)

        inside = is_inside(
            paddle.to_tensor([[0.3, 0.3, 0.3]], dtype=paddle.float32), hull
        )
        outside = is_inside(
            paddle.to_tensor([[2.0, 0.3, 0.3]], dtype=paddle.float32), hull
        )
        assert inside.item()
        assert not outside.item()

    def test_convex_shape_sampling_0d_1d_2d_3d(self):
        # 0D: single point
        p = paddle.to_tensor([[0.5, 0.5, 0.5]], dtype=paddle.float32)
        s0 = uniformly_sample_point_in_convex_shape(p, 0, n_samples=3)
        assert paddle.allclose(s0, p.expand([3, 3]), atol=1e-6)

        # 1D: line segment
        seg = paddle.to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=paddle.float32)
        s1 = uniformly_sample_point_in_convex_shape(seg.unsqueeze(0), 1, n_samples=1)
        assert 0.0 <= s1[0, 0].item() <= 1.0

        # 2D: triangle
        tri = paddle.to_tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=paddle.float32)
        s2 = uniformly_sample_point_in_convex_shape(
            tri.unsqueeze(0).expand([10, 3, 3]), 2, n_samples=10
        )
        for i in range(10):
            assert 0.0 <= s2[i, 0].item() <= 1.0
            assert 0.0 <= s2[i, 1].item() <= 1.0
            assert paddle.abs(s2[i, 2]).item() < 1e-6

        # 3D: unit cube
        cube_verts = paddle.to_tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.],
        ], dtype=paddle.float32)
        cube_hull = paddle.to_tensor([
            [1, 0, 0, -1], [-1, 0, 0, 0],
            [0, 1, 0, -1], [0, -1, 0, 0],
            [0, 0, 1, -1], [0, 0, -1, 0],
        ], dtype=paddle.float32)
        s3 = uniformly_sample_point_in_convex_shape(
            cube_verts, 3, n_samples=10, hull_equations=cube_hull
        )
        for i in range(10):
            assert 0.0 <= s3[i, 0].item() <= 1.0
            assert 0.0 <= s3[i, 1].item() <= 1.0
            assert 0.0 <= s3[i, 2].item() <= 1.0


class TestSpaceGroupSampler:
    """Integrated test: SpaceGroupSampler sampling and log-prob."""

    def test_sample_produces_valid_range(self):
        sampler = SpaceGroupSampler()
        sample, log_prob = sampler.sample_and_log_prob(batch_size=5)
        assert sample.shape == [5]
        assert log_prob.shape == [5]
        assert (sample >= 0).all().item()
        assert (sample < 230).all().item()

    def test_log_prob_consistent_with_sample(self):
        sampler = SpaceGroupSampler()
        sample, log_prob = sampler.sample_and_log_prob(batch_size=10)
        direct_lp = sampler.log_prob(sample)
        assert paddle.allclose(log_prob, direct_lp, atol=1e-5)
