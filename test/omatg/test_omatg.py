# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle

from ppmat.models.omatg.model import OMATGCSPNetFull
from ppmat.models.omatg.si.core import DEFAULT_MAX_ATOMS


def _make_batch(batch_size=2, atoms_per_struct=(3, 4), max_z=10):
    """Build a minimal OMatG sample dict batch for forward/loss tests."""
    num_list = list(atoms_per_struct[:batch_size])
    total = sum(num_list)
    atom_types = paddle.randint(1, max_z, [total], dtype="int64")
    frac_coords = paddle.rand([total, 3])
    lattices = paddle.rand([batch_size, 3, 3]) * 2.0 + paddle.eye(3).unsqueeze(0)
    num_atoms = paddle.to_tensor(num_list, dtype="int64")
    node2graph = paddle.repeat_interleave(
        paddle.arange(batch_size, dtype="int64"), num_atoms
    )
    return {
        "n_atoms": num_atoms,
        "species": atom_types,
        "cell": lattices,
        "pos": frac_coords,
        "pos_is_fractional": paddle.ones([batch_size], dtype="bool"),
        "batch": node2graph,
        "ptr": paddle.concat(
            [
                paddle.to_tensor([0], dtype="int64"),
                paddle.cumsum(num_atoms, axis=0).cast("int64"),
            ]
        ),
    }


def _make_concat_sample(n):
    """Build a ConcatData-wrapped sample dict (DefaultCollator-friendly)."""
    import numpy as np

    from ppmat.datasets.custom_data_type import ConcatData

    cell = (paddle.eye(3) * 5.0).numpy()
    nums = np.arange(1, n + 1, dtype="int64")
    pos = paddle.rand([n, 3]).numpy()
    return {
        "n_atoms": ConcatData(np.array([n], dtype="int64")),
        "species": ConcatData(nums),
        "cell": ConcatData(cell.reshape(1, 3, 3)),
        "pos": ConcatData(pos),
        "pos_is_fractional": ConcatData(np.array([True], dtype="bool")),
    }


def _csp_si_scheduler_cfg(integration_time_steps=210):
    """Build the SI scheduler config for CSP (Linear-ODE-style).

    Every kwarg of ``SingleStochasticInterpolant.__init__`` is listed
    explicitly (gamma/epsilon/integrator_kwargs ``None``,
    ``correct_center_of_mass_motion`` as a bool) so the config doubles as
    a regression check against silent default drift.
    """
    return {
        "species": {
            "__class_name__": "SingleStochasticInterpolantIdentity",
            "__init_params__": {},
        },
        "pos": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "PeriodicLinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": True,
                "velocity_annealing_factor": 10.18,
            },
        },
        "cell": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "LinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": False,
                "velocity_annealing_factor": 1.82,
            },
        },
        "integration_time_steps": integration_time_steps,
        "relative_si_costs": {
            "species_loss": 0.0,
            "pos_loss_b": 0.9994,
            "cell_loss_b": 0.0006,
        },
    }


def _dng_si_scheduler_cfg(integration_time_steps=710):
    """Build the SI scheduler config for DNG (Linear-SDE + DFM mask)."""
    return {
        "species": {
            "__class_name__": "DiscreteFlowMatchingMask",
            "__init_params__": {"noise": 0.189},
        },
        "pos": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "PeriodicLinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": {
                    "__class_name__": "LatentGammaSqrt",
                    "__init_params__": {"a": 0.018},
                },
                "epsilon": {
                    "__class_name__": "VanishingEpsilon",
                    "__init_params__": {"c": 9.7, "mu": 0.17, "sigma": 0.029},
                },
                "differential_equation_type": "SDE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": True,
                "velocity_annealing_factor": 6.33,
            },
        },
        "cell": {
            "__class_name__": "SingleStochasticInterpolant",
            "__init_params__": {
                "interpolant": {
                    "__class_name__": "LinearInterpolant",
                    "__init_params__": {},
                },
                "gamma": None,
                "epsilon": None,
                "differential_equation_type": "ODE",
                "integrator_kwargs": None,
                "correct_center_of_mass_motion": False,
                "velocity_annealing_factor": 1.07,
            },
        },
        "integration_time_steps": integration_time_steps,
        "relative_si_costs": {
            "species_loss": 0.5918,
            "pos_loss_b": 0.1309,
            "pos_loss_z": 0.2708,
            "cell_loss_b": 0.0065,
        },
    }


class TestOMATGCSPNetFull(unittest.TestCase):
    """CSP / DNG training mode minimal forward and loss tests."""

    def test_csp_forward_loss(self):
        """CSP mode forward returns finite loss_dict."""
        model = OMATGCSPNetFull(
            hidden_dim=64,
            num_layers=2,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=32,
            edge_style="fc",
            pred_type=False,
        )
        output = model(_make_batch())
        self.assertIn("loss_dict", output)
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "loss is NaN")

    def test_dng_forward_loss(self):
        """DNG mode forward returns finite loss_dict including loss_type."""
        model = OMATGCSPNetFull(
            hidden_dim=64,
            num_layers=2,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=32,
            edge_style="fc",
            pred_type=True,
        )
        model.enable_masked_species()
        output = model(_make_batch())
        self.assertIn("loss_type", output["loss_dict"])
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG loss is NaN")


class TestSITrainingPath(unittest.TestCase):
    """SI velocity-matching training path (CSP and DNG)."""

    def _build_csp_si_model(self):
        """Build a small CSP model with SI (Linear-ODE)."""
        from ppmat.models.omatg.model import IndependentSampler
        from ppmat.models.omatg.si.core import build_si_from_cfg

        cfg = _csp_si_scheduler_cfg(integration_time_steps=210)
        si = build_si_from_cfg(cfg)
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=False,
            use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = cfg["relative_si_costs"]
        model.use_si = True
        return model

    def test_csp_si_forward_loss(self):
        """CSP SI training path produces velocity-matching loss."""
        out = self._build_csp_si_model()(_make_batch())
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "SI loss is NaN")

    def test_dng_si_forward_loss(self):
        """DNG SI training path (SDE + gamma + DFM mask) stays finite."""
        from ppmat.models.omatg.model import IndependentSampler
        from ppmat.models.omatg.si.core import build_si_from_cfg

        cfg = _dng_si_scheduler_cfg(integration_time_steps=710)
        si = build_si_from_cfg(cfg)
        sampler = IndependentSampler(
            dataset_name="mp_20", mask_species=True, mirror_species=False
        )
        model = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=True,
            use_si=False,
        )
        model.enable_masked_species()
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = cfg["relative_si_costs"]
        model.use_si = True
        out = model(_make_batch())
        self.assertIn("loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG SDE loss is NaN")


class TestSIFactory(unittest.TestCase):
    """Config-driven SI end-to-end: YAML -> build_model -> forward -> sample."""

    def test_config_driven_dng_si_from_yaml(self):
        """build_model with DNG yaml trains and samples end-to-end."""
        import copy

        from omegaconf import OmegaConf

        from ppmat.datasets.collate_fn import DefaultCollator
        from ppmat.models import build_model

        cfg = OmegaConf.load("structure_generation/configs/omatg/omatg_mp20_dng.yaml")
        model_cfg = OmegaConf.to_container(cfg["Model"], resolve=True)
        init_params = model_cfg["__init_params__"]
        init_params.update(hidden_dim=32, num_layers=1, time_embed_dim=16)
        init_params["si_scheduler_cfg"]["integration_time_steps"] = 10
        model = build_model(copy.deepcopy(model_cfg))
        self.assertTrue(model.use_si)

        batch = DefaultCollator()([_make_concat_sample(3), _make_concat_sample(4)])
        out = model(batch)
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG SI loss is NaN")
        result = model.sample(batch, num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            coords = paddle.to_tensor(entry["frac_coords"])
            self.assertFalse(
                bool(paddle.isnan(coords).any()), "sampled frac_coords contain NaN"
            )


class TestSISampling(unittest.TestCase):
    """SI integrate-based sampling."""

    def test_csp_si_sample(self):
        """CSP SI sampling produces structures via si.integrate."""
        from ppmat.models.omatg.model import IndependentSampler
        from ppmat.models.omatg.si.core import build_si_from_cfg

        cfg = _csp_si_scheduler_cfg(integration_time_steps=10)
        si = build_si_from_cfg(cfg)
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=False,
            use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = cfg["relative_si_costs"]
        model.use_si = True
        result = model.sample(_make_batch(), num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        self.assertIn("frac_coords", result["result"][0])

    def test_dng_si_sample_stable(self):
        """DNG SDE sampling stays finite (lattice cell clipped)."""
        from ppmat.models.omatg.model import IndependentSampler
        from ppmat.models.omatg.si.core import build_si_from_cfg

        cfg = _dng_si_scheduler_cfg(integration_time_steps=10)
        si = build_si_from_cfg(cfg)
        sampler = IndependentSampler(
            dataset_name="mp_20", mask_species=True, mirror_species=False
        )
        model = OMATGCSPNetFull(
            hidden_dim=32,
            num_layers=1,
            max_atoms=DEFAULT_MAX_ATOMS,
            time_embed_dim=16,
            pred_type=True,
            use_si=False,
        )
        model.enable_masked_species()
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = cfg["relative_si_costs"]
        model.use_si = True
        result = model.sample(_make_batch(), num_inference_steps=10)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            coords = paddle.to_tensor(entry["frac_coords"])
            self.assertFalse(
                bool(paddle.isnan(coords).any()), "sampled frac_coords contain NaN"
            )
            lengths = paddle.to_tensor(entry["lengths"])
            self.assertFalse(
                bool(paddle.isnan(lengths).any()), "sampled lengths contain NaN"
            )


if __name__ == "__main__":
    unittest.main()
