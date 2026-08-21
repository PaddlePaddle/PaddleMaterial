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

import os
import unittest
from pathlib import Path

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.models import build_model
from ppmat.models.miad.miad import MiAD

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MP20_TEST_CSV = str(_PROJECT_ROOT / "data" / "mp_20" / "test.csv")
_MIAD_YAML = str(
    _PROJECT_ROOT / "structure_generation" / "configs" / "miad" / "miad_mp20.yaml"
)


def setUpModule():
    paddle.seed(42)


def _load_miad_yaml():
    config = OmegaConf.load(_MIAD_YAML)
    return OmegaConf.to_container(config, resolve=True)


def _make_model_from_yaml():
    return build_model(_load_miad_yaml()["Model"])


TINY_MODEL_CFG = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "num_layers": 2,
    "max_atoms": 100,
    "mirage_num_atoms": 8,
    "act_fn": "silu",
    "dis_emb": "sin",
    "num_freqs": 10,
    "edge_style": "fc",
    "ln": False,
    "ip": True,
    "smooth": True,
    "pred_type": True,
}

TINY_DIFFUSION_CFG = {
    "method": "DiffCSP",
    "task": "gen_mp20",
    "cont_time": False,
    "num_steps": 10,
    "time_embed_dim": 32,
    "lat_diffusion": {
        "method": "ddpm",
        "scheduler_cfg": {
            "__class_name__": "DDPMScheduler",
            "__init_params__": {
                "num_train_timesteps": 10,
                "beta_schedule": "diffcsp_cosine",
            },
        },
    },
    "frac_diffusion": {
        "method": "wrapped_normal",
        "step_lr": 1e-5,
        "scheduler_cfg": {
            "__class_name__": "ScoreSdeVeSchedulerWrapped",
            "__init_params__": {
                "num_train_timesteps": 10,
                "sigma_min": 0.005,
                "sigma_max": 0.5,
                "sampling_eps": 0.001,
            },
        },
    },
    "type_diffusion": {
        "__class_name__": "D3PM",
        "__init_params__": {
            "loss_scale": 1000,
            "scheduler_cfg": {
                "__class_name__": "D3PMUniformScheduler",
                "__init_params__": {
                    "num_train_timesteps": 10,
                    "num_types": 100,
                },
            },
        },
    },
}


def _make_synthetic_batch(batch_size=2, atoms_per_crystal=5):
    num_atoms = paddle.full([batch_size], atoms_per_crystal, dtype="int64")
    total_atoms = batch_size * atoms_per_crystal
    batch_idx = paddle.concat(
        [paddle.full([atoms_per_crystal], i, dtype="int64") for i in range(batch_size)]
    )
    lattices = paddle.randn([batch_size, 3, 3], dtype="float32")
    frac_coords = paddle.rand([total_atoms, 3], dtype="float32")
    atom_types = paddle.randint(1, 10, [total_atoms], dtype="int64")
    return {
        "x0": [lattices, frac_coords, atom_types],
        "batch_size": batch_size,
        "num_atoms": num_atoms,
        "batch_idx": batch_idx,
        "atom_types": atom_types,
    }


_SAMPLE_NUM_ATOMS = (5, 7)
_SAMPLE_INFERENCE_STEPS = 5


def _make_tiny_model():
    return MiAD(model_cfg=TINY_MODEL_CFG, diffusion_cfg=TINY_DIFFUSION_CFG)


def _sample_small_batch(model, num_inference_steps=_SAMPLE_INFERENCE_STEPS):
    return model.sample(
        {"num_atoms": paddle.to_tensor(_SAMPLE_NUM_ATOMS, dtype="int64")},
        num_inference_steps=num_inference_steps,
    )["result"]


class MiADSmokeTest(unittest.TestCase):
    """End-to-end model flows: forward (eval/train) and sampling."""

    def test_forward_smoke(self):
        # Eval forward with pre-built x0.
        model = _make_tiny_model()
        model.eval()
        with paddle.no_grad():
            output = model(_make_synthetic_batch())
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))
        # Train forward with raw structure_array (mirage infusion pads + masks)
        model.train()
        train_num_atoms = paddle.to_tensor([5, 3], dtype="int64")
        total_atoms = int(train_num_atoms.sum())
        batch = {
            "structure_array": {
                "num_atoms": train_num_atoms,
                "frac_coords": paddle.rand([total_atoms, 3], dtype="float32"),
                "atom_types": paddle.randint(1, 10, [total_atoms], dtype="int64"),
                "lattice": paddle.randn([2, 3, 3], dtype="float32"),
            }
        }
        loss = model(batch)["loss_dict"]["loss"]
        self.assertTrue(paddle.isfinite(loss))

    def test_sample_output_format(self):
        model = _make_tiny_model()
        model.eval()
        result = _sample_small_batch(model)
        self.assertEqual(len(result), 2)
        for entry in result:
            for key in ("num_atoms", "atom_types", "frac_coords", "lattice"):
                self.assertIn(key, entry)
            self.assertEqual(entry["frac_coords"].shape[-1], 3)
            # Mirage atoms (type 0) must be filtered out of the output.
            self.assertEqual(entry["num_atoms"], entry["atom_types"].shape[0])
            self.assertTrue((entry["atom_types"] != 0).all())


class MiADConfigTest(unittest.TestCase):
    """End-to-end flow driven by the released YAML config."""

    def test_forward_and_sample_from_yaml(self):
        model = _make_model_from_yaml()
        model.eval()
        with paddle.no_grad():
            output = model(_make_synthetic_batch())
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))
        result = _sample_small_batch(model)
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertIn("num_atoms", entry)
            self.assertIn("lattice", entry)


@unittest.skipUnless(
    os.path.exists(_MP20_TEST_CSV),
    f"MP-20 test data not found at {_MP20_TEST_CSV}; skipping dataset tests",
)
class MiADDatasetTest(unittest.TestCase):
    """Real-data pipeline: dataset -> collate -> model forward."""

    def test_collate_to_forward(self):
        dataset = MP20Dataset(
            path=_MP20_TEST_CSV,
            build_structure_cfg={"format": "cif_str", "num_cpus": 1},
        )
        self.assertGreater(len(dataset), 0)
        collator = DefaultCollator()
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collator(samples)
        model = _make_tiny_model()
        model.eval()
        with paddle.no_grad():
            output = model(batch)
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))


class MiADStateDictTest(unittest.TestCase):
    """Official checkpoint layout compatibility, end to end."""

    def test_official_layout_weight_load(self):
        # Empirical layout facts of the MiAD 70-key state_dict built from the
        # current YAML: no "decoder." prefix, PyTorch (out, in) Linear weights,
        # no prop_mlp. The official miad_mp20 checkpoint URL is not yet wired
        # into MODEL_REGISTRY, so only layout round-trip is asserted here.
        model = _make_model_from_yaml()
        sd = model.state_dict()
        self.assertEqual(len(sd), 70)
        self.assertFalse(any("prop_mlp" in k for k in sd.keys()))
        official = {}
        for k, v in sd.items():
            if k.startswith("decoder."):
                name = k[len("decoder."):]
                if name.endswith(".weight") and len(v.shape) == 2:
                    v = v.T
                official[name] = v
        model2 = _make_model_from_yaml()
        missing, unexpected = model2.set_state_dict(official)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)
        params1 = list(model.named_parameters())
        params2 = list(model2.named_parameters())
        self.assertEqual(len(params1), len(params2))
        for (n1, p1), (n2, p2) in zip(params1, params2):
            self.assertTrue(
                np.allclose(p1.numpy(), p2.numpy()),
                f"parameter {n1} differs after official-layout load",
            )
        model2.eval()
        with paddle.no_grad():
            output = model2(_make_synthetic_batch())
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))


class MiADSUNMetricTest(unittest.TestCase):
    """S.U.N. evaluation driven by the YAML metric config."""

    def test_sun_metric_from_yaml(self):
        from ppmat.metrics import build_metric

        metrics_fn = build_metric(_load_miad_yaml()["Sample"]["metrics"])
        model = _make_tiny_model()
        model.eval()
        structures = _sample_small_batch(model)
        results = metrics_fn(structures)
        for key in (
            "total", "valid", "non_trivial",
            "stability_rate", "uniqueness_rate", "novelty_rate",
            "sun_rate", "sun_count",
        ):
            self.assertIn(key, results)
        for key in (
            "stability_rate", "uniqueness_rate", "novelty_rate", "sun_rate",
        ):
            self.assertTrue(np.isfinite(results[key]), f"{key} must be finite")


if __name__ == "__main__":
    unittest.main()
