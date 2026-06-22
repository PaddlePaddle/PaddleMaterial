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
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.models import build_model
from ppmat.models.miad.miad import _extract_x0
from ppmat.models.miad.miad import MiAD

TINY_MODEL_CFG = {
    "hidden_dim": 64,
    "latent_dim": 32,
    "num_layers": 2,
    "max_atoms": 100,
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
    "lat_diffusion": {"method": "ddpm", "scheduler": "diffcsp_cosine"},
    "frac_diffusion": {"method": "wrapped_normal", "scheduler": "default_wrapped_normal"},
    "type_diffusion": {"method": "d3pm", "scheduler": "default_d3pm"},
}


def _make_fake_batch(batch_size=2, atoms_per_crystal=5):
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


class MiADSmokeTest(unittest.TestCase):
    """Self-contained smoke tests for MiAD (no weights, no external files)."""

    @classmethod
    def setUpClass(cls):
        paddle.seed(42)

    def test_forward_smoke(self):
        model = MiAD(model_cfg=TINY_MODEL_CFG, diffusion_cfg=TINY_DIFFUSION_CFG)
        model.eval()
        batch = _make_fake_batch()
        with paddle.no_grad():
            output = model(batch)
        self.assertIn("loss_dict", output)
        self.assertIn("loss", output["loss_dict"])
        loss = output["loss_dict"]["loss"]
        self.assertTrue(paddle.isfinite(loss))

    def test_sample_output_format(self):
        model = MiAD(model_cfg=TINY_MODEL_CFG, diffusion_cfg=TINY_DIFFUSION_CFG)
        model.eval()
        batch_data = {"num_atoms": paddle.to_tensor([5, 7], dtype="int64")}
        result = model.sample(batch_data, num_inference_steps=5)
        self.assertIn("result", result)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            for key in ("num_atoms", "atom_types", "frac_coords", "lattice"):
                self.assertIn(key, entry)
            self.assertEqual(entry["frac_coords"].shape[-1], 3)


class MiADConfigTest(unittest.TestCase):
    """Test MiAD construction via build_model from yaml config."""

    def test_config_load(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        self.assertIn("Model", config)
        self.assertEqual(config["Model"]["__class_name__"], "MiAD")
        self.assertIn("diffusion_cfg", config["Model"]["__init_params__"])
        self.assertIn("model_cfg", config["Model"]["__init_params__"])

    def test_build_model_path(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        model = build_model(config["Model"])
        self.assertIsInstance(model, MiAD)
        self.assertIsInstance(model, paddle.nn.Layer)

    def test_train_path(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        model = build_model(config["Model"])
        model.eval()
        batch = _make_fake_batch()
        with paddle.no_grad():
            output = model(batch)
        self.assertIn("loss_dict", output)
        self.assertIn("loss", output["loss_dict"])
        self.assertTrue(paddle.isfinite(output["loss_dict"]["loss"]))

    def test_sample_path(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        model = build_model(config["Model"])
        model.eval()
        batch_data = {"num_atoms": paddle.to_tensor([5, 7], dtype="int64")}
        result = model.sample(batch_data, num_inference_steps=5)
        self.assertIn("result", result)
        self.assertEqual(len(result["result"]), 2)
        for entry in result["result"]:
            self.assertIn("num_atoms", entry)
            self.assertIn("lattice", entry)


class MiADDatasetTest(unittest.TestCase):
    """Dataset smoke test for MiAD."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = MP20Dataset(
            path="./data/mp_20/test.csv",
            build_structure_cfg={"format": "cif_str", "num_cpus": 1},
        )

    def test_dataset_load(self):
        self.assertGreater(len(self.dataset), 0)

    def test_dataset_sample_fields(self):
        sample = self.dataset[0]
        self.assertIn("structure_array", sample)
        sa = sample["structure_array"]
        self.assertIn("frac_coords", sa)
        self.assertIn("atom_types", sa)
        self.assertIn("lattice", sa)
        self.assertIn("num_atoms", sa)

    def test_collate_fn(self):
        collator = DefaultCollator()
        samples = [self.dataset[i] for i in range(min(4, len(self.dataset)))]
        batch = collator(samples)
        batch = _extract_x0(batch)
        self.assertIn("x0", batch)
        self.assertIn("batch_size", batch)
        self.assertIn("num_atoms", batch)
        self.assertIn("batch_idx", batch)
        x0 = batch["x0"]
        self.assertEqual(len(x0), 3)
        self.assertEqual(x0[0].ndim, 3)
        self.assertEqual(x0[1].ndim, 2)
        self.assertEqual(x0[2].ndim, 1)
        self.assertEqual(x0[0].shape[-1], 3)
        self.assertEqual(x0[1].shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
