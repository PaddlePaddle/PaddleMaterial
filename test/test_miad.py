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

from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.models import build_model
from ppmat.models.miad.collate import MiADCollator  # noqa: F401


class MiADConfigTest(unittest.TestCase):
    """Test that MiAD config can be parsed and model constructed."""

    def test_config_load(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        self.assertIn("Model", config)
        self.assertEqual(config["Model"]["__class_name__"], "MiAD")
        self.assertIn("diffusion_cfg", config["Model"]["__init_params__"])
        self.assertIn("model_cfg", config["Model"]["__init_params__"])

    def test_model_construction(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        model = build_model(config["Model"])
        self.assertIsNotNone(model)
        self.assertIsInstance(model, paddle.nn.Layer)

    def test_model_forward(self):
        config = OmegaConf.load("structure_generation/configs/miad/miad_mp20.yaml")
        config = OmegaConf.to_container(config, resolve=True)
        model = build_model(config["Model"])
        model.eval()

        batch_size = 2
        num_atoms = paddle.to_tensor([5, 7], dtype="int64")
        total_atoms = int(num_atoms.sum())
        batch_idx = paddle.concat(
            [paddle.full([int(n)], i, dtype="int64") for i, n in enumerate(num_atoms)]
        )
        lattices = paddle.randn([batch_size, 3, 3], dtype="float32")
        frac_coords = paddle.rand([total_atoms, 3], dtype="float32")
        atom_types = paddle.randint(1, 10, [total_atoms], dtype="int64")

        batch = {
            "x0": [lattices, frac_coords, atom_types],
            "batch_size": batch_size,
            "num_atoms": num_atoms,
            "batch_idx": batch_idx,
            "atom_types": atom_types,
        }

        with paddle.no_grad():
            output = model(batch)

        self.assertIn("loss_dict", output)
        self.assertIn("loss", output["loss_dict"])


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
        collator = MiADCollator()
        samples = [self.dataset[i] for i in range(min(4, len(self.dataset)))]
        batch = collator(samples)
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
