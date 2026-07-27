# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
import paddle
from ase import Atoms
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models.deeph import DeepHGraphConverter
from ppmat.models.deeph import DeepHHamiltonian
from ppmat.predictor.deeph import DeepHPredictor


class TestDeepH(unittest.TestCase):
    def test_deeph_config_can_be_loaded(self):
        config_path = os.path.join(
            "electronic_structure",
            "configs",
            "deeph",
            "deeph_graphene.yaml",
        )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        self.assertEqual(config["Model"]["__class_name__"], "DeepHHamiltonian")
        self.assertEqual(
            config["Dataset"]["train"]["dataset"]["__class_name__"],
            "DeepHDataset",
        )
        init_params = config["Dataset"]["train"]["dataset"]["__init_params__"]
        self.assertIn("config", init_params)
        self.assertNotIn("config_files", init_params)
        self.assertEqual(len(init_params["config"]["basic"]["orbital"]), 169)

    def test_deeph_forward(self):
        paddle.seed(42)
        model = DeepHHamiltonian(
            num_species=1,
            in_atom_fea_len=16,
            in_edge_fea_len=32,
            num_orbital=9,
            num_l=3,
            gauss_stop=6.0,
            if_exp=True,
            normalization="LayerNorm",
            target_name="label",
        )

        num_nodes = 3
        num_edges = 4
        num_sub_edges = 8
        batch = {
            "x": paddle.zeros([num_nodes], dtype="int64"),
            "edge_index": paddle.to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64"),
            "edge_attr": paddle.ones([num_edges, 1], dtype="float32"),
            "batch": paddle.zeros([num_nodes], dtype="int64"),
            "sub_atom_idx": paddle.to_tensor(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                ],
                dtype="int64",
            ),
            "sub_edge_idx": paddle.to_tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype="int64"),
            "sub_edge_ang": paddle.zeros([num_sub_edges, 9], dtype="float32"),
            "sub_index": paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype="int64"),
            "label": paddle.zeros([num_edges, 9], dtype="float32"),
            "mask": paddle.ones([num_edges, 9], dtype="bool"),
        }

        output = model(batch)

        self.assertIn("loss", output["loss_dict"])
        self.assertIn("label", output["pred_dict"])
        self.assertEqual(list(output["pred_dict"]["label"].shape), [num_edges, 9])

    def test_deeph_data_uses_default_collator(self):
        sample = {
            "x": ConcatData(np.zeros([2], dtype=np.int64)),
            "edge_index": ConcatData(np.asarray([[0, 1], [1, 0]], dtype=np.int64)),
            "edge_attr": ConcatData(np.ones([2, 1], dtype=np.float32)),
            "batch": ConcatData(np.zeros([2], dtype=np.int64)),
            "label": ConcatData(np.zeros([2, 9], dtype=np.float32)),
            "mask": ConcatData(np.ones([2, 9], dtype=bool)),
            "sub_atom_idx": ConcatData(np.asarray([[0, 1], [1, 0]], dtype=np.int64)),
            "sub_edge_idx": ConcatData(np.asarray([0, 1], dtype=np.int64)),
            "sub_edge_ang": ConcatData(np.zeros([2, 9], dtype=np.float32)),
            "sub_index": ConcatData(np.asarray([0, 1], dtype=np.int64)),
        }

        collated = DefaultCollator()([sample])

        self.assertIsInstance(collated, dict)
        self.assertEqual(collated["x"].tolist(), [0, 0])
        self.assertEqual(collated["sub_atom_idx"].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(collated["sub_edge_idx"].tolist(), [0, 1])
        self.assertEqual(collated["sub_index"].tolist(), [0, 1])

    def test_deeph_prediction_can_export_hamiltonian_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "orbital_types.dat"), "w") as f:
                f.write("0 1\n")
                f.write("0 1\n")

            batch = {
                "edge_index": np.asarray([[0], [1]], dtype=np.int64),
                "edge_attr": np.asarray(
                    [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                "structure_lattice": np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                "structure_atomic_numbers": np.asarray([6, 6], dtype=np.int64),
                "structure_folder": tmpdir,
            }
            pred = np.asarray([[0.25]], dtype=np.float32)
            orbital = [{"6 6": [0, 1]}]
            predictor = DeepHPredictor()

            hamiltonian = predictor.build_hamiltonian_npz(pred, batch, orbital)

            self.assertEqual(list(hamiltonian.keys()), ["[0, 0, 0, 1, 2]"])
            self.assertEqual(hamiltonian["[0, 0, 0, 1, 2]"].shape, (4, 4))
            self.assertAlmostEqual(
                float(hamiltonian["[0, 0, 0, 1, 2]"][0, 1]),
                0.25,
                places=6,
            )

            output_path = os.path.join(tmpdir, "rh_pred.npz")
            predictor.save_hamiltonian_npz(output_path, pred, batch, orbital)
            loaded = np.load(output_path)
            self.assertIn("[0, 0, 0, 1, 2]", loaded.files)
            self.assertAlmostEqual(
                float(loaded["[0, 0, 0, 1, 2]"][0, 1]),
                0.25,
                places=6,
            )

    def test_deeph_graph_converter_accepts_ase_atoms(self):
        atoms = Atoms(
            symbols=["C", "C"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=np.eye(3) * 4.0,
            pbc=True,
        )
        structure = DeepHGraphConverter._to_structure(atoms)

        self.assertIsInstance(structure, Structure)
        self.assertEqual(len(structure), 2)


if __name__ == "__main__":
    unittest.main()
