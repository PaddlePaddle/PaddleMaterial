import os
import unittest

import numpy as np
import paddle
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.gegnn_dataset import BinaryActivityDataset
from ppmat.datasets.gegnn_dataset import _MOLECULAR_GRAPH_CFG
from ppmat.datasets.gegnn_dataset import _canonical_atom_feats
from ppmat.datasets.gegnn_dataset import build_molecular_graph
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models.gegnn import GEGNNBinary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGEGNNBinary(unittest.TestCase):
    @staticmethod
    def _synthetic_sample(composition, gamma=None):
        samples = ["CCO", "O"]
        solvent_features = []
        for smiles in samples:
            mol = Chem.MolFromSmiles(smiles)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            solvent_features.append(
                {
                    "graph": build_molecular_graph(
                        mol, build_graph_converter(_MOLECULAR_GRAPH_CFG)
                    ),
                    "hba": hba,
                    "hbd": hbd,
                    "intra_hb": min(hba, hbd),
                }
            )
        solv1, solv2 = solvent_features
        sample = {
            "g1": solv1["graph"],
            "g2": solv2["graph"],
            "x1": composition,
            "x2": 1.0 - composition,
            "intra_hb1": solv1["intra_hb"],
            "intra_hb2": solv2["intra_hb"],
            "inter_hb": min(solv1["hba"], solv2["hbd"])
            + min(solv1["hbd"], solv2["hba"]),
            "empty_solvsys": BinaryActivityDataset.generate_solvsys(1),
        }
        if gamma is not None:
            sample["gamma1"], sample["gamma2"] = gamma
        return sample

    @classmethod
    def _synthetic_batch(cls):
        samples = [
            cls._synthetic_sample(composition, gamma)
            for composition, gamma in ((0.25, [0.10, -0.05]), (0.75, [0.20, 0.15]))
        ]
        return DefaultCollator()(samples)

    @staticmethod
    def _new_model():
        return GEGNNBinary(in_dim=74, hidden_dim=8, n_classes=1)

    def test_instantiation(self):
        self.assertIsNotNone(self._new_model())

    def test_excess_gibbs_formula(self):
        excess_gibbs_energy = paddle.to_tensor([[0.5], [1.0]], dtype="float32")
        x1 = paddle.to_tensor([[0.25], [0.75]], dtype="float32")
        derivative = paddle.to_tensor([[2.0], [-0.5]], dtype="float32")

        gamma = GEGNNBinary.activity_coefficients(excess_gibbs_energy, x1, derivative)

        np.testing.assert_allclose(
            gamma.numpy(),
            np.array([[2.0, 0.0], [0.875, 1.375]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )

    def test_label_free_collation_for_prediction(self):
        batch = DefaultCollator()([self._synthetic_sample(0.5)])

        self.assertNotIn("gamma", batch)
        prediction = self._new_model().predict(batch)
        self.assertEqual(prediction["gamma"].shape, [1, 2])

    def test_forward_loss_and_predict_on_synthetic_batch(self):
        batch = self._synthetic_batch()
        model = self._new_model()

        output = model(batch)
        gamma = output["pred_dict"]["gamma"]
        loss = output["loss_dict"]["loss"]
        model.eval()
        prediction = model.predict(batch)

        self.assertEqual(gamma.shape, [2, 2])
        self.assertIn("supervised_loss", output["loss_dict"])
        self.assertNotIn("gibbs_duhem_loss", output["loss_dict"])
        self.assertTrue(np.isfinite(loss.numpy()).all())
        self.assertIn("excess_gibbs_energy", output["pred_dict"])
        self.assertIn("d_excess_gibbs_energy_dx1", output["pred_dict"])
        self.assertIn("gamma", prediction)
        self.assertEqual(prediction["gamma"].shape, [2, 2])
        self.assertTrue(np.isfinite(prediction["gamma"].numpy()).all())

    def test_optimized_prediction_matches_full_autograd(self):
        batch = self._synthetic_batch()
        model = self._new_model()
        model.eval()

        reference = model._predict_with_derivative(batch, create_graph=False)
        optimized = model._predict_head_gradient(batch)
        for expected, actual in zip(reference, optimized):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=3e-5, atol=3e-6
            )

    def test_two_step_adam_training_is_deterministic(self):
        batch = self._synthetic_batch()
        paddle.seed(2025)
        first_model = self._new_model()
        paddle.seed(2025)
        second_model = self._new_model()
        first_optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=first_model.parameters()
        )
        second_optimizer = paddle.optimizer.Adam(
            learning_rate=1e-3, parameters=second_model.parameters()
        )

        for _ in range(2):
            first_result = first_model(batch)
            second_result = second_model(batch)
            np.testing.assert_allclose(
                first_result["pred_dict"]["gamma"].numpy(),
                second_result["pred_dict"]["gamma"].numpy(),
                rtol=0.0,
                atol=1e-4,
            )

            first_loss = first_result["loss_dict"]["loss"]
            first_loss.backward()
            self.assertTrue(
                any(
                    parameter.grad is not None for parameter in first_model.parameters()
                )
            )
            first_optimizer.step()
            first_optimizer.clear_grad()

            second_loss = second_result["loss_dict"]["loss"]
            second_loss.backward()
            second_optimizer.step()
            second_optimizer.clear_grad()

            np.testing.assert_allclose(
                first_loss.numpy(), second_loss.numpy(), rtol=0.0, atol=1e-6
            )
            for first_parameter, second_parameter in zip(
                first_model.parameters(), second_model.parameters()
            ):
                np.testing.assert_allclose(
                    first_parameter.numpy(),
                    second_parameter.numpy(),
                    rtol=0.0,
                    atol=1e-6,
                )

    def test_build_model_from_config(self):
        config_path = os.path.join(
            BASE_DIR,
            "property_prediction",
            "configs",
            "gegnn",
            "gegnn_binary_activity.yaml",
        )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        self.assertIsInstance(build_model(config["Model"]), GEGNNBinary)

    def test_gegnn_config_contract(self):
        config_path = os.path.join(
            BASE_DIR,
            "property_prediction",
            "configs",
            "gegnn",
            "gegnn_binary_activity.yaml",
        )
        config = OmegaConf.load(config_path)
        init_params = config.Model["__init_params__"]

        self.assertEqual(config.Model["__class_name__"], "GEGNNBinary")
        self.assertNotIn("Loss", config)
        self.assertEqual(init_params["in_dim"], 74)
        scheduler = config.Optimizer["__init_params__"]["lr"]
        self.assertEqual(scheduler["__class_name__"], "ReduceOnPlateau")
        scheduler_params = scheduler["__init_params__"]
        self.assertEqual(scheduler_params["learning_rate"], 1e-3)
        self.assertEqual(scheduler_params["factor"], 0.8)
        self.assertEqual(scheduler_params["patience"], 3)
        self.assertEqual(scheduler_params["min_lr"], 1e-7)
        self.assertEqual(scheduler_params["indicator"], "train_loss")
        self.assertEqual(scheduler_params["indicator_name"], "loss")
        self.assertNotIn("pinn_lambda", init_params)
        self.assertNotIn("finite_difference_step", init_params)
        self.assertFalse(config.Trainer["eval_with_no_grad"])
        self.assertEqual(
            OmegaConf.to_container(config, resolve=False)["Dataset"]["train"][
                "dataset"
            ]["__init_params__"]["path"],
            "./data/binary_activity/output_binary_with_inf_all.csv",
        )
        self.assertNotIn("val", config.Dataset)
        self.assertNotIn("collate_params", config.Dataset["train"]["loader"])


class TestAtomFeaturization(unittest.TestCase):
    def test_canonical_atom_feats(self):
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)
        feats = _canonical_atom_feats(atom)
        self.assertEqual(feats.shape[0], 74)
        self.assertEqual(feats.dtype, np.float32)

    def test_molecular_graph_converter(self):
        mol = Chem.MolFromSmiles("CCO")
        graph = build_molecular_graph(
            mol, build_graph_converter(_MOLECULAR_GRAPH_CFG)
        )
        self.assertIsNotNone(graph)
        self.assertEqual(graph.num_nodes, 3)
        self.assertIn("h", graph.node_feat)
        self.assertEqual(graph.node_feat["h"].shape[1], 74)


if __name__ == "__main__":
    unittest.main()
