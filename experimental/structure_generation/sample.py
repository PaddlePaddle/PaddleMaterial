import argparse
import os
import os.path as osp
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))  # ruff: noqa

from typing import Optional

import numpy as np
import paddle
from omegaconf import OmegaConf
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter

from ppmat.datasets import build_dataloader
from ppmat.datasets.build_structure import BuildStructure
from ppmat.metrics import build_metric
from ppmat.models import MODEL_NAMES
from ppmat.models import build_model
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load


class StructureSampler:
    def __init__(
        self,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        # if model_name is not None, then config_path and checkpoint_path must be provided
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is None."
        else:
            logger.info("Since model_name is given, downloading it...")
            path = download.get_weights_path_from_url(MODEL_NAMES[model_name])
            path = osp.join(path, model_name)

            config_path = osp.join(path, f"{model_name}.yaml")

            if "best.pdparams" in os.listdir(path):
                checkpoint_path = osp.join(path, "checkpoints", "best.pdparams")
            elif "latest.pdparams" in os.listdir(path):
                checkpoint_path = osp.join(path, "checkpoints", "latest.pdparams")
            else:
                checkpoint_path = None
                for file in os.listdir(osp.join(path, "checkpoints")):
                    if file.endswith(".pdparams"):
                        checkpoint_path = osp.join(path, "checkpoints", file)

                assert (
                    checkpoint_path is not None
                ), "Do not find any .pdparams files under directory {}".format(
                    osp.join(path, "checkpoints")
                )
        logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)

        self.config = config

        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        self.model_config = model_config

        self.model_name = model_name
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        self.model = build_model(model_config)
        save_load.load_pretrain(self.model, checkpoint_path)
        self.model.eval()

        # sample config
        sample_config = config.get("Sample", None)
        self.sample_config = sample_config

    def compute_metric(
        self,
        save_path=None,
    ):
        dataset_cfg = self.sample_config["data"]
        data_loader = build_dataloader(dataset_cfg)

        build_structure_cfg = self.sample_config["build_structure_cfg"]
        structure_converter = BuildStructure(**build_structure_cfg)

        metrics_cfg = self.sample_config.get("metrics")
        assert metrics_cfg is not None, "metrics config must be provided."
        metrics_fn = build_metric(metrics_cfg)

        logger.info(f"Total iterations: {len(data_loader)}")
        logger.info("Start sampling process...\n")

        total_results = []
        for iter_id, batch_data in enumerate(data_loader):
            pred_data = self.model.sample(batch_data)
            structures = structure_converter(pred_data["result"])
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                for i, structure in enumerate(structures):
                    formula = structure.formula.replace(" ", "-")
                    tar_file = os.path.join(
                        save_path, f"{formula}_{iter_id + 1}_{i + 1}.cif"
                    )
                    if structure is not None:
                        writer = CifWriter(structure)
                        writer.write_file(tar_file)
                    else:
                        logger.info(
                            f"No structure generated for iteration {iter_id}, index {i}"
                        )
            total_results.extend(pred_data["result"])
        metric = metrics_fn(total_results)
        return metric

    def sample(self, data, save_path=None, sample_params=None):
        if sample_params is None:
            sample_params = {}
        assert isinstance(sample_params, dict), "sample_params must be a dict or None."
        pred_data = self.model.sample(data, **sample_params)
        return pred_data

    def sample_by_num_atoms(self, num_atoms, save_path=None, sample_params=None):
        # todo: implement this function
        pass

    def sample_by_chemical_formula(
        self, chemical_formula, save_path=None, sample_params=None
    ):
        assert isinstance(chemical_formula, str), "chemical_formula must be a string."
        composition = Composition(chemical_formula)
        atom_types = []
        for elem, num in composition.items():
            atom_types.extend([elem.Z] * int(num))
        atom_types = np.array(atom_types).astype("int64")

        data = {
            "structure_array": {
                "atom_types": paddle.to_tensor(atom_types),
                "num_atoms": paddle.to_tensor(
                    np.array([atom_types.shape[0]]).astype("int64")
                ),
            }
        }

        result = self.sample(data, save_path=save_path, sample_params=sample_params)
        return result

    def sample_by_condition(self, composition, save_path=None, sample_params=None):
        # todo: implement this function
        pass


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument(
        "--config_path",
        type=str,
        default="./structure_generation/configs/diffcsp/diffcsp_mp20.yaml",
        help="Path to the configuration file.",
    )
    argparse.add_argument(
        "--checkpoint_path",
        type=str,
        default="./output/diffcsp_mp20/checkpoints/latest.pdparams",
        help="Path to the checkpoint file.",
    )
    argparse.add_argument("--save_path", type=str, default="results")
    argparse.add_argument("--chemical_formula", type=str, default="LiMnO2")
    args = argparse.parse_args()

    sampler = StructureSampler(
        model_name=args.model_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    result = sampler.sample_by_chemical_formula(
        chemical_formula=args.chemical_formula,
        save_path=args.save_path,
    )
    print(result)

    result = sampler.compute_metric(save_path=args.save_path)
    print(result)
