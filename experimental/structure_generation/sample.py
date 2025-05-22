import argparse
import os
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
from ppmat.datasets.transform import build_post_transforms
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


class StructureSampler:
    """Structure Sampler.

    This class provides an interface for sampling structures using pre-trained deep
    learning models. Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        # if model_name is not None, then config_path and checkpoint_path must be provided
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is None."

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."
            model = build_model(model_config)
            save_load.load_pretrain(model, checkpoint_path)

        else:
            logger.info("Since model_name is given, downloading it...")
            model, config = build_model_from_name(model_name, weights_name)

        self.model = model
        self.config = config

        self.model.eval()

        # sample config
        sample_config = config.get("Sample", None)
        self.sample_config = sample_config

        self.post_transforms_cfg = self.sample_config.get("post_transforms", None)
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def compute_metric(
        self,
        save_path=None,
    ):
        metrics_cfg = self.sample_config.get("metrics")
        assert metrics_cfg is not None, "metrics config must be provided."
        metrics_fn = build_metric(metrics_cfg)

        total_results = self.sample_by_dataloader(save_path)

        metric = metrics_fn(total_results)
        return metric

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def sample(self, data, sample_params=None):
        if sample_params is None:
            sample_params = {}
        assert isinstance(sample_params, dict), "sample_params must be a dict or None."
        pred_data = self.model.sample(data, **sample_params)
        pred_data = self.post_process(pred_data)
        return pred_data

    def sample_by_dataloader(
        self,
        save_path=None,
    ):
        dataset_cfg = self.sample_config["data"]
        data_loader = build_dataloader(dataset_cfg)

        build_structure_cfg = self.sample_config["build_structure_cfg"]
        structure_converter = BuildStructure(**build_structure_cfg)

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
        return total_results

    def sample_by_num_atoms(self, num_atoms, save_path=None, sample_params=None):
        assert isinstance(num_atoms, int), "num_atoms must be an integer."
        data = {
            "structure_array": {
                "num_atoms": paddle.to_tensor(np.array([num_atoms]).astype("int64")),
            }
        }

        result = self.sample(data, sample_params=sample_params)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save results to {save_path}")
            build_structure_cfg = self.sample_config["build_structure_cfg"]
            structure_converter = BuildStructure(**build_structure_cfg)
            structures = structure_converter(result["result"])
            for i, structure in enumerate(structures):
                formula = structure.formula.replace(" ", "-")
                tar_file = os.path.join(save_path, f"{formula}_{i + 1}.cif")
                if structure is not None:
                    writer = CifWriter(structure)
                    writer.write_file(tar_file)
                else:
                    logger.info(f"No structure generated for index {i}")

        return result

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
        result = self.sample(data, sample_params=sample_params)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save results to {save_path}")
            build_structure_cfg = self.sample_config["build_structure_cfg"]
            structure_converter = BuildStructure(**build_structure_cfg)
            structures = structure_converter(result["result"])
            for i, structure in enumerate(structures):
                formula = structure.formula.replace(" ", "-")
                tar_file = os.path.join(save_path, f"{formula}_{i + 1}.cif")
                if structure is not None:
                    writer = CifWriter(structure)
                    writer.write_file(tar_file)
                else:
                    logger.info(f"No structure generated for index {i}")

        return result

    def sample_by_condition(self, composition, save_path=None, sample_params=None):
        # todo: implement this function
        pass


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    argparse.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    argparse.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    argparse.add_argument("--save_path", type=str, default="results")
    argparse.add_argument("--chemical_formula", type=str, default="LiMnO2")
    argparse.add_argument("--num_atoms", type=int, default=4)
    argparse.add_argument(
        "--mode",
        type=str,
        choices=[
            "by_chemical_formula",
            "by_num_atoms",
            "by_dataloader",
            "compute_metric",
        ],
        default="by_chemical_formula",
    )

    args = argparse.parse_args()

    sampler = StructureSampler(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    if args.mode == "compute_metric":
        metric_result = sampler.compute_metric(save_path=args.save_path)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    elif args.mode == "by_chemical_formula":
        result = sampler.sample_by_chemical_formula(
            chemical_formula=args.chemical_formula,
            save_path=args.save_path,
        )
    elif args.mode == "by_num_atoms":
        result = sampler.sample_by_num_atoms(
            num_atoms=args.num_atoms,
            save_path=args.save_path,
        )
    elif args.mode == "by_dataloader":
        result = sampler.sample_by_dataloader(
            save_path=args.save_path,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
