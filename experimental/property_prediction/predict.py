import os
import os.path as osp
import sys

__dir__ = osp.dirname(osp.abspath(__file__))  # ruff: noqa
sys.path.insert(0, osp.abspath(osp.join(__dir__, "..")))  # ruff: noqa


import argparse
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.models import MODEL_NAMES
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load


class PropertyPredictor:
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

        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        self.model_config = model_config

        self.model_name = model_name
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        self.model = build_model(model_config)
        save_load.load_pretrain(self.model, checkpoint_path)
        self.model.eval()

        predict_config = config.get("Predict", None)
        self.predict_config = predict_config

        self.graph_converter_fn = None
        if self.predict_config is not None:
            graph_converter_config = predict_config.get("graph_converter", None)
            if graph_converter_config is not None:
                self.graph_converter_fn = build_graph_converter(graph_converter_config)

    def graph_converter(self, structure):
        if self.graph_converter_fn is None:
            return structure
        return self.graph_converter_fn(structure)

    def from_structures(self, structures):

        data = self.graph_converter(structures)
        out = self.model.predict(data)
        return out

    def from_cif_file(self, cif_file_path, save_path=None):
        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"
        if osp.isdir(cif_file_path):
            cif_files = [
                osp.join(cif_file_path, f)
                for f in os.listdir(cif_file_path)
                if f.endswith(".cif")
            ]
            results = []
            for cif_file in tqdm(cif_files):
                structure = Structure.from_file(cif_file)
                result = self.from_structures(structure)
                results.append(result)
            if save_path is not None:
                # save cif_files and result to csv file
                df = pd.DataFrame({"cif_file": cif_files, "result": results})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")

            return results
        else:
            structure = Structure.from_file(cif_file_path)
            result = self.from_structures(structure)

            if save_path is not None:
                df = pd.DataFrame({"cif_file": [cif_file_path], "result": [result]})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")

            return result


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument(
        "--model_name", type=str, default="comformer_mp2018_train_60k_e_form"
    )
    argparse.add_argument(
        "--config_path",
        type=str,
        default=None,
        dest="Path to the configuration file.",
    )
    argparse.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        dest="Path to the checkpoint file.",
    )
    argparse.add_argument(
        "--cif_file_path", type=str, default="./property_prediction/example_data/cifs/"
    )
    args = argparse.parse_args()

    predictor = PropertyPredictor(
        model_name=args.model_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    results = predictor.from_cif_file(args.cif_file_path, "result.csv")
    print(results)
