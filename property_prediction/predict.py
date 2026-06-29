# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import os.path as osp
from collections import defaultdict
from typing import Optional

import paddle
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.transform import build_post_transforms
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


class PropertyPredictor:
    """Property predictor.

    Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify ``model_name`` and (optionally) ``weights_name`` to
       download and load pre-trained weights from ``MODEL_REGISTRY``.

    2. **Custom Model Loading**
       Provide explicit ``config_path`` and ``checkpoint_path``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
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

        predict_config = config.get("Predict", None)
        self.predict_config = predict_config
        self.eval_with_no_grad = (
            predict_config.get("eval_with_no_grad", True)
            if predict_config is not None else True
        )

        self.graph_converter_fn = None
        if self.predict_config is not None:
            graph_converter_config = predict_config.get("graph_converter", None)
            if graph_converter_config is not None:
                self.graph_converter_fn = build_graph_converter(
                    graph_converter_config
                )

        self.post_transforms_cfg = (
            predict_config.get("post_transforms", None)
            if predict_config is not None else None
        )
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def graph_converter(self, structure):
        if self.graph_converter_fn is None:
            return structure
        return self.graph_converter_fn(structure)

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def from_structures(self, structures):
        data = self.graph_converter(structures)
        if self.eval_with_no_grad:
            with paddle.no_grad():
                out = self.model.predict(data)
        else:
            out = self.model.predict(data)
        out = self.post_process(out)
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
                keys = list(results[0].keys())
                result_properties = defaultdict(list)
                for key in keys:
                    for r in results:
                        result_properties[key].append(r[key])
                df = pd.DataFrame({"cif_file": cif_files, **result_properties})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")
            return results
        else:
            structure = Structure.from_file(cif_file_path)
            result = self.from_structures(structure)
            keys = list(result.keys())
            result_properties = defaultdict(list)
            for key in keys:
                result_properties[key].append(result[key])
            if save_path is not None:
                df = pd.DataFrame(
                    {"cif_file": [cif_file_path], **result_properties}
                )
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")
            return result

    def from_xyz_file(self, xyz_file_path, save_path=None):
        """Predict molecular properties from XYZ file(s).

        Args:
            xyz_file_path: Path to a single ``.xyz`` file or a directory
                of ``.xyz`` files.
            save_path: Optional CSV path to save results.

        Returns:
            Single result dict or list of result dicts.
        """
        from ppmat.datasets.qm9_dataset import _SYMBOL_TO_Z

        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"

        if osp.isdir(xyz_file_path):
            xyz_files = sorted([
                osp.join(xyz_file_path, f)
                for f in os.listdir(xyz_file_path)
                if f.endswith(".xyz")
            ])
        else:
            xyz_files = [xyz_file_path]

        graph_cfg = None
        dataset_cfg = self.config.get("Model", {}).get("dataset", None)
        if dataset_cfg is not None:
            graph_cfg = dataset_cfg.get("build_graph_cfg", None)
        converter = None
        if graph_cfg is not None:
            converter = build_graph_converter(graph_cfg)

        results = []
        for xyz_path in tqdm(xyz_files, desc="Predict"):
            with open(xyz_path, "r") as f:
                lines = f.readlines()
            n_atoms = int(lines[0].strip())
            z_list, pos_list = [], []
            for i in range(n_atoms):
                parts = lines[2 + i].strip().split()
                symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                z_list.append(_SYMBOL_TO_Z.get(symbol, 0))
                pos_list.append([x, y, z])
            z_t = paddle.to_tensor(z_list, dtype=paddle.int64)
            pos_t = paddle.to_tensor(
                pos_list, dtype=paddle.get_default_dtype()
            )
            batch_t = paddle.zeros([n_atoms], dtype=paddle.int64)

            data = {"z": z_t, "pos": pos_t, "batch": batch_t}
            if converter is not None:
                data["edge_index"] = converter(pos_t, batch_t)

            out = self.model.predict(data)
            results.append(out)

        if save_path is not None and results:
            keys = list(results[0].keys())
            result_properties = defaultdict(list)
            for key in keys:
                for r in results:
                    result_properties[key].append(r[key])
            df = pd.DataFrame({
                "xyz_file": [osp.basename(f) for f in xyz_files],
                **result_properties,
            })
            df.to_csv(save_path, index=False)
            logger.info(f"Saved prediction results to {save_path}")

        return results if len(results) > 1 else results[0]


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name.",
    )
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
    argparse.add_argument(
        "--cif_file_path",
        type=str,
        default=None,
        help="Path to CIF file(s) for crystal property prediction.",
    )
    argparse.add_argument(
        "--xyz_file_path",
        type=str,
        default=None,
        help="Path to XYZ file(s) for molecular property prediction.  "
        "When neither --cif_file_path nor --xyz_file_path is given, "
        "defaults to the example molecule (qm9_sample.xyz).",
    )
    argparse.add_argument(
        "--save_path",
        type=str,
        default="result.csv",
        help="Path to save the prediction result.",
    )
    args = argparse.parse_args()

    predictor = PropertyPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    if args.xyz_file_path is not None:
        results = predictor.from_xyz_file(args.xyz_file_path, args.save_path)
    elif args.cif_file_path is not None:
        results = predictor.from_cif_file(args.cif_file_path, args.save_path)
    else:
        results = predictor.from_xyz_file(
            "./property_prediction/example_data/molecules/", args.save_path
        )
    print(results)
