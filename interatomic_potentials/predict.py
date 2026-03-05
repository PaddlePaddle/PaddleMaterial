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
import shutil
from collections import defaultdict
from typing import Dict
from typing import Optional

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.transform import build_post_transforms
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load


def _collect_qm9_urls(config: Dict) -> list[str]:
    urls = []
    dataset_cfg = config.get("Dataset", {})
    for split in ("train", "val", "test"):
        split_cfg = dataset_cfg.get(split, {})
        ds_cfg = split_cfg.get("dataset", {})
        if ds_cfg.get("__class_name__") != "QM9Dataset":
            continue
        init_params = ds_cfg.get("__init_params__", {})
        url = init_params.get("url", None)
        if isinstance(url, str) and len(url) > 0:
            urls.append(url)

    seen = set()
    dedup_urls = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        dedup_urls.append(url)
    return dedup_urls


def _find_atomref_file(path: str) -> Optional[str]:
    if not path or not osp.exists(path):
        return None

    if osp.isfile(path):
        return path if osp.basename(path) == "atomref.npz" else None

    direct_candidates = [
        osp.join(path, "atomref.npz"),
        osp.join(path, "qm9", "atomref.npz"),
    ]
    for candidate in direct_candidates:
        if osp.exists(candidate):
            return candidate

    for root, _, files in os.walk(path):
        if "atomref.npz" in files:
            return osp.join(root, "atomref.npz")
    return None


def _build_default_qm9_atomref() -> np.ndarray:
    atomrefs = {
        6: [0.0, 0.0, 0.0, 0.0, 0.0],
        7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
        8: [-13.57459040, -1029.82456413, -1485.26398105, -2042.57270460, -2713.44632457],
        9: [-13.54887564, -1029.79887659, -1485.23829350, -2042.54701705, -2713.42063702],
        10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    }
    atom_ref = np.zeros((100, 5), dtype=np.float32)
    z_list = [1, 6, 7, 8, 9]
    for col, key in enumerate([6, 7, 8, 9, 10]):
        values = atomrefs[key]
        for atomic_num, value in zip(z_list, values):
            atom_ref[atomic_num, col] = value
    return atom_ref


def _ensure_schnet_atomref(config: Dict):
    model_cfg = config.get("Model", {})
    if model_cfg.get("__class_name__") != "SchNet":
        return

    model_params = model_cfg.get("__init_params__", {})
    atomref_path = model_params.get("atomref_path", None)
    if not atomref_path:
        return
    if osp.exists(atomref_path):
        return

    qm9_urls = _collect_qm9_urls(config)
    atomref_url = model_params.get("atomref_url", None)
    is_qm9_case = bool(qm9_urls) or ("qm9" in str(atomref_path).lower())
    if isinstance(atomref_url, str) and len(atomref_url) > 0:
        is_qm9_case = True
    if not is_qm9_case:
        return

    atomref_dir = osp.dirname(atomref_path) or "."
    os.makedirs(atomref_dir, exist_ok=True)

    candidate_urls = []
    if isinstance(atomref_url, str) and len(atomref_url) > 0:
        candidate_urls.append(atomref_url)
    candidate_urls.extend(qm9_urls)
    candidate_urls.extend(
        [
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.tar.gz",
        ]
    )

    seen = set()
    urls = []
    for url in candidate_urls:
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    for url in urls:
        try:
            if url.endswith(".npz"):
                local_path = download.get_path_from_url(
                    url,
                    atomref_dir,
                    md5sum=None,
                    check_exist=True,
                    decompress=False,
                )
            else:
                local_path = download.get_datasets_path_from_url(url, md5sum=None)

            source_atomref = _find_atomref_file(local_path)
            if source_atomref is None:
                continue

            if osp.abspath(source_atomref) != osp.abspath(atomref_path):
                shutil.copy2(source_atomref, atomref_path)
            logger.info(
                f"Auto prepared missing atomref file: {atomref_path} (source: {source_atomref})"
            )
            return
        except Exception as e:
            logger.warning(f"Failed to auto prepare atomref from {url}: {e}")

    atomref_np = _build_default_qm9_atomref()
    np.savez(atomref_path, atom_ref=atomref_np)
    logger.warning(
        f"atomref.npz not found in provided mirrors. "
        f"Generated default QM9 atom references at {atomref_path}."
    )


class PotentialPredictor:
    """Potential predictor.

    This class provides an interface for predicting properties of crystalline
    structures using pre-trained deep learning models. Supports two initialization
    modes:

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
        # if model_name is not None, then config_path and checkpoint_path must be
        # provided
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is "
            "None."

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            _ensure_schnet_atomref(config)

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
        self.eval_with_no_grad = predict_config.get("eval_with_no_grad", True)

        self.graph_converter_fn = None
        if self.predict_config is not None:
            graph_converter_config = predict_config.get("graph_converter", None)
            if graph_converter_config is not None:
                self.graph_converter_fn = build_graph_converter(graph_converter_config)

        self.post_transforms_cfg = predict_config.get("post_transforms", None)
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
        data = data.tensor()
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

                # save cif_files and result to csv file
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
                df = pd.DataFrame({"cif_file": [cif_file_path], **result_properties})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")

            return result


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
        default="./interatomic_potentials/",
        help="Path to the CIF file whose material properties you want to predict.",
    )
    argparse.add_argument(
        "--save_path",
        type=str,
        default="result.csv",
        help="Path to save the prediction result.",
    )
    args = argparse.parse_args()

    predictor = PotentialPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )

    results = predictor.from_cif_file(args.cif_file_path, args.save_path)
    print(results)
