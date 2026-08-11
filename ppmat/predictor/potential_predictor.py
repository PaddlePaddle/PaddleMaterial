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

import os
import os.path as osp
from collections import defaultdict
from typing import Optional
from typing import Sequence

import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class PotentialPredictor(BasePredictor):
    """Potential predictor.

    This class provides an interface for predicting properties of crystal structures and
    molecules using pre-trained deep learning models. Supports two initialization
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
        checkpoint_path (Optional[str], optional): Path to a model checkpoint file
            (.pdparams) for custom models. If omitted, `Predict.checkpoint_path` from
            the config is used. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=device,
            config_overrides=config_overrides,
        )
        self.load_inference_model()

    def from_structures(self, structures):
        data = self.graph_converter(structures)
        data = data.tensor()
        return self._run_model(data)

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

    def from_xyz_file(self, xyz_file_path, save_path=None):
        """Predict molecular energy and forces from one XYZ file."""
        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"
        if not osp.isfile(xyz_file_path) or not xyz_file_path.endswith(".xyz"):
            raise ValueError(f"Expected one XYZ file, but got: {xyz_file_path}")
        if self.graph_converter_fn is None:
            raise ValueError("Molecular prediction requires a graph converter.")

        molecule = BuildMolecule(format="xyz_file", sanitize=False)(xyz_file_path)
        if molecule is None:
            raise ValueError(f"Failed to parse XYZ file: {xyz_file_path}")
        graph = self.graph_converter_fn(molecule)

        result = self._run_model(graph)

        if save_path is not None:
            row = {"xyz_file": osp.basename(xyz_file_path)}
            row.update(
                {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in result.items()
                }
            )
            pd.DataFrame([row]).to_csv(save_path, index=False)
            logger.info(f"Saved the prediction result to {save_path}")
        return result
