# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
from typing import Optional

import paddle
import pandas as pd
from tqdm import tqdm

from ppmat.predict import PPMatPredictor
from ppmat.utils import logger


class PropertyPredictor(PPMatPredictor):
    """Property predictor.

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
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

    def get_predict(
        self,
        files: list,
        structures: list,
        file_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        self.init_save_dir(file_path, save_path)

        os.makedirs(self.save_path, exist_ok=True)
        self.save_path = osp.join(self.save_path, "results_pred_property.csv")
        logger.info(f"Save predictions to {self.save_path}")

        # predict
        results = []
        for structure in tqdm(structures):
            data = self.graph_converter(structure)
            data = data.tensor()
            if self.eval_with_no_grad:
                with paddle.no_grad():
                    out = self.model.predict(data)
            else:
                out = self.model.predict(data)
            out = self.post_process(out)
            results.append(out)

        # save file names and output to csv file
        if not results:
            raise ValueError("No results to save csv file.")
        df = pd.DataFrame(results)
        df.insert(0, "file_name", files)
        df.to_csv(self.save_path, index=False)
        logger.info(f"Saved the prediction result to {self.save_path}")
