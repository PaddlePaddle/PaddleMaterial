# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
from PIL import Image
from tqdm import tqdm

from ppmat.datasets.build_image import BuildImage
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class SpectrumPredictor(BasePredictor):
    """Spectrum enhancement predictor.

    This class provides an interface for enhancing spectrum images using
    pre-trained deep learning models. Supports two initialization modes:

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

    def from_image(self, image):
        return self._run_model(image)

    @staticmethod
    def to_image(pred: paddle.Tensor) -> Image.Image:
        pred = paddle.clip(pred, min=0.0, max=255.0)
        pred = pred.squeeze().detach().cpu().numpy()
        if pred.ndim == 3 and pred.shape[0] in (1, 3):
            pred = np.transpose(pred, (1, 2, 0))
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        return Image.fromarray(pred.astype(np.uint8))

    @staticmethod
    def _save_image(
        image: Image.Image,
        output_dir: Path,
        file_name: str,
        file_suffix: str = ".png",
    ) -> Path:
        if Path(file_name).suffix == "":
            file_name = f"{file_name}{file_suffix}"
        save_path = output_dir / file_name
        image.save(save_path)
        return save_path

    def from_image_file(
        self,
        image_file_path: str,
        save_path: Optional[str] = None,
    ):
        """Predict enhanced spectra from an image file or directory.

        Args:
            image_file_path: Path to one image or a directory of images.
            save_path: Optional directory for enhanced PNG images.

        Returns:
            List of prediction dictionaries.
        """
        image_path = Path(image_file_path)
        if image_path.is_dir():
            image_files = [
                path
                for path in sorted(image_path.iterdir())
                if path.is_file()
                and path.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            ]
        else:
            image_files = [image_path]

        if not image_files or not all(
            path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            for path in image_files
        ):
            raise ValueError(f"Expected an image file or directory: {image_file_path}")

        output_dir = Path(save_path) if save_path is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        image_builder = BuildImage(
            format="image_file",
            mode="L",
            dtype="float32",
        )
        results = []
        for image_file in tqdm(image_files, desc="Predict"):
            image = image_builder(image_file)
            result = self.from_image(image)
            results.append(result)
            if output_dir is not None:
                pred = result[self.model.target_name]
                self._save_image(
                    self.to_image(pred),
                    output_dir,
                    image_file.stem,
                )

        if output_dir is not None:
            logger.info(f"Saved {len(results)} predictions to {output_dir}")
        return results
