# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
from PIL import Image
from tqdm import tqdm

from ppmat.datasets import build_dataloader
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class SpectrumPredictor(BasePredictor):
    """Spectrum enhancement predictor.

    Dataset prediction uses the configured ``Dataset.<split>`` branch directly,
    keeping prediction aligned with the training/evaluation data interface.
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

        model_init = self.config.get("Model", {}).get("__init_params__", {})
        self.input_name = model_init.get("input_name", "noisy")
        self.target_name = model_init.get("target_name", "gt_enhance")

    def predict_batch(self, batch):
        return self._run_model(batch)

    def _get_prediction_tensor(self, output) -> paddle.Tensor:
        if not isinstance(output, dict):
            raise TypeError(f"Expected dict output, but got {type(output)}.")
        if self.target_name not in output:
            raise KeyError(
                f"Prediction key '{self.target_name}' not found in output keys "
                f"{list(output.keys())}."
            )
        return output[self.target_name]

    @staticmethod
    def _tensor_to_image(pred: paddle.Tensor) -> np.ndarray:
        pred = paddle.clip(pred, min=0.0, max=255.0)
        pred = pred.squeeze().detach().cpu().numpy()
        if pred.ndim == 3 and pred.shape[0] in (1, 3):
            pred = np.transpose(pred, (1, 2, 0))
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        return pred.astype(np.uint8)

    @staticmethod
    def _config_split_key(split: str) -> str:
        """Return the ``Dataset`` config key for a split name."""

        return "val" if split == "validation" else split

    @staticmethod
    def _normalize_file_name(file_name, default_name: str) -> str:
        if isinstance(file_name, (list, tuple)):
            file_name = file_name[0] if file_name else default_name
        if isinstance(file_name, str):
            return file_name
        return default_name

    @staticmethod
    def _save_image(
        pred: np.ndarray,
        output_dir: Path,
        file_name: str,
        file_suffix: str = ".png",
    ) -> Path:
        if Path(file_name).suffix == "":
            file_name = f"{file_name}{file_suffix}"
        save_path = output_dir / file_name
        Image.fromarray(pred).save(save_path)
        return save_path

    def _predict_from_dataset_cfg(
        self,
        dataset_cfg,
        save_path: str,
    ):
        dataloader = build_dataloader(copy.deepcopy(dataset_cfg), vocab=self.vocab)
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_suffix = (
            dataset_cfg.get("dataset", {})
            .get("__init_params__", {})
            .get("file_suffix", ".png")
        )
        saved_paths = []
        for idx, batch in enumerate(tqdm(dataloader)):
            output = self.predict_batch(batch)
            pred = self._get_prediction_tensor(output)
            if len(pred.shape) >= 4:
                pred_list = [pred[i] for i in range(pred.shape[0])]
            else:
                pred_list = [pred]

            names = batch.get("name")
            for batch_idx, pred_item in enumerate(pred_list):
                if isinstance(names, (list, tuple)):
                    name = names[batch_idx] if batch_idx < len(names) else None
                else:
                    name = names
                file_name = self._normalize_file_name(
                    name, f"{idx * len(pred_list) + batch_idx}{file_suffix}"
                )
                saved_paths.append(
                    self._save_image(
                        self._tensor_to_image(pred_item),
                        output_dir,
                        file_name,
                        file_suffix,
                    )
                )
        return saved_paths

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

    @staticmethod
    def _stage_image(source_path: Path, target_dir: Path, file_suffix: str) -> Path:
        file_suffix = file_suffix if file_suffix.startswith(".") else f".{file_suffix}"
        target_path = target_dir / f"{source_path.stem}{file_suffix}"
        if source_path.suffix.lower() == file_suffix.lower():
            shutil.copy2(source_path, target_path)
        else:
            with Image.open(source_path) as image:
                image.save(target_path)
        return target_path

    @contextmanager
    def _dataset_cfg_from_input_path(
        self,
        input_path: str,
        split: str = "test",
    ):
        split = self._config_split_key(split)
        dataset_cfg = copy.deepcopy(self.config.get("Dataset", {}).get(split, None))
        if dataset_cfg is None:
            raise KeyError(f"Dataset.{split} is not defined in config.")

        init_params = dataset_cfg.get("dataset", {}).get("__init_params__", {})
        dataset_split = init_params.get("split", "test")
        disk_split = "test" if dataset_split == "val" else dataset_split
        noisy_subdir = init_params.get("noisy_subdir", "noisy")
        file_suffix = init_params.get("file_suffix", ".png")
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        init_params["target_subdir"] = None
        init_params.pop("target_name", None)

        if input_path.is_file():
            with tempfile.TemporaryDirectory(
                prefix="ppmat_spectrum_predict_"
            ) as temp_dir:
                temp_root = Path(temp_dir)
                staged_noisy_dir = temp_root / disk_split / noisy_subdir
                staged_noisy_dir.mkdir(parents=True, exist_ok=True)
                self._stage_image(input_path, staged_noisy_dir, file_suffix)
                init_params["path"] = str(temp_root)
                yield dataset_cfg
            return

        if (input_path / disk_split / noisy_subdir).is_dir():
            init_params["path"] = str(input_path)
            yield dataset_cfg
            return

        if input_path.name == noisy_subdir and input_path.parent.name in (
            "train",
            "test",
        ):
            init_params["path"] = str(input_path.parent.parent)
            init_params["split"] = input_path.parent.name
            yield dataset_cfg
            return

        image_files = [
            file_path
            for file_path in sorted(input_path.iterdir())
            if self._is_image_file(file_path)
        ]
        if not image_files:
            raise FileNotFoundError(f"No image files found under {input_path}.")

        with tempfile.TemporaryDirectory(prefix="ppmat_spectrum_predict_") as temp_dir:
            temp_root = Path(temp_dir)
            staged_noisy_dir = temp_root / disk_split / noisy_subdir
            staged_noisy_dir.mkdir(parents=True, exist_ok=True)
            for image_file in image_files:
                self._stage_image(image_file, staged_noisy_dir, file_suffix)
            init_params["path"] = str(temp_root)
            logger.info(f"Load {len(image_files)} noisy images from {input_path}")
            yield dataset_cfg

    def from_dataset(
        self,
        split: str = "test",
        save_path: str = "./output/spectrum_enhancement/predictions",
    ):
        split = self._config_split_key(split)
        dataset_cfg = self.config.get("Dataset", {}).get(split, None)
        if dataset_cfg is None:
            raise KeyError(f"Dataset.{split} is not defined in config.")
        return self._predict_from_dataset_cfg(dataset_cfg, save_path)

    def from_image_path(
        self,
        input_path: str,
        save_path: str = "./output/spectrum_enhancement/predictions",
        split: str = "test",
    ):
        with self._dataset_cfg_from_input_path(input_path, split) as dataset_cfg:
            return self._predict_from_dataset_cfg(dataset_cfg, save_path)
