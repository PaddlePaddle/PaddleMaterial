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

import argparse
import copy
import os.path as osp
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from ppmat.datasets import build_dataloader
from ppmat.predictor import BasePredictor
from ppmat.utils import logger


class SpectrumPredictor(BasePredictor):
    """Spectrum enhancement predictor.

    The model-loading and post-process flow follows the repository predictor
    entries such as ``property_prediction/predict.py``. Dataset prediction uses
    the configured ``Dataset.<split>`` branch directly, so prediction stays
    aligned with the training/evaluation data interface.
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
            work_dir="",
        )
        self.load_inference_model()

        model_init = self.config.get("Model", {}).get("__init_params__", {})
        self.input_name = model_init.get("input_name", "noisy")
        self.target_name = model_init.get("target_name", "gt_enhance")

    def predict_batch(self, batch):
        if self.eval_with_no_grad:
            with paddle.no_grad():
                out = self.model.predict(batch)
        else:
            out = self.model.predict(batch)
        return self.post_process(out)

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
    def _normalize_split(split: str) -> str:
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
        output_dir: str,
    ):
        dataloader = build_dataloader(copy.deepcopy(dataset_cfg))
        output_dir = Path(output_dir)
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

    @contextmanager
    def _dataset_cfg_from_input_path(
        self,
        input_path: str,
        split: str = "test",
    ):
        split = self._normalize_split(split)
        dataset_cfg = copy.deepcopy(self.config.get("Dataset", {}).get(split, None))
        if dataset_cfg is None:
            raise KeyError(f"Dataset.{split} is not defined in config.")

        init_params = dataset_cfg.get("dataset", {}).get("__init_params__", {})
        noisy_subdir = init_params.get("noisy_subdir", "noisy")
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        init_params["target_subdir"] = None
        init_params["target_name"] = None
        init_params["auto_download"] = False

        if input_path.is_file():
            with tempfile.TemporaryDirectory(
                prefix="ppmat_spectrum_predict_"
            ) as temp_dir:
                temp_root = Path(temp_dir)
                staged_noisy_dir = temp_root / noisy_subdir
                staged_noisy_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, staged_noisy_dir / input_path.name)
                init_params["data_path"] = str(temp_root)
                yield dataset_cfg
            return

        if (input_path / noisy_subdir).is_dir():
            init_params["data_path"] = str(input_path)
            yield dataset_cfg
            return

        image_files = [
            file_path
            for file_path in sorted(input_path.iterdir())
            if self._is_image_file(file_path)
        ]
        if not image_files:
            raise FileNotFoundError(f"No image files found under {input_path}.")

        init_params["data_path"] = str(input_path.parent)
        init_params["noisy_subdir"] = input_path.name
        logger.info(f"Load {len(image_files)} noisy images from {input_path}")
        yield dataset_cfg

    def from_dataset(
        self,
        split: str = "test",
        output_dir: str = "./output/spectrum_enhancement/predictions",
    ):
        split = self._normalize_split(split)
        dataset_cfg = self.config.get("Dataset", {}).get(split, None)
        if dataset_cfg is None:
            raise KeyError(f"Dataset.{split} is not defined in config.")
        return self._predict_from_dataset_cfg(dataset_cfg, output_dir)

    def from_image_path(
        self,
        input_path: str,
        output_dir: str = "./output/spectrum_enhancement/predictions",
        split: str = "test",
    ):
        with self._dataset_cfg_from_input_path(input_path, split) as dataset_cfg:
            return self._predict_from_dataset_cfg(dataset_cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name.")
    parser.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help=(
            "Path to noisy image file or directory. If omitted, predict from "
            "Dataset.<split> in the config."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "validation", "test"],
        help="Dataset split used when input_path is omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save prediction images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu" if paddle.device.cuda.device_count() > 0 else "cpu",
        choices=["cpu", "gpu"],
        help="Device to run inference.",
    )
    return parser.parse_args()


def _default_output_dir(
    config,
    config_path: Optional[str],
    model_name: Optional[str],
) -> str:
    trainer_output_dir = config.get("Trainer", {}).get("output_dir")
    if trainer_output_dir:
        return osp.join(trainer_output_dir, "predictions")
    if config_path:
        return osp.join("./output", Path(config_path).stem, "predictions")
    if model_name:
        return osp.join("./output", model_name, "predictions")
    return "./output/spectrum_enhancement/predictions"


def _checkpoint_path_from_config(config_path: Optional[str]) -> Optional[str]:
    if config_path is None:
        return None
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    predict_config = config.get("Predict", None)
    if predict_config is None:
        return None
    return predict_config.get("checkpoint_path", None)


if __name__ == "__main__":
    args = parse_args()
    paddle.set_device(args.device)

    checkpoint_path = args.checkpoint_path
    if args.model_name is None and checkpoint_path is None:
        checkpoint_path = _checkpoint_path_from_config(args.config_path)

    predictor = SpectrumPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=checkpoint_path,
    )

    output_dir = args.output_dir or _default_output_dir(
        predictor.config, args.config_path, args.model_name
    )
    if args.input_path is not None:
        saved_paths = predictor.from_image_path(args.input_path, output_dir, args.split)
    else:
        saved_paths = predictor.from_dataset(args.split, output_dir)
    logger.info(f"Saved {len(saved_paths)} predictions to {output_dir}")
