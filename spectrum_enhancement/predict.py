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

from __future__ import annotations

import argparse
import copy
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import numpy as np
import paddle
from omegaconf import OmegaConf
from PIL import Image

from ppmat.datasets.stem_image_dataset import STEMImageDataset
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


def _normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    if split == "validation":
        return "val"
    return split


CASE_PROCESSOR_REGISTRY: Dict[str, Type["BaseCaseProcessor"]] = {}


def register_case_processor(cls: Type["BaseCaseProcessor"]) -> Type["BaseCaseProcessor"]:
    case_name = cls.case_name.strip().lower()
    if not case_name:
        raise ValueError("Case processor must define a non-empty `case_name`.")
    CASE_PROCESSOR_REGISTRY[case_name] = cls
    return cls


class BaseCaseProcessor(ABC):
    """
    Case-level hooks for custom data processing and output processing.

    To add a new model case:
    1. Subclass BaseCaseProcessor.
    2. Implement the abstract methods.
    3. Register with @register_case_processor.
    """

    case_name = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def build_dataset(self, args: argparse.Namespace) -> paddle.io.Dataset:
        raise NotImplementedError

    def prepare_model_input(
        self,
        sample: Dict[str, Any],
        index: int,
        args: argparse.Namespace,
    ) -> Any:
        return sample

    def forward_model(
        self,
        model: paddle.nn.Layer,
        model_input: Any,
        args: argparse.Namespace,
    ) -> Any:
        if hasattr(model, "predict"):
            return model.predict(model_input)
        return model(model_input)

    @abstractmethod
    def parse_model_output(
        self,
        model_output: Any,
        sample: Dict[str, Any],
        index: int,
        args: argparse.Namespace,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_prediction(
        self,
        parsed_output: Any,
        sample: Dict[str, Any],
        index: int,
        output_dir: Path,
        args: argparse.Namespace,
    ) -> Path:
        raise NotImplementedError


@register_case_processor
class SFINCaseProcessor(BaseCaseProcessor):
    case_name = "sfin"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_init = config.get("Model", {}).get("__init_params__", {})
        self.target_name = model_init.get("target_name", "gt_enhance")

    def _resolve_dataset_init_params(self, split: Optional[str]) -> Dict[str, Any]:
        dataset_cfg_root = self.config.get("Dataset", {})
        if not isinstance(dataset_cfg_root, dict):
            return {}

        split = _normalize_split(split)
        candidate_keys = []
        if split is not None:
            candidate_keys.append(split)
        candidate_keys.extend(["test", "val", "train"])

        for key in candidate_keys:
            branch_cfg = dataset_cfg_root.get(key)
            if not isinstance(branch_cfg, dict):
                continue
            dataset_cfg = branch_cfg.get("dataset", {})
            if not isinstance(dataset_cfg, dict):
                continue
            if dataset_cfg.get("__class_name__") == "STEMImageDataset":
                return copy.deepcopy(dataset_cfg.get("__init_params__", {}))
        return {}

    def build_dataset(self, args: argparse.Namespace) -> paddle.io.Dataset:
        init_params = self._resolve_dataset_init_params(args.split)

        init_params["data_path"] = args.data_path or init_params.get("data_path", "./data_test")
        init_params["file_suffix"] = args.file_suffix or init_params.get("file_suffix", ".png")
        init_params["split"] = (
            _normalize_split(args.split)
            if args.split is not None
            else init_params.get("split", None)
        )

        if args.data_count > 0:
            init_params["data_count"] = args.data_count
        else:
            init_params["data_count"] = None

        if args.noisy_subdir is not None:
            init_params["noisy_subdir"] = args.noisy_subdir
        if args.target_subdir is not None:
            init_params["target_subdir"] = args.target_subdir

        # Preserve config defaults unless CLI explicitly overrides.
        if args.download is not None:
            init_params["download"] = bool(args.download)
        if args.force_download is not None:
            init_params["force_download"] = bool(args.force_download)
        return STEMImageDataset(**init_params)

    def prepare_model_input(
        self,
        sample: Dict[str, Any],
        index: int,
        args: argparse.Namespace,
    ) -> Dict[str, Any]:
        if not isinstance(sample, dict):
            return sample

        model_init = self.config.get("Model", {}).get("__init_params__", {})
        input_key = model_init.get("input_name", "noisy")
        key_candidates = [input_key, "image", "noisy", "input", "x"]

        model_input = dict(sample)
        for key in key_candidates:
            x = model_input.get(key)
            if isinstance(x, paddle.Tensor):
                if x.ndim == 3:
                    model_input[key] = x.unsqueeze(0)
                elif x.ndim == 2:
                    model_input[key] = x.unsqueeze(0).unsqueeze(0)
                break
        return model_input

    def _pick_prediction_tensor(self, output: Any) -> paddle.Tensor:
        if isinstance(output, dict):
            if "pred_dict" in output and isinstance(output["pred_dict"], dict):
                output = output["pred_dict"]

            for key in [self.target_name, "pred", "output", "enhanced", "image"]:
                if key in output and output[key] is not None:
                    pred = output[key]
                    break
            else:
                pred = None
                for value in output.values():
                    if isinstance(value, paddle.Tensor):
                        pred = value
                        break
                if pred is None:
                    raise KeyError(
                        "Cannot find prediction tensor in model output dict. "
                        f"Keys: {list(output.keys())}"
                    )
        elif isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("Model output list/tuple is empty.")
            pred = output[0]
        else:
            pred = output

        if not isinstance(pred, paddle.Tensor):
            pred = paddle.to_tensor(pred)
        return pred

    def parse_model_output(
        self,
        model_output: Any,
        sample: Dict[str, Any],
        index: int,
        args: argparse.Namespace,
    ) -> np.ndarray:
        pred = self._pick_prediction_tensor(model_output)
        pred = paddle.clip(pred, min=0.0, max=255.0)
        pred_np = pred.squeeze().detach().cpu().numpy().astype(np.uint8)

        if pred_np.ndim == 3 and pred_np.shape[0] in (1, 3):
            pred_np = np.transpose(pred_np, (1, 2, 0))
        if pred_np.ndim == 3 and pred_np.shape[-1] == 1:
            pred_np = pred_np[..., 0]
        return pred_np

    def save_prediction(
        self,
        parsed_output: np.ndarray,
        sample: Dict[str, Any],
        index: int,
        output_dir: Path,
        args: argparse.Namespace,
    ) -> Path:
        file_name = sample.get("name", f"{index}{args.file_suffix or '.png'}")
        if args.save_suffix:
            suffix = args.save_suffix
            if not suffix.startswith("."):
                suffix = f".{suffix}"
            file_name = f"{Path(file_name).stem}{suffix}"
        elif Path(file_name).suffix == "":
            file_name = f"{file_name}{args.file_suffix or '.png'}"

        save_path = output_dir / file_name
        Image.fromarray(parsed_output).save(save_path)
        return save_path


class SpectrumPredictor:
    def __init__(
        self,
        case: str,
        device: str,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
    ):
        self.case = case.strip().lower()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.weights_name = weights_name
        self.device = device

        paddle.set_device(device)

        if self.model_name:
            logger.info(
                f"Loading predefined model by name: {self.model_name} "
                f"(weights_name={self.weights_name})"
            )
            self.model, self.config = build_model_from_name(
                self.model_name, self.weights_name
            )
            self.model.eval()
        else:
            if not self.config_path:
                raise ValueError(
                    "`config_path` is required when `model_name` is not provided."
                )
            if not self.checkpoint_path:
                raise ValueError(
                    "`checkpoint_path` is required when `model_name` is not provided."
                )
            self.config = self._load_config(self.config_path)
            self.model = self._build_model_and_load_checkpoint()

        self.case_processor = self._build_case_processor(self.case)
        self.eval_with_no_grad = (
            self.config.get("Predict", {}).get("eval_with_no_grad", True)
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)

    def _build_model_and_load_checkpoint(self):
        model_cfg = self.config.get("Model")
        if model_cfg is None:
            raise ValueError("`Model` section is required in config.")
        model = build_model(model_cfg)
        self._load_checkpoint(model, self.checkpoint_path)
        model.eval()
        return model

    def _build_case_processor(self, case: str) -> BaseCaseProcessor:
        processor_cls = CASE_PROCESSOR_REGISTRY.get(case)
        if processor_cls is None:
            available = ", ".join(sorted(CASE_PROCESSOR_REGISTRY.keys()))
            raise ValueError(f"Unsupported case '{case}'. Available cases: [{available}]")
        return processor_cls(self.config)

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: Optional[str]) -> None:
        if not checkpoint_path:
            raise ValueError("`checkpoint_path` must not be empty.")
        checkpoint_loaded = False
        try:
            checkpoint = paddle.load(checkpoint_path)
            if isinstance(checkpoint, dict):
                state_dict = None
                for key in ("model_state_dict", "model", "state_dict"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        state_dict = checkpoint[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint
                model.set_state_dict(state_dict)
                checkpoint_loaded = True
        except Exception:
            checkpoint_loaded = False

        if not checkpoint_loaded:
            save_load.load_pretrain(model, checkpoint_path)

    def run(self, args: argparse.Namespace) -> list[Path]:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.case_processor.build_dataset(args)
        if len(dataset) == 0:
            raise ValueError("No samples found in dataset.")

        saved_paths = []
        context = paddle.no_grad() if self.eval_with_no_grad else nullcontext()
        with context:
            for idx in range(len(dataset)):
                sample = dataset[idx]
                model_input = self.case_processor.prepare_model_input(sample, idx, args)
                model_output = self.case_processor.forward_model(
                    self.model,
                    model_input,
                    args,
                )
                parsed_output = self.case_processor.parse_model_output(
                    model_output,
                    sample,
                    idx,
                    args,
                )
                save_path = self.case_processor.save_prediction(
                    parsed_output,
                    sample,
                    idx,
                    output_dir,
                    args,
                )
                saved_paths.append(save_path)

        return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic spectrum enhancement prediction with case-level hooks."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="sfin",
        help="Prediction case name. Extend by registering a new case processor.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Optional predefined model name from MODEL_REGISTRY. "
            "If provided, `config_path` and `checkpoint_path` are optional."
        ),
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help=(
            "Optional weight filename when `model_name` is used "
            "(e.g., best.pdparams / latest.pdparams)."
        ),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml",
        help="Path to model config yaml (used when model_name is not provided).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path or URL to checkpoint (*.pdparams) (used when model_name is not provided).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Root directory of input data. If omitted, infer from config Dataset section.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "validation", "test"],
        help="Dataset split to use. If omitted, case processor chooses default split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save predictions. Defaults to <Trainer.output_dir>/predictions.",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=None,
        help="Input file suffix for dataset scanning (e.g., .png).",
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        default=None,
        help="Optional output file suffix override (e.g., .png).",
    )
    parser.add_argument(
        "--data_count",
        type=int,
        default=-1,
        help="Max number of samples to process, <=0 means all.",
    )
    parser.add_argument(
        "--noisy_subdir",
        type=str,
        default=None,
        help="Optional override for noisy image sub-directory.",
    )
    parser.add_argument(
        "--target_subdir",
        type=str,
        default=None,
        help="Optional override for target image sub-directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=None,
        help=(
            "Enable auto-download when data_path is missing. "
            "If omitted, keep dataset config default."
        ),
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        default=None,
        help=(
            "Force re-download dataset archive. "
            "If omitted, keep dataset config default."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu" if paddle.device.cuda.device_count() > 0 else "cpu",
        choices=["cpu", "gpu"],
        help="Device to run inference.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    if args.output_dir:
        return args.output_dir
    trainer_output_dir = config.get("Trainer", {}).get("output_dir")
    if trainer_output_dir:
        return str(Path(trainer_output_dir) / "predictions")
    if args.config_path:
        return str(Path("./output") / Path(args.config_path).stem / "predictions")
    if args.model_name:
        return str(Path("./output") / args.model_name / "predictions")
    return str(Path("./output") / "spectrum_enhancement" / "predictions")


def validate_args(args: argparse.Namespace) -> None:
    # Backward-compatible behavior:
    # - Existing config+checkpoint workflow keeps working.
    # - New model_name workflow is optional.
    if args.model_name:
        return
    if not args.config_path or not args.checkpoint_path:
        raise ValueError(
            "Either provide `--model_name`, or provide both "
            "`--config_path` and `--checkpoint_path`."
        )


def main():
    args = parse_args()
    validate_args(args)
    predictor = SpectrumPredictor(
        case=args.case,
        device=args.device,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        weights_name=args.weights_name,
    )
    args.output_dir = resolve_output_dir(args, predictor.config)
    saved_paths = predictor.run(args)
    logger.info(f"Saved {len(saved_paths)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
