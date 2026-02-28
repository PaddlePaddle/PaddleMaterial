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
import datetime
import os
import os.path as osp
from abc import ABC
from typing import Any
from typing import Dict
from typing import Type

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import run_dataset_transform
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.eager_comp_setting import setting_eager_mode


def read_independent_dataloader_config(config: Dict[str, Any]):
    if config["Global"].get("do_train", True):
        train_data_cfg = config["Dataset"].get("train")
        assert train_data_cfg is not None, (
            "train_data_cfg must be defined when Global.do_train is True"
        )
        train_loader = build_dataloader(train_data_cfg)
    else:
        train_loader = None

    if config["Global"].get("do_eval", False) or config["Global"].get("do_train", True):
        val_data_cfg = config["Dataset"].get("val")
        if val_data_cfg is not None:
            val_loader = build_dataloader(val_data_cfg)
        else:
            logger.info("No validation dataset defined.")
            val_loader = None
    else:
        val_loader = None

    if config["Global"].get("do_test", False):
        test_data_cfg = config["Dataset"].get("test")
        assert test_data_cfg is not None, (
            "test_data_cfg must be defined when Global.do_test is True"
        )
        test_loader = build_dataloader(test_data_cfg)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


TRAIN_CASE_REGISTRY: Dict[str, Type["BaseTrainCase"]] = {}


def register_train_case(cls: Type["BaseTrainCase"]) -> Type["BaseTrainCase"]:
    case_name = cls.case_name.strip().lower()
    if not case_name:
        raise ValueError("Train case must define a non-empty `case_name`.")
    TRAIN_CASE_REGISTRY[case_name] = cls
    return cls


class BaseTrainCase(ABC):
    case_name = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_dataloaders(self):
        set_signal_handlers()
        dataset_cfg = self.config.get("Dataset", {})
        if dataset_cfg.get("split_dataset_ratio") is not None:
            loader = build_dataloader(dataset_cfg)
            train_loader = loader.get("train", None)
            val_loader = loader.get("val", None)
            test_loader = loader.get("test", None)
            return train_loader, val_loader, test_loader
        return read_independent_dataloader_config(self.config)

    def _maybe_apply_dataset_transform(self, train_loader, model_cfg: Dict[str, Any]):
        if not self.config["Global"].get("do_train", True):
            return
        dataset_trans_cfg = self.config.get("Dataset", {}).get("transform")
        if dataset_trans_cfg is None:
            return
        if train_loader is None:
            raise ValueError(
                "Dataset.transform is configured, but train_loader is None."
            )

        trans_cfg = copy.deepcopy(dataset_trans_cfg)
        trans_func = trans_cfg.pop("__class_name__", None)
        trans_params = trans_cfg.pop("__init_params__", {})
        if trans_func is None:
            raise KeyError("Dataset.transform.__class_name__ is required.")

        label_names = self.config.get("Global", {}).get("label_names")
        if label_names is None:
            raise KeyError(
                "Global.label_names is required when Dataset.transform is enabled."
            )

        logger.info(f"Using dataset transform function: {trans_func}")
        data_mean, data_std = run_dataset_transform(
            trans_func, train_loader, label_names, **trans_params
        )
        logger.info(
            f"Target is {label_names}, data mean is {data_mean}, data std is {data_std}"
        )

        model_cfg.setdefault("__init_params__", {})
        model_cfg["__init_params__"]["data_mean"] = data_mean
        model_cfg["__init_params__"]["data_std"] = data_std

    def build_model(self, train_loader, val_loader, test_loader):
        model_cfg = copy.deepcopy(self.config["Model"])
        self._maybe_apply_dataset_transform(train_loader, model_cfg)
        return build_model(model_cfg)

    def build_optimizer(self, model, train_loader):
        if self.config.get("Optimizer") is not None and self.config["Global"].get(
            "do_train", True
        ):
            assert train_loader is not None, (
                "train_loader must be defined when Optimizer is provided."
            )
            assert self.config["Trainer"].get("max_epochs") is not None, (
                "Trainer.max_epochs must be defined when Optimizer is provided."
            )
            return build_optimizer(
                self.config["Optimizer"],
                model,
                self.config["Trainer"]["max_epochs"],
                len(train_loader),
            )
        return None, None

    def build_metric(self):
        metric_cfg = self.config.get("Metric")
        return build_metric(metric_cfg) if metric_cfg is not None else None

    def build_trainer(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        metric_func,
    ):
        return BaseTrainer(
            self.config["Trainer"],
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compute_metric_func_dict=metric_func,
        )

    def post_build_trainer(self, trainer, model, train_loader, val_loader, test_loader):
        return

    def run(self, trainer, train_loader, val_loader, test_loader):
        if self.config["Global"].get("do_train", True):
            trainer.train()
        if self.config["Global"].get("do_eval", False):
            logger.info("Evaluating on validation set")
            trainer.eval(val_loader)
        if self.config["Global"].get("do_test", False):
            logger.info("Evaluating on test set")
            trainer.eval(test_loader)


@register_train_case
class SFINTrainCase(BaseTrainCase):
    case_name = "sfin"


class TrainRunner:
    def __init__(
        self,
        case: str,
        config_path: str,
        dynamic_args: list[str],
        append_timestamp: bool = False,
    ):
        self.case = case.strip().lower()
        self.config_path = config_path
        self.dynamic_args = dynamic_args
        self.append_timestamp = append_timestamp

    def _build_case(self, config: Dict[str, Any]) -> BaseTrainCase:
        case_cls = TRAIN_CASE_REGISTRY.get(self.case)
        if case_cls is None:
            available = ", ".join(sorted(TRAIN_CASE_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported train case '{self.case}'. Available: [{available}]"
            )
        return case_cls(config)

    def _load_and_merge_config(self):
        cfg = OmegaConf.load(self.config_path)
        cli_cfg = OmegaConf.from_dotlist(self.dynamic_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)

        if self.append_timestamp or cfg["Trainer"].get("append_timestamp", False):
            seed = cfg["Trainer"].get("seed", 42)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = cfg["Trainer"]["output_dir"]
            cfg["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"
        return cfg

    def _save_config(self, cfg):
        if dist.get_rank() == 0:
            os.makedirs(cfg["Trainer"]["output_dir"], exist_ok=True)
            config_name = os.path.basename(self.config_path)
            OmegaConf.save(cfg, osp.join(cfg["Trainer"]["output_dir"], config_name))

    @staticmethod
    def _setup_runtime(config: Dict[str, Any]):
        logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
        logger.init_logger(log_file=logger_path)
        logger.info(f"Logger saved to {logger_path}")

        seed = config["Trainer"].get("seed", 42)
        misc.set_random_seed(seed)
        logger.info(f"Set random seed to {seed}")

        enabled = config["Global"].get("prim_eager_enabled", False)
        white_list = config["Global"].get("prim_backward_white_list", None)
        setting_eager_mode(enabled, white_list)

    def run(self):
        cfg = self._load_and_merge_config()
        self._save_config(cfg)
        config = OmegaConf.to_container(cfg, resolve=True)

        self._setup_runtime(config)

        train_case = self._build_case(config)
        train_loader, val_loader, test_loader = train_case.build_dataloaders()
        model = train_case.build_model(train_loader, val_loader, test_loader)
        optimizer, lr_scheduler = train_case.build_optimizer(model, train_loader)
        metric_func = train_case.build_metric()
        trainer = train_case.build_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_func=metric_func,
        )
        train_case.post_build_trainer(
            trainer,
            model,
            train_loader,
            val_loader,
            test_loader,
        )
        train_case.run(trainer, train_loader, val_loader, test_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default="sfin",
        help="Train case name. Extend by registering a new train case.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--append_timestamp",
        action="store_true",
        help="Append timestamp to Trainer.output_dir.",
    )
    return parser.parse_known_args()


def main():
    if dist.get_world_size() > 1:
        fleet.init(is_collective=True)

    args, dynamic_args = parse_args()
    runner = TrainRunner(
        case=args.case,
        config_path=args.config,
        dynamic_args=dynamic_args,
        append_timestamp=args.append_timestamp,
    )
    runner.run()


if __name__ == "__main__":
    main()
