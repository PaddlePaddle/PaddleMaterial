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
from typing import Any
from typing import Dict

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import run_dataset_transform
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer import build_trainer
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


def build_dataloaders(config: Dict[str, Any]):
    set_signal_handlers()
    dataset_cfg = config.get("Dataset", {})
    if dataset_cfg.get("split_dataset_ratio") is not None:
        loader = build_dataloader(dataset_cfg)
        return loader.get("train"), loader.get("val"), loader.get("test")
    return read_independent_dataloader_config(config)


def maybe_apply_dataset_transform(
    config: Dict[str, Any],
    train_loader,
    model_cfg: Dict[str, Any],
):
    if not config["Global"].get("do_train", True):
        return
    dataset_trans_cfg = config.get("Dataset", {}).get("transform")
    if dataset_trans_cfg is None:
        return
    if train_loader is None:
        raise ValueError("Dataset.transform is configured, but train_loader is None.")

    trans_cfg = copy.deepcopy(dataset_trans_cfg)
    trans_func = trans_cfg.pop("__class_name__", None)
    trans_params = trans_cfg.pop("__init_params__", {})
    if trans_func is None:
        raise KeyError("Dataset.transform.__class_name__ is required.")

    label_names = config.get("Global", {}).get("label_names")
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


def setup_runtime(config: Dict[str, Any]):
    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")

    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    enabled = config["Global"].get("prim_eager_enabled", False)
    white_list = config["Global"].get("prim_backward_white_list", None)
    setting_eager_mode(enabled, white_list)


def build_components(config: Dict[str, Any]):
    train_loader, val_loader, test_loader = build_dataloaders(config)

    model_cfg = copy.deepcopy(config["Model"])
    maybe_apply_dataset_transform(config, train_loader, model_cfg)
    model = build_model(model_cfg)

    trainer_runtime_cfg = config["Trainer"]
    if "__init_params__" in trainer_runtime_cfg:
        trainer_runtime_cfg = trainer_runtime_cfg.get("__init_params__", {}).get(
            "config", trainer_runtime_cfg
        )

    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert train_loader is not None, (
            "train_loader must be defined when Optimizer is provided."
        )
        assert trainer_runtime_cfg.get("max_epochs") is not None, (
            "Trainer.max_epochs must be defined when Optimizer is provided."
        )
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            trainer_runtime_cfg["max_epochs"],
            len(train_loader),
        )
    else:
        optimizer, lr_scheduler = None, None

    metric_cfg = config.get("Metric")
    metric_func = build_metric(metric_cfg) if metric_cfg is not None else None

    trainer_cfg = config["Trainer"]
    if "__init_params__" not in trainer_cfg:
        trainer_cfg = {
            "__class_name__": "BaseTrainer",
            "__init_params__": {"config": trainer_cfg},
        }

    trainer = build_trainer(
        trainer_cfg,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        compute_metric_func_dict=metric_func,
    )
    return trainer, train_loader, val_loader, test_loader


def run(config: Dict[str, Any]):
    setup_runtime(config)
    trainer, train_loader, val_loader, test_loader = build_components(config)

    if config["Global"].get("do_train", True):
        trainer.train()
    if config["Global"].get("do_eval", False):
        logger.info("Evaluating on validation set")
        trainer.eval(val_loader)
    if config["Global"].get("do_test", False):
        logger.info("Evaluating on test set")
        trainer.eval(test_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
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

    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(dynamic_args)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if args.append_timestamp or cfg["Trainer"].get("append_timestamp", False):
        seed = cfg["Trainer"].get("seed", 42)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = cfg["Trainer"]["output_dir"]
        cfg["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"

    if dist.get_rank() == 0:
        os.makedirs(cfg["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(cfg, osp.join(cfg["Trainer"]["output_dir"], config_name))

    config = OmegaConf.to_container(cfg, resolve=True)
    run(config)


if __name__ == "__main__":
    main()
