# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import datetime
import math
import os
import os.path as osp

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.eager_comp_setting import setting_eager_mode


def read_independent_dataloader_config(config):
    """Build train/val/test dataloaders when datasets are defined independently."""
    if config["Global"].get("do_train", True):
        train_data_cfg = config["Dataset"].get("train")
        assert (
            train_data_cfg is not None
        ), "train dataset must be defined when do_train is True"
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
        assert (
            test_data_cfg is not None
        ), "test dataset must be defined when do_test is True"
        test_loader = build_dataloader(test_data_cfg)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file",
    )

    args, dynamic_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    # set random seed
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # add timestamp to output_dir for reproducible runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = config["Trainer"]["output_dir"]
    config["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"

    # save config to output_dir, only rank 0 process will do this
    if dist.get_rank() == 0:
        os.makedirs(config["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Trainer"]["output_dir"], config_name))

    # convert to dict for downstream usage
    config = OmegaConf.to_container(config, resolve=True)

    # init logger
    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")

    # enable primitive eager mode when requested
    enabled = config["Global"].get("prim_eager_enabled", False)
    white_list = config["Global"].get("prim_backward_white_list", None)
    setting_eager_mode(enabled, white_list)

    # build model from config
    model_cfg = config["Model"]
    model = build_model(model_cfg)

    # Optionally map legacy Trainer.*_samples to DensityCollator.n_samples
    trainer_cfg = config["Trainer"]
    dataset_cfg = config.get("Dataset", {})

    def _maybe_fill_n_samples(split_key: str, trainer_key: str) -> None:
        split_cfg = dataset_cfg.get(split_key)
        if split_cfg is None:
            return
        loader_cfg = split_cfg.get("loader")
        if loader_cfg is None:
            return
        collate_fn = loader_cfg.get("collate_fn")
        # Only touch DensityCollator-based loaders and string-form collate_fn.
        if not isinstance(collate_fn, str) or not collate_fn.startswith(
            "DensityCollator"
        ):
            return
        target_samples = trainer_cfg.get(trainer_key)
        if target_samples is None:
            return
        collate_params = loader_cfg.setdefault("collate_params", {})
        # Respect explicit n_samples in loader.collate_params if already set.
        if collate_params.get("n_samples") is None:
            collate_params["n_samples"] = target_samples

    _maybe_fill_n_samples("train", "train_samples")
    _maybe_fill_n_samples("val", "val_samples")

    # build dataloader from config
    set_signal_handlers()
    if config["Dataset"].get("split_dataset_ratio") is not None:
        loader = build_dataloader(config["Dataset"])
        train_loader = loader.get("train", None)
        val_loader = loader.get("val", None)
        test_loader = loader.get("test", None)
    else:
        train_loader, val_loader, test_loader = read_independent_dataloader_config(
            config
        )

    # build optimizer and learning rate scheduler from config
    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert (
            train_loader is not None
        ), "train_loader must be defined when optimizer is defined."
        trainer_cfg = config["Trainer"]

        # If a max_iter (global step budget) is provided, convert it into
        # an effective max_epochs so that BaseTrainer can see enough epochs
        # while still respecting the global step limit internally.
        max_iter = trainer_cfg.get("max_iter", None)
        if max_iter is not None and max_iter > 0:
            grad_accum = trainer_cfg.get("gradient_accumulation_steps", 1)
            steps_per_epoch = (
                len(train_loader) // grad_accum
                if (len(train_loader) and grad_accum)
                else 0
            )
            if len(train_loader) % grad_accum != 0:
                steps_per_epoch += 1
            if steps_per_epoch > 0:
                required_epochs = math.ceil(max_iter / steps_per_epoch)
                orig_epochs = trainer_cfg.get("max_epochs", required_epochs)
                if required_epochs > orig_epochs:
                    logger.info(
                        "Adjusting Trainer.max_epochs from %d to %d based on "
                        "max_iter=%d and steps_per_epoch=%d.",
                        orig_epochs,
                        required_epochs,
                        max_iter,
                        steps_per_epoch,
                    )
                trainer_cfg["max_epochs"] = max(orig_epochs, required_epochs)

        # Map optional Trainer.max_grad_norm to optimizer grad_clip if not set
        max_grad_norm = trainer_cfg.get("max_grad_norm", None)
        opt_cfg = config["Optimizer"]
        if (
            max_grad_norm is not None
            and not any(
                k in opt_cfg for k in ("clip_norm", "clip_norm_global", "clip_value")
            )
        ):
            opt_cfg["clip_norm_global"] = max_grad_norm

        assert (
            trainer_cfg.get("max_epochs") is not None
        ), "max_epochs must be defined when optimizer is defined."

        optimizer, lr_scheduler = build_optimizer(
            opt_cfg,
            model,
            trainer_cfg["max_epochs"],
            len(train_loader),
        )
    else:
        optimizer, lr_scheduler = None, None

    # build metric from config
    metric_cfg = config.get("Metric")
    if metric_cfg is not None:
        metric_func = build_metric(metric_cfg)
    else:
        metric_func = None

    trainer = BaseTrainer(
        config["Trainer"],
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        compute_metric_func_dict=metric_func,
    )

    if config["Global"].get("do_train", True):
        trainer.train()
    if config["Global"].get("do_eval", False) and val_loader is not None:
        logger.info("Evaluating on validation set")
        trainer.eval(val_loader)
    if config["Global"].get("do_test", False) and test_loader is not None:
        logger.info("Evaluating on test set")
        trainer.eval(test_loader)
