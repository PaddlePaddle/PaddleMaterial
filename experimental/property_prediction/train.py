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
import argparse
import os
import os.path as osp
import sys
import datetime

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))  # ruff: noqa

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.datasets.transform import *


def read_independent_dataloader_config(config):
    '''
    Args:
        config (dict): config dict
    ''' 
    if config["Global"].get("do_train", True):
        train_data_cfg = config["Dataset"].get("train")
        assert (
            train_data_cfg is not None
        ), "train_data_cfg must be defined, when do_train is true"
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
        ), "test_data_cfg must be defined, when do_test is true"
        test_loader = build_dataloader(test_data_cfg)
    else:
        test_loader = None
        
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    if dist.get_world_size() > 1:
        fleet.init(is_collective=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml",
        help="Path to config file",
    )

    args, dynamic_args = parser.parse_known_args()

    # load config and merge with cli args
    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    # set random seed
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # add timestamp to output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = config["Trainer"]["output_dir"]
    config["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"

    # save config to output_dir, only rank 0 process will do this
    if dist.get_rank() == 0:
        os.makedirs(config["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Trainer"]["output_dir"], config_name))
    # convert to dict
    config = OmegaConf.to_container(config, resolve=True)

    # init logger
    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")


 
    # build dataloader from config
    set_signal_handlers()
    if config["Dataset"].get("split_dataset_ratio") is not None:
        # Split the dataset into train/val/test and build corresponding dataloaders
        loader = build_dataloader(config["Dataset"])
        train_loader = loader.get("train", None)
        val_loader = loader.get("val", None)
        test_loader = loader.get("test", None)
    else:
        # Use pre-split (independent) train/val/test datasets and build dataloaders
        train_loader, val_loader, test_loader = read_independent_dataloader_config(config)

    # build model from config
    model_cfg = config["Model"]

    # scaling dataset
    if "transform" in config["Dataset"] and config['Global'].get("do_train", False) == True:
        dataset_trans_cfg = config["Dataset"].get("transform")
        if dataset_trans_cfg is not None:
            trans_func = dataset_trans_cfg.pop("__class_name__")
            trans_parms = dataset_trans_cfg.pop("__init_params__")
            logger.info(f"Using transform function: {trans_func}")
        else:
            trans_func = "no_scaling"
            trans_parms = {}
            logger.warning("No transform specified, using 'no_scaling' instead.")
        data_mean, data_std = eval(trans_func)(train_loader, config['Global']['label_names'], **trans_parms)
        logger.info(f"Target is {config['Global']['label_names']}, data mean is {data_mean}, data std is {data_std}")
        
        model_cfg['__init_params__']['data_mean'] = data_mean
        model_cfg['__init_params__']['data_std'] = data_std


    model = build_model(model_cfg)

    # build optimizer and learning rate scheduler from config
    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert (
            train_loader is not None
        ), "train_loader must be defined when optimizer is defined."
        assert (
            config["Trainer"].get("max_epochs") is not None
        ), "max_epochs must be defined when optimizer is defined."
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            config["Trainer"]["max_epochs"],
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

    # # initialize trainer
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
    if config["Global"].get("do_eval", False):
        logger.info("Evaluating on validation set")
        time_info, loss_info, metric_info = trainer.eval(val_loader)
    if config["Global"].get("do_test", False):
        logger.info("Evaluating on test set")
        time_info, loss_info, metric_info = trainer.eval(test_loader)


