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

import paddle.distributed as dist
from omegaconf import OmegaConf
from ppmat.modules import build_module

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import build_post_process
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import misc

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "test"]
    )
    args, dynamic_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    if dist.get_rank() == 0:
        os.makedirs(config["Trainer"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Trainer"]["output_dir"], config_name))
    config = OmegaConf.to_container(config, resolve=True)

    logger.init_logger(
        log_file=osp.join(config["Trainer"]["output_dir"], f"{args.mode}.log")
    )
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # build module from config
    module_cfg = config["Module"]
    module = build_module(module_cfg)

    # build dataloader from config
    set_signal_handlers()
    train_data_cfg = config["Dataset"]["train"]
    train_loader = build_dataloader(train_data_cfg)
    val_data_cfg = config["Dataset"]["val"]
    val_loader = build_dataloader(val_data_cfg)
    test_data_cfg = config["Dataset"]["test"]
    test_loader = build_dataloader(test_data_cfg)

    # build post processing from config
    post_process_cfg = config["PostProcess"]
    post_process_class = build_post_process(post_process_cfg)

    # build optimizer and learning rate scheduler from config
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"], module, config["Trainer"]["epochs"], len(train_loader)
    )
    # initialize trainer
    trainer = BaseTrainer(
        config,
        module,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        loss_dict, metric_dict = trainer.eval()
    elif args.mode == "test":
        loss_dict, metric_dict = trainer.test()
