import os
import os.path as osp
import random
from typing import  Mapping, TypeVar

import numpy as np
import paddle
import paddle.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mattergen.diffusion.trainer import TrainerDiffusion
from mattergen.utils import logger
from mattergen.common.data.utils import set_signal_handlers
from paddle_utils import *

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)


T = TypeVar("T")


def maybe_instantiate(instance_or_config: T | Mapping, expected_type=None, **kwargs) -> T:
    """
    If instance_or_config is a mapping with a _target_ field, instantiate it.
    Otherwise, return it as is.
    """
    if isinstance(instance_or_config, Mapping) and "_target_" in instance_or_config:
        instance = instantiate(instance_or_config, **kwargs)
    else:
        instance = instance_or_config
    assert expected_type is None or isinstance(
        instance, expected_type
    ), f"Expected {expected_type}, got {type(instance)}"
    return instance


def main(config, seed: int | None = None):
    """
    Main entry point to train and evaluate a diffusion model.

    save_config: if True, the config will be saved both as a YAML file and in each 
    checkpoint. This doesn't work if the config contains things that can't be 
    `yaml.dump`-ed, so if you don't care about saving and loading checkpoints and want 
    to use a config that contains things like `paddle.nn.Layer`s already instantiated, 
    set this to False.
    """

    if dist.get_rank() == 0:
        os.makedirs(config.trainer.output_dir, exist_ok=True)
        OmegaConf.save(config, osp.join(config.trainer.output_dir, "config.yaml"))

    set_signal_handlers()
    logger.init_logger(
        log_file=osp.join(config.trainer.output_dir, f"{config.trainer.mode}.log")
    )
    seed = seed or config.trainer.seed
    if seed is not None:
        paddle.seed(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
    logger.info(f"Seeding everything with {seed}")

    model = maybe_instantiate(config.lightning_module.diffusion_module)
    datamodule = maybe_instantiate(config.data_module)

    optimizer_cfg = config.lightning_module.optimizer_partial
    optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
    optimizer_cfg.update(
        dict(
            model_list=model,
            epochs=config.trainer.max_epochs,
            iters_per_epoch=len(datamodule.train_dataloader()),
        )
    )

    optimizer, lr_scheduler = maybe_instantiate(optimizer_cfg)

    trainer = TrainerDiffusion(
        config=config,
        model=model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        test_dataloader=datamodule.test_dataloader(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    if config.trainer.mode == "train":
        trainer.train()
    elif config.trainer.mode == "eval":
        if dist.get_rank == 0:
            trainer.eval()
    elif config.trainer.mode == "test":
        if dist.get_rank == 0:
            trainer.test()
