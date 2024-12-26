import argparse
import os
import os.path as osp

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import build_post_process
from ppmat.losses import build_loss
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.trainer import Trainer
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
        default="./property_prediction/configs/megnet_mp18.yaml",
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
        os.makedirs(config["Global"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Global"]["output_dir"], config_name))

    config = OmegaConf.to_container(config, resolve=True)

    logger.init_logger(
        log_file=osp.join(config["Global"]["output_dir"], f"{args.mode}.log")
    )
    seed = config["Global"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # build model from config
    model_cfg = config["Model"]
    model = build_model(model_cfg)

    # build dataloader from config
    set_signal_handlers()
    train_data_cfg = config["Dataset"]["train"]
    train_loader = build_dataloader(train_data_cfg)
    val_data_cfg = config["Dataset"]["val"]
    val_loader = build_dataloader(val_data_cfg)
    test_data_cfg = config["Dataset"]["test"]
    test_loader = build_dataloader(test_data_cfg)

    # build loss from config
    loss_cfg = config["Loss"]
    loss_class = build_loss(loss_cfg)

    # build post processing from config
    post_process_cfg = config["PostProcess"]
    post_process_class = build_post_process(post_process_cfg)

    # build metric from config
    metric_cfg = config["Metric"]
    metric_class = build_metric(metric_cfg)

    # build optimizer and learning rate scheduler from config
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"], model, config["Global"]["epochs"], len(train_loader)
    )
    # initialize trainer
    trainer = Trainer(
        config,
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_class=loss_class,
        metric_class=metric_class,
        post_process_class=post_process_class,
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        loss_dict, metric_dict = trainer.eval()
    elif args.mode == "test":
        loss_dict, metric_dict = trainer.test()
    # loss_dict, metric_dict = trainer.test()
