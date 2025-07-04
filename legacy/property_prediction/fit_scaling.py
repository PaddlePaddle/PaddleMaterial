import argparse
import os
import os.path as osp
import shutil

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import build_post_process
from ppmat.losses import build_loss
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.models.gemnet.layers.scaling import AutomaticFit
from ppmat.optimizer import build_optimizer
from ppmat.trainer.trainer import Trainer
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.io import write_json

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./property_prediction/configs/gemnet_mp20.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=1,
        help="Epoch number",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    seed = config["Global"].get("seed", 42)
    misc.set_random_seed(seed)

    if dist.get_rank() == 0:
        os.makedirs(config["Global"]["output_dir"], exist_ok=True)
        try:
            shutil.copy(args.config, config["Global"]["output_dir"])
        except shutil.SameFileError:
            pass

    logger.init_logger(
        log_file=osp.join(config["Global"]["output_dir"], "fit_scalling.log")
    )
    logger.info(f"Set random seed to {seed}")

    comment = (
        config["Model"]["__name__"]
        + "_"
        + config["Dataset"]["val"]["dataset"]["__name__"]
    )

    def init(scale_file):
        preset = {"comment": comment}
        write_json(scale_file, preset)

    scale_file = config["Model"]["scale_file"]
    if os.path.exists(scale_file):
        print("Selected: Overwrite the current file.")
    else:
        init(scale_file)
    AutomaticFit.set2fitmode()

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

    while not AutomaticFit.fitting_completed():
        for i in range(args.epoch):
            loss_dict, metric_dict = trainer.eval()

        current_var = AutomaticFit.activeVar
        if current_var is not None:
            current_var.fit()
        else:
            print("Found no variable to fit. Something went wrong!")
    print(f"\n Fitting done. Results saved to: {scale_file}")
