from __future__ import annotations

import argparse
import copy
import os
import os.path as osp
import warnings

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf
from paddle.framework import core

from interatomic_potentials.dataset import GraphData  # noqa
from interatomic_potentials.dataset import get_train_val_test_loader
from ppmat.models.chgnet.model.model import CHGNet
from ppmat.optimizer import build_optimizer
from ppmat.utils import logger
from ppmat.utils import misc

EAGER_COMP_OP_BLACK_LIST = [
    "abs_grad",
    "cast_grad",
    # "concat_grad",
    "cos_double_grad",
    "cos_grad",
    "cumprod_grad",
    "cumsum_grad",
    "dropout_grad",
    "erf_grad",
    "exp_grad",
    "expand_grad",
    "floor_grad",
    # "gather_grad",
    "gather_nd_grad",
    "gelu_grad",
    "group_norm_grad",
    "instance_norm_grad",
    # "layer_norm_grad",
    "leaky_relu_grad",
    "log_grad",
    "max_grad",
    "pad_grad",
    "pow_double_grad",
    "pow_grad",
    "prod_grad",
    "relu_grad",
    "roll_grad",
    "rsqrt_grad",
    "scatter_grad",
    "scatter_nd_add_grad",
    "sigmoid_grad",
    "silu_grad",
    "sin_double_grad",
    "sin_grad",
    "slice_grad",
    # "split_grad",
    "sqrt_grad",
    "stack_grad",
    "sum_grad",
    "tanh_double_grad",
    "tanh_grad",
    "topk_grad",
    "transpose_grad",
    "add_double_grad",
    "add_grad",
    "assign_grad",
    "batch_norm_grad",
    "divide_grad",
    "elementwise_pow_grad",
    "maximum_grad",
    "min_grad",
    "minimum_grad",
    "multiply_grad",
    "subtract_grad",
    "tile_grad",
]

enable = True

EAGER_COMP_OP_BLACK_LIST = list(set(EAGER_COMP_OP_BLACK_LIST))
core.set_prim_eager_enabled(enable)
JIT = False
if JIT is False and enable:
    paddle.framework.core._set_prim_backward_blacklist(*EAGER_COMP_OP_BLACK_LIST)


# To suppress warnings for clearer output
warnings.simplefilter("ignore")

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)


def get_dataloader(config):
    dataset_cfg = config["Dataset"]["dataset"]

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_name = dataset_cfg.pop("__name__")
    dataset = eval(dataset_name)(**dataset_cfg)

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset,
        batch_size=config["Dataset"]["sampler"]["batch_size"],
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./interatomic_potentials/chgnet.yaml",
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

    train_loader, val_loader, test_loader = get_dataloader(config)

    model = CHGNet.load()

    loss_class = paddle.nn.MSELoss()
    # build optimizer and learning rate scheduler from config
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"], model, config["Global"]["epochs"], len(train_loader)
    )

    task_key = ["e", "f"]
    for iter_id, batch_data in enumerate(train_loader):
        graphs = batch_data[0]
        pred_data = model(graphs, task="efsm")
        loss = 0
        for key in task_key:
            loss += loss_class(pred_data[key][0], batch_data[1][key][0])

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        logger.info(f"Iter: {iter_id}, Loss: {loss.numpy()}")
