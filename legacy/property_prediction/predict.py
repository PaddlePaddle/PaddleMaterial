import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import paddle
import paddle.distributed as dist
import pandas as pd
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import build_post_process
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils import save_load

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)


def predict(model, dataloader, log_freq=1):

    """Eval program for one epoch.

    Args:
        epoch_id (int): Epoch id.
    """
    reader_cost = 0.0
    batch_cost = 0.0
    reader_tic = time.perf_counter()
    batch_tic = time.perf_counter()
    model.eval()

    data_length = len(dataloader)
    all_preds = defaultdict(list)
    for iter_id, batch_data in enumerate(dataloader):
        reader_cost = time.perf_counter() - reader_tic

        with paddle.no_grad():
            pred_data = model(batch_data)

        if post_process_class is not None:
            # since the label data may be not in batch_data, we need to pass it to
            # post_process_class
            pred_data, _ = post_process_class(pred_data)

        batch_cost = time.perf_counter() - batch_tic
        if paddle.distributed.get_rank() == 0 and (
            iter_id % log_freq == 0 or iter_id == data_length - 1
        ):
            msg = "Predict: "
            msg += f"Step: [{iter_id+1}/{data_length}]"
            msg += f" | reader cost: {reader_cost:.5f}s"
            msg += f" | batch cost: {batch_cost:.5f}s"
            logger.info(msg)
        if "id" in batch_data:
            all_preds["id"].extend(batch_data["id"])
        for key, value in pred_data.items():
            all_preds[key].extend(paddle.squeeze(value, axis=-1).tolist())
        batch_tic = time.perf_counter()
        reader_tic = time.perf_counter()
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./property_prediction/configs/megnet_mp18.yaml",
        help="Path to config file",
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

    logger.init_logger(log_file=osp.join(config["Global"]["output_dir"], "predict.log"))
    seed = config["Global"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # build model from config
    model_cfg = config["Model"]
    model = build_model(model_cfg)
    pretrained_model_path = config["Global"].get("pretrained_model_path", None)
    if pretrained_model_path is not None:
        save_load.load_pretrain(model, pretrained_model_path)
    else:
        logger.warning("No pretrained model path provided.")
    # build dataloader from config
    set_signal_handlers()
    predict_data_cfg = config["Dataset"]["predict"]
    predict_loader = build_dataloader(predict_data_cfg)

    # build post processing from config
    post_process_cfg = config["PostProcess"]
    post_process_class = build_post_process(post_process_cfg)

    log_freq = config["Global"].get("log_freq", 1)

    results = predict(model, predict_loader, log_freq=log_freq)

    save_path = osp.join(config["Global"]["output_dir"], "result.csv")
    # save result to csv file use pandas
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

    logger.info(f"Save result to {save_path}")
