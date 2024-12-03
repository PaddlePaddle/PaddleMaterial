import argparse
import os
import os.path as osp
import pickle
import time
from collections import defaultdict

import paddle
import paddle.distributed as dist
from omegaconf import OmegaConf
from pymatgen.io.cif import CifWriter

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.utils import build_structure_from_array
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils import save_load

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)


def predict(model, dataloader, step_lr, log_freq=1, is_save_traj=False):

    """predict function."""
    reader_cost = 0.0
    batch_cost = 0.0
    reader_tic = time.perf_counter()
    batch_tic = time.perf_counter()
    model.eval()

    data_length = len(dataloader)
    logger.info(f"Total Test Steps: {data_length}")
    pred_data_total = {"result": [], "traj": defaultdict(list)}
    for iter_id, batch_data in enumerate(dataloader):
        reader_cost = time.perf_counter() - reader_tic

        with paddle.no_grad():
            pred_data = model.sample(
                batch_data, step_lr=step_lr, is_save_traj=is_save_traj
            )

        pred_data_total["result"].extend(pred_data["result"])
        if is_save_traj:
            for key, value in pred_data["traj"].items():
                pred_data_total["traj"][key].extend(value)

        batch_cost = time.perf_counter() - batch_tic
        if paddle.distributed.get_rank() == 0 and (
            iter_id % log_freq == 0 or iter_id == data_length - 1
        ):
            msg = "Predict: "
            msg += f"Step: [{iter_id+1}/{data_length}]"
            msg += f" | reader cost: {reader_cost:.5f}s"
            msg += f" | batch cost: {batch_cost:.5f}s"
            logger.info(msg)
        batch_tic = time.perf_counter()
        reader_tic = time.perf_counter()
    return pred_data_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./structure_generation/configs/diffcsp_mp20.yaml",
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

    log_freq = config["Global"].get("log_freq", 1)
    is_save_traj = config["Global"].get("is_save_traj", False)
    step_lr = config["Global"].get("step_lr", 0.000005)

    results = predict(
        model,
        predict_loader,
        step_lr=step_lr,
        log_freq=log_freq,
        is_save_traj=is_save_traj,
    )

    # convert results to structures
    structures = build_structure_from_array(results["result"], niggli=False)
    # save structures to cif file
    output_path = osp.join(config["Global"]["output_dir"], "cifs")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Save result to {output_path}")
    for i, structure in enumerate(structures):
        formula = structure.formula.replace(" ", "-")
        tar_file = os.path.join(output_path, f"{formula}_{i + 1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            logger.info(f"{i + 1} Error Structure.")

    # save results
    if is_save_traj:
        save_path = osp.join(config["Global"]["output_dir"], "result_traj.pkl")
        logger.info(f"Save result to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(results["traj"], f)
