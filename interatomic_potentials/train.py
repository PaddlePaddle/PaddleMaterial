# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
import shutil
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.eager_comp_setting import setting_eager_mode

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)


def _collect_qm9_urls(config: Dict) -> list[str]:
    urls = []
    dataset_cfg = config.get("Dataset", {})
    for split in ("train", "val", "test"):
        split_cfg = dataset_cfg.get(split, {})
        ds_cfg = split_cfg.get("dataset", {})
        if ds_cfg.get("__class_name__") != "QM9Dataset":
            continue
        init_params = ds_cfg.get("__init_params__", {})
        url = init_params.get("url", None)
        if isinstance(url, str) and len(url) > 0:
            urls.append(url)
    # Keep order and drop duplicates.
    seen = set()
    dedup_urls = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        dedup_urls.append(url)
    return dedup_urls


def _find_atomref_file(path: str) -> Optional[str]:
    if not path or not osp.exists(path):
        return None

    if osp.isfile(path):
        return path if osp.basename(path) == "atomref.npz" else None

    direct_candidates = [
        osp.join(path, "atomref.npz"),
        osp.join(path, "qm9", "atomref.npz"),
    ]
    for candidate in direct_candidates:
        if osp.exists(candidate):
            return candidate

    for root, _, files in os.walk(path):
        if "atomref.npz" in files:
            return osp.join(root, "atomref.npz")
    return None


def _build_default_qm9_atomref() -> np.ndarray:
    # schnetpack/PyG-compatible QM9 atom references (columns: zpve, U0, U, H, G).
    atomrefs = {
        6: [0.0, 0.0, 0.0, 0.0, 0.0],
        7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
        8: [-13.57459040, -1029.82456413, -1485.26398105, -2042.57270460, -2713.44632457],
        9: [-13.54887564, -1029.79887659, -1485.23829350, -2042.54701705, -2713.42063702],
        10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    }
    atom_ref = np.zeros((100, 5), dtype=np.float32)
    z_list = [1, 6, 7, 8, 9]  # H, C, N, O, F
    for col, key in enumerate([6, 7, 8, 9, 10]):
        values = atomrefs[key]
        for atomic_num, value in zip(z_list, values):
            atom_ref[atomic_num, col] = value
    return atom_ref


def _ensure_schnet_atomref(config: Dict):
    model_cfg = config.get("Model", {})
    if model_cfg.get("__class_name__") != "SchNet":
        return

    model_params = model_cfg.get("__init_params__", {})
    atomref_path = model_params.get("atomref_path", None)
    if not atomref_path:
        return
    if osp.exists(atomref_path):
        return

    qm9_urls = _collect_qm9_urls(config)
    atomref_url = model_params.get("atomref_url", None)
    is_qm9_case = bool(qm9_urls) or ("qm9" in str(atomref_path).lower())
    if isinstance(atomref_url, str) and len(atomref_url) > 0:
        is_qm9_case = True
    if not is_qm9_case:
        # Keep legacy behavior for non-QM9 SchNet cases.
        return

    atomref_dir = osp.dirname(atomref_path) or "."
    os.makedirs(atomref_dir, exist_ok=True)

    candidate_urls = []
    if isinstance(atomref_url, str) and len(atomref_url) > 0:
        candidate_urls.append(atomref_url)
    candidate_urls.extend(qm9_urls)
    candidate_urls.extend(
        [
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.tar.gz",
        ]
    )

    # Keep order and drop duplicates.
    seen = set()
    urls = []
    for url in candidate_urls:
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    for url in urls:
        try:
            if url.endswith(".npz"):
                local_path = download.get_path_from_url(
                    url,
                    atomref_dir,
                    md5sum=None,
                    check_exist=True,
                    decompress=False,
                )
            else:
                local_path = download.get_datasets_path_from_url(url, md5sum=None)

            source_atomref = _find_atomref_file(local_path)
            if source_atomref is None:
                continue

            if osp.abspath(source_atomref) != osp.abspath(atomref_path):
                shutil.copy2(source_atomref, atomref_path)
            logger.info(
                f"Auto prepared missing atomref file: {atomref_path} (source: {source_atomref})"
            )
            return
        except Exception as e:
            logger.warning(f"Failed to auto prepare atomref from {url}: {e}")

    # Final fallback keeps SchNet runnable even when mirror package lacks atomref.
    atomref_np = _build_default_qm9_atomref()
    np.savez(atomref_path, atom_ref=atomref_np)
    logger.warning(
        f"atomref.npz not found in provided mirrors. "
        f"Generated default QM9 atom references at {atomref_path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file",
    )

    args, dynamic_args = parser.parse_known_args()

    # load config and merge with cli args
    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

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

    # set random seed
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # set prim eager mode
    enabled = config["Global"].get("prim_eager_enabled", False)
    white_list = config["Global"].get("prim_backward_white_list", None)
    setting_eager_mode(enabled, white_list)

    # SchNet needs atomref before model construction; auto-prepare if missing.
    _ensure_schnet_atomref(config)

    # build model from config
    model_cfg = config["Model"]
    model = build_model(model_cfg)

    # build dataloader from config
    set_signal_handlers()
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
