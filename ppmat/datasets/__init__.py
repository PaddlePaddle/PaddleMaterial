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
import copy
import os
import random
import signal
from typing import Dict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import io
from paddle.io import BatchSampler  # noqa
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler  # noqa

from ppmat.datasets import collate_fn
from ppmat.datasets.high_level_water_dataset import HighLevelWaterDataset
from ppmat.datasets.jarvis_dataset import JarvisDataset
from ppmat.datasets.mp20_dataset import AlexMP20MatterGenDataset
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.datasets.mp20_dataset import MP20MatterGenDataset
from ppmat.datasets.mp2018_dataset import MP2018Dataset
from ppmat.datasets.mp2024_dataset import MP2024Dataset
from ppmat.datasets.mptrj_dataset import MPTrjDataset
from ppmat.datasets.num_atom_crystal_dataset import NumAtomsCrystalDataset
from ppmat.datasets.split_mptrj_data import none_to_zero
from ppmat.datasets.transform import build_transforms
from ppmat.utils import logger

__all__ = [
    "MP20Dataset",
    "MP2018Dataset",
    "MP2024Dataset",
    "MP20MatterGenDataset",
    "AlexMP20MatterGenDataset",
    "NumAtomsCrystalDataset",
    "set_signal_handlers",
    "MPTrjDataset",
    "JarvisDataset",
    "HighLevelWaterDataset",
]


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = paddle.get_rng_state()[0].current_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def term_mp(sig_num, frame):
    """kill all child processes"""
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def set_signal_handlers():
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
    except AttributeError:
        # In case `os.getpgid` is not available, no signal handler will be set,
        # because we cannot do safe cleanup.
        pass
    else:
        # XXX: `term_mp` kills all processes in the process group, which in
        # some cases includes the parent process of current process and may
        # cause unexpected results. To solve this problem, we set signal
        # handlers only when current process is the group leader. In the
        # future, it would be better to consider killing only descendants of
        # the current process.
        if pid == pgid:
            # support exit using ctrl+c
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)


def build_dataloader(cfg: Dict):
    """Build dataloader from config.

    Args:
        cfg (Dict): config dictionary.

    Returns:
        paddle.io.DataLoader: paddle.io.DataLoader object.
    """

    if cfg is None:
        return None
    world_size = dist.get_world_size()
    cfg = copy.deepcopy(cfg)

    dataset_cfg = cfg["dataset"]
    cls_name = dataset_cfg.pop("__class_name__")
    init_params = dataset_cfg.pop("__init_params__")

    if "transforms" in init_params:
        init_params["transforms"] = build_transforms(init_params.pop("transforms"))

    dataset = eval(cls_name)(**init_params)

    loader_config = cfg.get("loader")
    if loader_config is None:
        loader_config = {
            "num_workers": 0,
            "use_shared_memory": True,
            "collate_fn": "DefaultCollator",
        }
        logger.message("No loader config is provided, use default config.")
        logger.message("Default loader config: {}".format(loader_config))

    num_workers = loader_config.pop("num_workers", 0)
    use_shared_memory = loader_config.pop("use_shared_memory", True)

    collate_obj = getattr(
        collate_fn, loader_config.pop("collate_fn", "DefaultCollator")
    )()

    # build sampler
    if cfg.get("split_dataset_ratio") is not None:
        ratio_dict = cfg["split_dataset_ratio"]
        ratio_dict = {k: none_to_zero(v) for k, v in ratio_dict.items()}

        if ratio_dict["train"] + ratio_dict["val"] + ratio_dict["test"] != 1.0:
            raise ValueError(
                f"The sum of train_ratio, val_ratio and test_ratio "
                f"should be equal to 1.0, but got "
                f"{ratio_dict['train'] + ratio_dict['val'] + ratio_dict['test']}"
            )

        # split train/valid/test dataset numbers
        total_nums = len(dataset)
        if ratio_dict["test"] == 0:
            train_nums = int(total_nums * ratio_dict["train"])
            val_nums = total_nums - train_nums
            test_nums = 0
        else:
            train_nums = int(total_nums * ratio_dict["train"])
            val_nums = int(total_nums * ratio_dict["val"])
            test_nums = total_nums - train_nums - val_nums
        logger.info(
            f"Number of train, val and test dataset "
            f"are {train_nums}, {val_nums} and {test_nums}."
        )

        train_dataset, val_dataset, test_dataset = io.random_split(
            dataset, [train_nums, val_nums, test_nums]
        )
        dataset_dict = {
            "train": train_dataset if len(train_dataset) != 0 else None,
            "val": val_dataset if len(val_dataset) != 0 else None,
            "test": test_dataset if len(test_dataset) != 0 else None,
        }

        data_loader_dict = {}
        for data_name, dataset in dataset_dict.items():
            if dataset is None:
                data_loader_dict[data_name] = None
                continue
            sampler_cfg = cfg.get(f"{data_name}_sampler", None)
            batch_sampler = set_build_sample(sampler_cfg, world_size, dataset)
            data_loader_dict[data_name] = DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                return_list=True,
                use_shared_memory=use_shared_memory,
                collate_fn=collate_obj,
                worker_init_fn=worker_init_fn,
                **loader_config,
            )

        return data_loader_dict

    else:
        sampler_cfg = cfg.get("sampler", None)
        batch_sampler = set_build_sample(sampler_cfg, world_size, dataset)

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            collate_fn=collate_obj,
            worker_init_fn=worker_init_fn,
            **loader_config,
        )
        return data_loader


def set_build_sample(sampler_cfg, world_size, dataset):
    if sampler_cfg is not None:
        batch_sampler_cls = sampler_cfg.pop("__class_name__")
        init_params = sampler_cfg.pop("__init_params__")

        if batch_sampler_cls == "BatchSampler":
            if world_size > 1:
                batch_sampler_cls = "DistributedBatchSampler"
                logger.warning(
                    f"Automatically use 'DistributedBatchSampler' instead of "
                    f"'BatchSampler' when world_size({world_size}) > 1."
                )

        batch_sampler = getattr(io, batch_sampler_cls)(dataset, **init_params)
    else:
        batch_sampler_cls = "BatchSampler"
        if world_size > 1:
            batch_sampler_cls = "DistributedBatchSampler"
            logger.warning(
                f"Automatically use 'DistributedBatchSampler' instead of "
                f"'BatchSampler' when world_size({world_size}) > 1."
            )
        batch_sampler = getattr(io, batch_sampler_cls)(
            dataset,
            batch_size=init_params["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        logger.message(
            "'shuffle' and 'drop_last' are both set to False in default as sampler "
            "config is not specified."
        )

    return batch_sampler
