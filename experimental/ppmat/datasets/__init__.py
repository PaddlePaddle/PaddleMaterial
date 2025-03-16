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

import numpy as np
import paddle
from paddle.io import BatchSampler  # noqa
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler  # noqa

from ppmat.datasets import collate_fn
from ppmat.datasets.mp2018_dataset import MP2018Dataset
from ppmat.datasets.mp_base_dataset import MPBaseDataset
from ppmat.datasets.transform import build_transforms
from ppmat.utils import logger

__all__ = [
    "MPBaseDataset",
    "MP2018Dataset",
    "set_signal_handlers",
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


def build_dataloader(cfg):
    cfg = copy.deepcopy(cfg)

    dataset_cfg = cfg["dataset"]
    cls_name = dataset_cfg.pop("__name__")
    if "transforms" in dataset_cfg:
        dataset_cfg["transforms"] = build_transforms(dataset_cfg.pop("transforms"))
    dataset = eval(cls_name)(**dataset_cfg)

    loader_config = cfg.get("loader")
    if loader_config is None:
        loader_config = {
            "num_workers": 0,
            "use_shared_memory": True,
            "collate_fn": "DefaultCollator",
        }
        logger.message("No loader config is provided, use default config.")
        logger.message("Default loader config: {}".format(loader_config))

    num_workers = loader_config.get("num_workers", 0)
    use_shared_memory = loader_config.get("use_shared_memory", True)

    sampler_cfg = cfg["sampler"]
    cls_name = sampler_cfg.pop("__name__")
    batch_sampler = eval(cls_name)(dataset, **sampler_cfg)

    collate_obj = getattr(
        collate_fn, loader_config.get("collate_fn", "DefaultCollator")
    )()

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        collate_fn=collate_obj,
        worker_init_fn=worker_init_fn,
    )

    return data_loader
