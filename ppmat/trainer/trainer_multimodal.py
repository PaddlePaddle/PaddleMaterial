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

import os
import sys
import time
from collections import defaultdict
from typing import Callable
from typing import Optional

# import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from packaging import version
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet

from ppmat.models.digress import diffusion_utils

# from ppmat.models.digress.noise_schedule import DiscreteUniformTransition
# from ppmat.models.digress.noise_schedule import MarginalUniformTransition
# from ppmat.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete
from ppmat.models.digress.utils import digressutils as utils

# from ppmat.trainer.trainer_diffusion import TrainerDiffusion
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.io import read_json  # noqa
from ppmat.utils.io import write_json

# from ppmat.models.digress.utils.diffusionprior_utils import l2norm


# from ppmat.models.digress.utils.diffusionprior_utils import l2norm


# from ppmat.models.digress.base_model import ContrastGraphTransformer


class TrainerGraph:
    """Class for Trainer."""

    def __init__(
        self,
        config,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        test_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        metric_class: Optional[Callable] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metric_class = metric_class
        self.lr_scheduler = lr_scheduler

        # get config from config file
        self.epochs = config["Trainer"]["epochs"]
        self.output_dir = config["Tracker"]["save"]["output_dir"]
        self.save_freq = config["Tracker"]["save"]["save_freq"]
        self.log_freq = config["Tracker"]["log"]["log_freq"]
        self.start_eval_epoch = config["Trainer"]["start_eval_epoch"]
        self.eval_freq = config["Trainer"]["eval_freq"]
        self.seed = config["Trainer"]["seed"]
        self.pretrained_model_path = config["Trainer"].get(
            "pretrained_model_path", None
        )
        self.checkpoint_path = config["Trainer"].get("checkpoint_path", None)
        self.scale_grad = config["Trainer"].get("scale_grad", False)
        self.is_save_traj = config["Tracker"]["save"].get("is_save_traj", False)
        self.step_lr = config["Trainer"].get("step_lr", 0.000005)

        self.iters_per_epoch = len(self.train_dataloader)

        if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            if (
                self.config["Trainer"]["Optimizer"]["lr"].get("indicator", "train_loss")
                == "eval_loss"
            ):
                assert self.eval_freq == 1, (
                    "ReduceOnPlateau only support eval_freq==1 when indicator="
                    "'eval_loss'"
                )
                assert self.lr_scheduler.by_epoch is True, (
                    "ReduceOnPlateau only support by_epoch=True, when indicator="
                    "'eval_loss"
                )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # initialize distributed environment
        if self.world_size > 1:
            fleet.init(is_collective=True)
            logger.warning(
                f"Detected 'world_size'({self.world_size}) > 1, it is recommended to "
                "scale up the learning rate and reduce the 'epochs' or "
                "'iters_per_epoch' according to the 'world_size' both linearly if you "
                "are training model."
            )

        # load pretrained model, usually used for transfer learning
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(self.model, self.pretrained_model_path)

        # initialize an dict for tracking best metric during training
        self.best_metric = {
            "loss": float("inf"),
            "epoch": 0,
        }
        # load model checkpoint, usually used for resume training
        if self.checkpoint_path is not None:
            if self.pretrained_model_path is not None:
                logger.warning(
                    "Detected 'pretrained_model_path' is given, weights in which might"
                    " be overridden by weights loaded from given 'checkpoint_path'."
                )
            loaded_metric = save_load.load_checkpoint(
                self.checkpoint_path,
                self.model,
                self.optimizer,
            )
            if isinstance(loaded_metric, dict):
                self.best_metric.update(loaded_metric)

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.model, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'Solver.__init__' and keep it's type as 'nn.Layer'."
                )

            self.model = fleet.distributed_model(self.model)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0
        self.log_paddle_version()

    def log_paddle_version(self):
        # log paddlepaddle's version
        if version.Version(paddle.__version__) != version.Version("0.0.0"):
            paddle_version = paddle.__version__
            if version.Version(paddle.__version__) < version.Version("2.6.0"):
                logger.warning(
                    f"Detected paddlepaddle version is '{paddle_version}', "
                    "currently it is recommended to use release 2.6 or develop version."
                )
        else:
            paddle_version = f"develop({paddle.version.commit[:7]})"

        logger.info(f"Using paddlepaddle {paddle_version}")

    @paddle.no_grad()
    def eval_epoch(self, dataloader, epoch_id: int):
        """Eval program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.eval()
        total_loss = defaultdict(list)
        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            loss_dict = self.model(batch_data)

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            batch_cost = time.perf_counter() - batch_tic
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Epoch [{epoch_id}/{self.epochs}] "
                msg += f"| Step: [{iter_id+1}/{data_length}]"
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    def train_epoch(self, dataloader, epoch_id: int):
        """Train program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.train()
        total_loss = defaultdict(list)

        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            loss_dict = self.model(batch_data)

            loss = loss_dict["loss"]
            loss.backward()
            # if self.scale_grad:
            #     # TODO: no scale_shared_grads defined here!!!
            #     scale_shared_grads(self.model)

            self.optimizer.step()
            self.optimizer.clear_grad()

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            # if solver.world_size > 1:
            #     # fuse + allreduce manually before optimization if use DDP + no_sync
            #     # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
            #     hpu.fused_allreduce_gradients(list(self.model.parameters()), None)
            # update learning rate by step
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                    if (
                        self.config["Optimizer"]["lr"].get("indicator", "train_loss")
                        == "train_loss"
                    ):
                        train_loss = loss_dict["loss"]
                        train_loss = paddle.to_tensor(train_loss)
                        if self.world_size > 1:
                            dist.all_reduce(train_loss)
                            train_loss = train_loss / self.world_size
                        self.lr_scheduler.step(train_loss)
                else:
                    self.lr_scheduler.step()

            batch_cost = time.perf_counter() - batch_tic
            # update and log training information
            self.global_step += 1
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | lr: {self.optimizer._learning_rate():.6f}".rstrip("0")
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    def train(self) -> None:
        """Training."""
        self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = self.best_metric["epoch"] + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch_id)

            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for k, v in train_loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
            logger.info(msg)
            save_metric_dict = {"epoch": epoch_id}
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_dict = self.eval_epoch(self.val_dataloader, epoch_id)
                save_metric_dict.update(eval_loss_dict)

                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                for k, v in eval_loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

                # update best metric
                if eval_loss_dict["loss"] <= self.best_metric["loss"]:
                    self.best_metric.update(eval_loss_dict)
                    self.best_metric["epoch"] = epoch_id

                    save_load.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.best_metric,
                        output_dir=self.output_dir,
                        prefix="best",
                    )
            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                    if (
                        self.config["Trainer"]["Optimizer"]["lr"].get(
                            "indicator", "train_loss"
                        )
                        == "train_loss"
                    ):
                        train_loss = train_loss_dict["loss"]
                        train_loss = paddle.to_tensor(train_loss)
                        if self.world_size > 1:
                            dist.all_reduce(train_loss)
                            train_loss = train_loss / self.world_size
                        self.lr_scheduler.step(train_loss)
                    else:
                        eval_loss = paddle.to_tensor(0.0)
                        if dist.get_rank() == 0:
                            eval_loss = paddle.to_tensor(eval_loss_dict["loss"])
                        if self.world_size > 1:
                            for rank_id in range(self.world_size):
                                dist.broadcast(eval_loss, src=rank_id)
                        self.lr_scheduler.step(eval_loss)
                else:
                    self.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.save_freq > 0 and epoch_id % self.save_freq == 0:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    save_metric_dict,
                    output_dir=self.output_dir,
                    prefix=f"epoch_{epoch_id}",
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                save_metric_dict,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(epoch_id == start_epoch),
            )

    def eval(self):
        loss_dict = self.eval_epoch(self.val_dataloader, epoch_id=1)
        msg = "Eval: "
        for k, v in loss_dict.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
        logger.info(msg)
        return loss_dict

    def test(self):
        dataloader = self.test_dataloader
        is_save_traj = self.is_save_traj

        step_lr = self.step_lr
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.eval()

        data_length = len(dataloader)
        logger.info(f"Total Test Steps: {data_length}")
        pred_data_total = {"result": [], "traj": defaultdict(list)}
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            pred_data = self.model.sample(
                batch_data, step_lr=step_lr, is_save_traj=is_save_traj
            )
            pred_data_total["result"].extend(pred_data["result"])
            if is_save_traj:
                for key, value in pred_data["traj"].items():
                    pred_data_total["traj"][key].extend(value)

            batch_cost = time.perf_counter() - batch_tic
            # we set to log information only when rank 0 every step
            if paddle.distributed.get_rank() == 0:
                msg = "Test:"
                msg += f" Step: [{iter_id+1}/{data_length}]"
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                msg += f" | eta: {(batch_cost*(data_length-(iter_id+1))):.5f}s"
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        save_path = os.path.join(self.output_dir, "test_result.json")
        # pred_data_total = read_json(save_path)
        write_json(save_path, pred_data_total)
        logger.info(f"Test Result Saved to {save_path}")
        metric_dict = self.metric_class(pred_data_total["result"])
        logger.info(f"Test Metric: {metric_dict}")
        return pred_data_total, metric_dict


class TrainerCLIP:
    def __init__(
        self,
        config,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        metric_class,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
        # dataset_infos,
        # train_metrics,
        # sampling_metrics,
        # visualization_tools,
        # extra_features,
        # domain_features,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.metric_class = metric_class
        self.lr_scheduler = lr_scheduler

        # get config from config file
        self.epochs = config["Trainer"]["epochs"]
        self.output_dir = config["Tracker"]["save"]["output_dir"]
        self.save_freq = config["Tracker"]["save"]["save_freq"]
        self.log_freq = config["Tracker"]["log"]["log_freq"]
        self.start_eval_epoch = config["Trainer"]["start_eval_epoch"]
        self.eval_freq = config["Trainer"]["eval_freq"]
        self.seed = config["Trainer"]["seed"]
        self.pretrained_model_path = config["Trainer"].get(
            "pretrained_model_path", None
        )
        self.checkpoint_path = config["Trainer"].get("checkpoint_path", None)
        self.cal_metric_during_train = config["Trainer"]["cal_metric_during_train"]
        self.scale_grad = config["Trainer"].get("scale_grad", False)

        self.iters_per_epoch = len(self.train_dataloader)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # initialize distributed environment
        if self.world_size > 1:
            fleet.init(is_collective=True)
            logger.warning(
                f"Detected 'world_size'({self.world_size}) > 1, it is recommended to "
                "scale up the learning rate and reduce the 'epochs' or "
                "'iters_per_epoch' according to the 'world_size' both linearly if you "
                "are training model."
            )

        # load pretrained model, usually used for transfer learning
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(self.model, self.pretrained_model_path)

        # load model checkpoint, usually used for resume training
        if self.checkpoint_path is not None:
            if self.pretrained_model_path is not None:
                logger.warning(
                    "Detected 'pretrained_model_path' is given, weights in which might"
                    " be overridden by weights loaded from given 'checkpoint_path'."
                )

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.model, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'Solver.__init__' and keep it's type as 'nn.Layer'."
                )

            self.model = fleet.distributed_model(self.model)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0
        self.log_paddle_version()

    def log_paddle_version(self):
        # log paddlepaddle's version
        if version.Version(paddle.__version__) != version.Version("0.0.0"):
            paddle_version = paddle.__version__
            if version.Version(paddle.__version__) < version.Version("2.6.0"):
                logger.warning(
                    f"Detected paddlepaddle version is '{paddle_version}', "
                    "currently it is recommended to use release 2.6 or develop version."
                )
        else:
            paddle_version = f"develop({paddle.version.commit[:7]})"

        logger.info(f"Using paddlepaddle {paddle_version}")

    def forward(self, noisy_data, extra_data, node_mask, X, E, condition):
        # 拼接
        X_ = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype("float32")
        E_ = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype("float32")
        y_ = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype("float32")

        # 把 condition 转到 Paddle Tensor
        # 如果是 int64 -> 'int64', or as needed
        condition_tensor = paddle.to_tensor(condition, dtype="int64")

        # 调用 self.model
        return self.model(X_, E_, y_, node_mask, X, E, condition_tensor)

    def train_epoch(self, dataloader, epoch_id: int):
        """Train program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.train()

        total_loss = defaultdict(list)
        # import pdb

        # pdb.set_trace()
        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            loss_dict = self.model(batch_data)
            loss = loss_dict["loss"]
            loss.backward()

            self.optimizer.step()
            self.optimizer.clear_grad()

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            # if solver.world_size > 1:
            #     # fuse + allreduce manually before optimization if use DDP + no_sync
            #     # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
            #     hpu.fused_allreduce_gradients(list(self.model.parameters()), None)
            # update learning rate by step
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            batch_cost = time.perf_counter() - batch_tic
            # update and log training information
            self.global_step += 1
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | lr: {self.optimizer.get_lr():.6f}".rstrip("0")
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )

                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    @paddle.no_grad()
    def eval_epoch(self, dataloader, epoch_id: int):
        """Eval program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.eval()
        self.metric_class.reset()
        total_loss = defaultdict(list)
        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            loss_dict = self.model(batch_data)

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            batch_cost = time.perf_counter() - batch_tic
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Epoch [{epoch_id}/{self.epochs}] "
                msg += f"| Step: [{iter_id+1}/{data_length}]"
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    def train(self) -> None:
        """Training."""
        # self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = 0 + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch_id)

            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for k, v in train_loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
            logger.info(msg)
            save_metric_dict = {"epoch": epoch_id}
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_dict = self.eval_epoch(self.val_dataloader, epoch_id)

                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                for k, v in eval_loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.save_freq > 0 and epoch_id % self.save_freq == 0:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    save_metric_dict,
                    output_dir=self.output_dir,
                    prefix=f"epoch_{epoch_id}",
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                save_metric_dict,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(epoch_id == start_epoch),
            )

    # --------------------------
    # 训练阶段
    # --------------------------
    def train_step(self, data, i):

        if data.edge_index.size(0) == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        # 前向
        predV1, predV2 = self.forward(
            noisy_data, extra_data, node_mask, X, E, conditionAll
        )

        # L2 normalize
        V1_e = F.normalize(predV1, p=2, axis=1)
        V2_e = F.normalize(predV2, p=2, axis=1)

        # 矩阵相乘 => (bs, bs)
        # 原: torch.matmul(V1_e, V2_e.T)*exp(torch.tensor(self.tem))
        temperature = paddle.to_tensor(self.tem, dtype=V1_e.dtype)
        logits = paddle.matmul(V1_e, V2_e, transpose_y=True) * paddle.exp(temperature)

        # 交叉熵损失
        n = V1_e.shape[0]
        labels = paddle.arange(0, n, dtype="int64")  # (bs,)
        loss_fn = nn.CrossEntropyLoss()

        # loss_v1
        loss_v1 = loss_fn(logits, labels)
        # loss_v2 => 对称
        loss_v2 = loss_fn(logits.transpose([1, 0]), labels)

        loss = (loss_v1 + loss_v2) / 2.0

        if i % 100 == 0:
            print(f"train_loss: {loss.numpy()}")

        return {"loss": loss}

    # --------------------------
    # 验证阶段
    # --------------------------
    def val_step(self, data, i):

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        predV1, predV2 = self.forward(
            noisy_data, extra_data, node_mask, X, E, conditionAll
        )
        V1_e = F.normalize(predV1, p=2, axis=1)
        V2_e = F.normalize(predV2, p=2, axis=1)

        temperature = paddle.to_tensor(self.tem, dtype=V1_e.dtype)
        logits = paddle.matmul(V1_e, V2_e, transpose_y=True) * paddle.exp(temperature)

        n = V1_e.shape[0]
        labels = paddle.arange(0, n, dtype="int64")
        loss_fn = nn.CrossEntropyLoss()

        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.transpose([1, 0]), labels)
        loss = (loss_v1 + loss_v2) / 2.0

        self.val_loss.append(loss)

        if i % 8 == 0:
            print(f"val_loss: {loss.numpy()}")

        return {"loss": loss}

    # --------------------------
    # 测试阶段
    # --------------------------
    def test_step(self, data, i):

        # 可根据需求实现
        pass

    # --------------------------
    # apply_noise
    # --------------------------
    def apply_noise(self, X, E, y, node_mask):
        bs = X.shape[0]
        # t_int in [1, T]
        t_int = paddle.randint(low=1, high=self.T + 1, shape=[bs, 1], dtype="int64")
        t_int = t_int.astype("float32")
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=None)
        # probX = X @ Qtb.X => paddle.matmul(X, Qtb.X)
        probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).astype("float32").mask(node_mask)

        noisy_data = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }
        return noisy_data

    def compute_extra_data(self, noisy_data):
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = paddle.concat([extra_features.X, extra_molecular_features.X], axis=-1)
        extra_E = paddle.concat([extra_features.E, extra_molecular_features.E], axis=-1)
        extra_y = paddle.concat([extra_features.y, extra_molecular_features.y], axis=-1)

        t = noisy_data["t"]
        extra_y = paddle.concat([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # --------------------------
    # 可选的一些回调
    # --------------------------
    def on_train_epoch_start(self):
        print("Starting train epoch...")

    def on_train_epoch_end(self):
        # 这里可以做一些清理或日志记录
        sys.stdout.flush()

    def on_validation_epoch_start(self):
        self.val_loss = []

    def on_validation_epoch_end(self):
        val_loss_sum = paddle.add_n([v for v in self.val_loss])  # or sum(self.val_loss)
        # sum(...) => 需要是相同dtype
        val_loss_val = (
            val_loss_sum.numpy()[0]
            if len(val_loss_sum.shape) > 0
            else val_loss_sum.numpy()
        )
        print(f"Epoch {0} : Val Loss {val_loss_val:.2f}")  # 或 self.current_epoch

    def on_test_epoch_start(self):
        pass

    def on_test_epoch_end(self):
        print("Done testing.")


class TrainerDiffusionPrior:
    def __init__(
        self,
        config,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        test_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
    ):
        super().__init__()

        # 初始化TrainerDiffusionPrior类的实例变量
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 训练参数及日志设置，从config中读取
        self.epochs = config["Trainer"]["epochs"]
        self.output_dir = config["Tracker"]["save"]["output_dir"]
        self.save_freq = config["Tracker"]["save"]["save_freq"]
        self.log_freq = config["Tracker"]["log"]["log_freq"]
        self.start_eval_epoch = config["Trainer"]["start_eval_epoch"]
        self.eval_freq = config["Trainer"]["eval_freq"]
        self.seed = config["Trainer"]["seed"]
        self.pretrained_model_path = config["Trainer"].get(
            "pretrained_model_path", None
        )
        self.checkpoint_path = config["Tracker"].get("checkpoint_path", None)
        self.scale_grad = config["Trainer"].get("scale_grad", False)
        self.iters_per_epoch = len(self.train_dataloader)

        # load pretrained model, usually used for transfer learning
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(self.model, self.pretrained_model_path)

        # load model checkpoint, usually used for resume training
        if self.checkpoint_path is not None:
            if self.pretrained_model_path is not None:
                logger.warning(
                    "Detected 'pretrained_model_path' is given, weights in which might"
                    " be overridden by weights loaded from given 'checkpoint_path'."
                )

        # 获取当前进程的rank信息
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # initialize distributed environment
        if self.world_size > 1:
            fleet.init(is_collective=True)
            logger.warning(
                f"Detected 'world_size'({self.world_size}) > 1, it is recommended to "
                "scale up the learning rate and reduce the 'epochs' or "
                "'iters_per_epoch' according to the 'world_size' both linearly if you "
                "are training model."
            )

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.model, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'Solver.__init__' and keep it's type as 'nn.Layer'."
                )

            self.model = fleet.distributed_model(self.model)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0
        logger.info(
            f"Initialized DiffusionPriorTrainer: rank {self.rank} / \
                world_size {self.world_size}"
        )
        self.log_paddle_version()

    def log_paddle_version(self):
        # log paddlepaddle's version
        if version.Version(paddle.__version__) != version.Version("0.0.0"):
            paddle_version = paddle.__version__
            if version.Version(paddle.__version__) < version.Version("2.6.0"):
                logger.warning(
                    f"Detected paddlepaddle version is '{paddle_version}', "
                    "currently it is recommended to use release 2.6 or develop version."
                )
        else:
            paddle_version = f"develop({paddle.version.commit[:7]})"

        logger.info(f"Using paddlepaddle {paddle_version}")

    def train(self) -> None:
        """Training."""
        # self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = 0 + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch_id)

            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for k, v in train_loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
            logger.info(msg)
            save_metric_dict = {"epoch": epoch_id}
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_dict = self.eval_epoch(self.val_dataloader, epoch_id)

                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                for k, v in eval_loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.save_freq > 0 and epoch_id % self.save_freq == 0:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    save_metric_dict,
                    output_dir=self.output_dir,
                    prefix=f"epoch_{epoch_id}",
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                save_metric_dict,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(epoch_id == start_epoch),
            )

    def train_epoch(self, dataloader, epoch_id: int):
        """Train program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.train()
        total_loss = defaultdict(list)
        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            clip_graph_embeds, clip_text_embeds = self.model.generate_embed_vector(
                batch_data
            )

            loss_dict = self.model(
                text_embed=clip_text_embeds, moleculargraph_embed=clip_graph_embeds
            )
            loss = loss_dict["loss"]
            loss.backward()

            self.optimizer.step()
            self.optimizer.clear_grad()

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            # if solver.world_size > 1:
            #     # fuse + allreduce manually before optimization if use DDP + no_sync
            #     # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
            #     hpu.fused_allreduce_gradients(list(self.model.parameters()), None)
            # update learning rate by step
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            batch_cost = time.perf_counter() - batch_tic
            # update and log training information
            self.global_step += 1
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | lr: {self.optimizer.get_lr():.6f}".rstrip("0")
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )

                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    @paddle.no_grad()
    def eval_epoch(self, dataloader, epoch_id: int):
        """Eval program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        reader_cost = 0.0
        batch_cost = 0.0
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        self.model.eval()
        total_loss = defaultdict(list)
        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic

            clip_graph_embeds, clip_text_embeds = self.model.generate_embed_vector(
                batch_data
            )

            loss_dict = self.model(
                text_embed=clip_text_embeds, moleculargraph_embed=clip_graph_embeds
            )

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            batch_cost = time.perf_counter() - batch_tic
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Epoch [{epoch_id}/{self.epochs}] "
                msg += f"| Step: [{iter_id+1}/{data_length}]"
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        total_loss_avg = {k: sum(v) / len(v) for k, v in total_loss.items()}

        return total_loss_avg

    def update(self):
        """梯度更新、调度器步进以及"""

        self.optimizer.step()
        self.optimizer.clear_grad()

        # 根据是否存在预热调度器，进行调度器更新
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.step()
        else:
            self.lr_scheduler.step()

        self.global_step += 1


class TrainerMMDecoder(TrainerGraph):
    def __init__(
        self,
        config,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        test_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        metric_class: Optional[Callable] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
    ):
        super().__init__(
            config,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer,
            metric_class,
            lr_scheduler,
        )
