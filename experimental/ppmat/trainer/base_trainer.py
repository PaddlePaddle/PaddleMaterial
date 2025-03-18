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

from __future__ import annotations

import contextlib
import sys
import time
from typing import Dict
from typing import Literal
from typing import Optional

import paddle
import paddle.distributed as dist
from paddle import amp
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu

from ppmat.trainer.utils import compute_batch_size
from ppmat.trainer.utils import log_paddle_version
from ppmat.trainer.utils import scale_shared_grads
from ppmat.utils import AverageMeter
from ppmat.utils import format_time_manual
from ppmat.utils import logger
from ppmat.utils import save_load


class BaseTrainer:
    """Base Trainer for training model.

    Args:
        config (Dict): Training configuration.
        module (nn.Layer): Module to be trained.
        train_dataloader (Optional[paddle.io.DataLoader], optional): Training
            dataloader. Defaults to None.
        val_dataloader (Optional[paddle.io.DataLoader], optional): Validation
            dataloader. Defaults to None.
        test_dataloader (Optional[paddle.io.DataLoader], optional): Testing dataloader.
            Defaults to None.
        optimizer (Optional[optim.Optimizer], optional): Optimizer. Defaults to None.
        lr_scheduler (Optional[optim.lr.LRScheduler], optional): Learning Rate
            Scheduler. Defaults to None.

    """

    def __init__(
        self,
        config: Dict,
        module: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        test_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
    ):
        self.config = config
        self.module = module
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = lr_scheduler

        if optimizer is None:
            self.use_amp = False
            logger.info("Optimizer is None, AMP is disabled.")

        # get config from config file
        self.epochs = config["Trainer"]["epochs"]
        self.output_dir = config["Trainer"]["output_dir"]
        self.save_freq = config["Trainer"]["save_freq"]
        self.log_freq = config["Trainer"]["log_freq"]
        self.start_eval_epoch = config["Trainer"]["start_eval_epoch"]
        self.eval_freq = config["Trainer"]["eval_freq"]
        self.seed = config["Trainer"]["seed"]
        self.pretrained_model_path = config["Trainer"].get(
            "pretrained_model_path", None
        )
        self.checkpoint_path = config["Trainer"].get("checkpoint_path", None)
        self.cal_metric_during_train = config["Trainer"]["cal_metric_during_train"]
        self.scale_grad = config["Trainer"].get("scale_grad", False)
        self.use_amp = config["Trainer"].get("use_amp", False)
        self.amp_level = config["Trainer"].get("amp_level", "O1")
        if self.use_amp:
            logger.info(f"Using AMP with level {self.amp_level}.")

        self.iters_per_epoch = len(self.train_dataloader)

        # set automatic mixed precision(AMP) configuration
        self.scaler = paddle.amp.GradScaler(True) if self.use_amp else None

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
            save_load.load_pretrain(self.module, self.pretrained_model_path)

        # initialize an dict for tracking best model during training
        self.best_eval_info = {
            "epoch": 0,
            "eval_loss": float("inf"),
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
                self.module,
                self.optimizer,
                self.scaler,
            )
            if isinstance(loaded_metric, dict):
                self.best_eval_info.update(loaded_metric)

        # decorate model(s) and optimizer(s) for AMP
        if self.use_amp:
            self.module, self.optimizer = amp.decorate(
                self.module,
                self.optimizer,
                self.amp_level,
                save_dtype="float32",
            )

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.module, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'RegressionTrainer.__init__' and keep it's type "
                    "as 'nn.Layer'."
                )

            self.module = fleet.distributed_model(self.module)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0
        log_paddle_version()

    def autocast_context_manager(
        self, enable: bool, level: Literal["O0", "O1", "O2", "OD"] = "O1"
    ) -> contextlib.AbstractContextManager:
        """Smart autocast context manager for Auto Mix Precision.

        Args:
            enable (bool): Enable autocast.
            level (Literal["O0", "O1", "O2", "OD"]): Autocast level.

        Returns:
            contextlib.AbstractContextManager: Smart autocast context manager.
        """
        if enable:
            ctx_manager = amp.auto_cast(level=level)
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    @paddle.no_grad()
    def eval_epoch(self, dataloader: paddle.io.DataLoader, epoch_id: int, mode: str):
        """Evaluate model on a dataset.

        Args:
            dataloader (paddle.io.DataLoader): Dataloader of current epoch
            epoch_id (int): Epoch id.
        """

        self.module.eval()
        # initialize eval loss, metric, cost info
        eval_loss_info = {}
        eval_metric_info = {}
        eval_time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # get the length of dataloader
        data_length = len(dataloader)
        comp_batch_size_error = False

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()

        # start to evaluate
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            eval_time_info["reader_cost"].update(reader_cost)

            # auto compute batch size
            try:
                batch_size = compute_batch_size(batch_data)
            except Exception as e:
                logger.debug(
                    f"Failed to calculate batch size due to error: {e}. Falling back "
                    "to default batch size of 1."
                )
                comp_batch_size_error = True
                batch_size = 1

            result = self.module(batch_data, mode=mode)

            loss_dict = result.get("loss_dict", {})
            # update loss and metric for log
            for key in loss_dict:
                if key not in eval_loss_info:
                    eval_loss_info[key] = AverageMeter(key + "(loss)")
                eval_loss_info[key].update(loss_dict[key], batch_size)
            metric_dict = result.get("metric_dict", {})
            for key in metric_dict:
                if key not in eval_metric_info:
                    eval_metric_info[key] = AverageMeter(key + "(metric)")
                eval_metric_info[key].update(metric_dict[key], batch_size)

            batch_cost = time.perf_counter() - batch_tic
            eval_time_info["batch_cost"].update(batch_cost)
            if dist.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Epoch [{epoch_id}/{self.epochs}] "
                msg += f"| Step: [{iter_id+1}/{data_length}]"
                for _, average_meter in eval_time_info.items():
                    msg += f" | {average_meter.value}"
                for _, average_meter in eval_loss_info.items():
                    msg += f" | {average_meter.value}"
                for _, average_meter in eval_metric_info.items():
                    msg += f" | {average_meter.value}"
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        if comp_batch_size_error is True:
            logger.warning(
                "The automatic batch size calculation failed, resulting in inaccurate "
                "epoch-level average loss and metric values; however, model training "
                "will proceed unaffected."
            )
        return eval_loss_info, eval_metric_info

    def train_epoch(self, dataloader: paddle.io.DataLoader, epoch_id: int, mode: str):
        """Train program for one epoch.
        Args:
            dataloader (paddle.io.DataLoader): Dataloader of current epoch
            epoch_id (int): Epoch id.
        """
        # set model to train mode
        self.module.train()
        # initialize train loss, metric, cost info
        train_loss_info = {}
        train_metric_info = {}
        train_time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # get the length of dataloader
        data_length = len(dataloader)
        comp_batch_size_error = False

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        # start training loop
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            train_time_info["reader_cost"].update(reader_cost)

            # auto compute batch size
            try:
                batch_size = compute_batch_size(batch_data)
            except Exception as e:
                logger.debug(
                    f"Failed to calculate batch size due to error: {e}. Falling back "
                    "to default batch size of 1."
                )
                comp_batch_size_error = True
                batch_size = 1

            # run forward, maybe use amp
            with self.autocast_context_manager(self.use_amp, self.amp_level):
                result = self.module(batch_data, mode=mode)

                # run backward, maybe use amp
                loss_dict = result["loss_dict"]
                loss = loss_dict["loss"]

            if self.use_amp:
                loss_scaled = self.scaler.scale(loss)
                loss_scaled.backward()
            else:
                loss.backward()

            # scale gradients if the network has the attribute "shared_parameters"
            if self.scale_grad:
                scale_shared_grads(self.module)
            if self.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP + no_sync
                # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                hpu.fused_allreduce_gradients(list(self.module.parameters()), None)
            # update parameters
            if self.use_amp:
                self.scaler.minimize(self.optimizer, loss_scaled)
            else:
                self.optimizer.step()
            self.optimizer.clear_grad()

            # update learning rate by step
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # update loss and metric for log
            for key in loss_dict:
                if key not in train_loss_info:
                    train_loss_info[key] = AverageMeter(key + "(loss)")
                train_loss_info[key].update(loss_dict[key], batch_size)
            metric_dict = result.get("metric_dict", {})
            for key in metric_dict:
                if key not in train_metric_info:
                    train_metric_info[key] = AverageMeter(key + "(metric)")
                train_metric_info[key].update(metric_dict[key], batch_size)

            self.global_step += 1
            batch_cost = time.perf_counter() - batch_tic
            train_time_info["batch_cost"].update(batch_cost)
            # update and log training information
            if dist.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | lr: {self.optimizer._learning_rate():.6f}".rstrip("0")

                for _, average_meter in train_time_info.items():
                    msg += f" | {average_meter.value}"
                for _, average_meter in train_loss_info.items():
                    msg += f" | {average_meter.value}"
                for _, average_meter in train_metric_info.items():
                    msg += f" | {average_meter.value}"
                # compute eta time
                eta = train_time_info["batch_cost"].avg * (
                    self.max_steps - self.global_step
                )
                # convert eta to hours:minutes:seconds
                msg += f" | eta: {format_time_manual(eta)}"
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()
        if comp_batch_size_error is True:
            logger.warning(
                "The automatic batch size calculation failed, resulting in inaccurate "
                "epoch-level average loss and metric values; however, model training "
                "will proceed unaffected."
            )
        return train_loss_info, train_metric_info

    def train(self) -> None:
        """Training."""
        self.global_step = self.best_eval_info["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = self.best_eval_info["epoch"] + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            train_loss_info, train_metric_info = self.train_epoch(
                self.train_dataloader, epoch_id, mode="train"
            )

            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for _, average_meter in train_loss_info.items():
                msg += f" | {average_meter.avg_info}"
            for _, average_meter in train_metric_info.items():
                msg += f" | {average_meter.avg_info}"
            logger.info(msg)

            save_info_dict = {
                "epoch": epoch_id,
                "train_loss": train_loss_info["loss"].avg
                if "loss" in train_loss_info
                else float("inf"),
            }
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_info, eval_metric_info = self.eval_epoch(
                    self.val_dataloader, epoch_id, mode="eval"
                )
                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                for _, average_meter in eval_loss_info.items():
                    msg += f" | {average_meter.avg_info}"
                for _, average_meter in eval_metric_info.items():
                    msg += f" | {average_meter.avg_info}"
                logger.info(msg)

                save_info_dict["eval_loss"] = (
                    eval_loss_info["loss"].avg
                    if "loss" in eval_loss_info
                    else float("inf")
                )

                # update best metric
                if save_info_dict["eval_loss"] <= self.best_eval_info["eval_loss"]:
                    self.best_eval_info = save_info_dict
                    save_load.save_checkpoint(
                        self.module,
                        self.optimizer,
                        self.best_eval_info,
                        self.scaler,
                        output_dir=self.output_dir,
                        prefix="best",
                    )

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.save_freq > 0 and epoch_id % self.save_freq == 0:
                save_load.save_checkpoint(
                    self.module,
                    self.optimizer,
                    save_info_dict,
                    self.scaler,
                    output_dir=self.output_dir,
                    prefix=f"epoch_{epoch_id}",
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.module,
                self.optimizer,
                save_info_dict,
                self.scaler,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(epoch_id == start_epoch),
            )

    def eval(self):
        eval_loss_info, eval_metric_info = self.eval_epoch(
            self.val_dataloader, epoch_id=1, mode="eval"
        )
        msg = "Eval: "
        for _, average_meter in eval_loss_info.items():
            msg += f" | {average_meter.avg_info}"
        for _, average_meter in eval_metric_info.items():
            msg += f" | {average_meter.avg_info}"
        logger.info(msg)

    def test(self):
        test_loss_info, test_metric_info = self.eval_epoch(
            self.test_dataloader, epoch_id=1, mode="test"
        )
        msg = "Test: "
        for _, average_meter in test_loss_info.items():
            msg += f" | {average_meter.avg_info}"
        for _, average_meter in test_metric_info.items():
            msg += f" | {average_meter.avg_info}"
        logger.info(msg)
