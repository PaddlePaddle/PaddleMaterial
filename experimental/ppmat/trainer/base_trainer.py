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
from collections import OrderedDict
from collections import defaultdict
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
from paddle.optimizer.lr import ReduceOnPlateau

from ppmat.trainer.callbacks.trainer_callback import CallbackHandler
from ppmat.trainer.callbacks.trainer_callback import DefaultFlowCallback
from ppmat.trainer.callbacks.trainer_callback import ExportableState
from ppmat.trainer.callbacks.trainer_callback import TrainerControl
from ppmat.trainer.callbacks.trainer_callback import TrainerState
from ppmat.trainer.utils import compute_batch_size
from ppmat.trainer.utils import log_paddle_version
from ppmat.utils import AverageMeter
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils import save_load

DEFAULT_CALLBACKS = [DefaultFlowCallback]


class BaseTrainer:
    """Base Trainer for training model.

    Args:
        config (Dict): Training configuration.
        model (nn.Layer): Module to be trained.
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
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        eval_dataloader: Optional[paddle.io.DataLoader] = None,
        test_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
        callbacks: Optional[list] = None,
        compute_metric_func_dict: Optional[Dict] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.compute_metric_func_dict = compute_metric_func_dict

        self.config = config

        if optimizer is None:
            self.whether_use_amp = False
            logger.info("Optimizer is None, AMP is disabled.")

        # get config from config file
        self.max_epochs = config["max_epochs"]
        self.output_dir = config["output_dir"]
        self.save_freq = config["save_freq"]
        self.log_freq = config["log_freq"]
        self.start_eval_epoch = config["start_eval_epoch"]
        self.eval_freq = config["eval_freq"]
        self.seed = config["seed"]
        self.pretrained_model_path = config.get("pretrained_model_path", None)
        self.resume_from_checkpoint = config.get("resume_from_checkpoint", None)
        self.whether_compute_metric_during_train = config[
            "whether_compute_metric_during_train"
        ]
        self.whether_use_amp = config.get("whether_use_amp", False)
        self.amp_level = config.get("amp_level", "O1")
        self.metric_strategy_during_eval = config.get(
            "metric_strategy_during_eval", "step"
        )
        if self.whether_use_amp:
            logger.info(f"Using AMP with level {self.amp_level}.")

        self.iters_per_epoch = len(self.train_dataloader)

        # set automatic mixed precision(AMP) configuration
        self.scaler = paddle.amp.GradScaler(True) if self.whether_use_amp else None

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

        # decorate model(s) and optimizer(s) for AMP
        if self.whether_use_amp:
            self.model, self.optimizer = amp.decorate(
                self.model,
                self.optimizer,
                self.amp_level,
                save_dtype="float32",
            )

        # wrap model and optimizer to parallel object
        if self.world_size > 1:
            if isinstance(self.model, paddle.DataParallel):
                raise ValueError(
                    "Given model is already wrapped by paddle.DataParallel."
                    "Please do not wrap your model with DataParallel "
                    "before 'RegressionTrainer.__init__' and keep it's type "
                    "as 'nn.Layer'."
                )

            self.model = fleet.distributed_model(self.model)
            if self.optimizer is not None:
                self.optimizer = fleet.distributed_optimizer(self.optimizer)

        self.global_step = 0
        log_paddle_version()

        default_callbacks = DEFAULT_CALLBACKS
        self.callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        # self.add_callback(PrinterCallback)
        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.rank == 0,
            is_world_process_zero=self.rank == 0,
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.control = self.callback_handler.on_init_end(
            self.config, self.state, self.control
        )

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a
               [`~transformers.TrainerCallback`]. In the first case, will instantiate
               a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`]
        and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a
               [`~transformers.TrainerCallback`]. In the first case, will pop the first
               member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a
               [`~transformers.TrainerCallback`]. In the first case, will remove the
               first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return sum(
            p.numel() for p in self.model.parameters() if p.stop_gradient is False
        )

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
    def eval_epoch(self, dataloader: paddle.io.DataLoader):
        """Evaluate model on a dataset.

        Args:
            dataloader (paddle.io.DataLoader): Dataloader of current epoch
            epoch_id (int): Epoch id.
        """
        # set model to eval mode
        self.model.eval()
        # initialize eval loss, metric, cost info
        eval_loss_info = {}
        eval_metric_info = {}
        eval_time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # update training state
        # self.callback_handler.eval_dataloader = dataloader
        self.state.max_steps_in_eval_epoch = len(dataloader)
        self.state.mode = "eval"
        self.state.step_in_eval_epoch = 0

        num_eval_samples = len(dataloader.dataset)
        # call on_eval_epoch_begin
        self.callback_handler.on_eval_epoch_begin(self.config, self.state, self.control)

        all_pred_dict = defaultdict(list)
        all_label_dict = defaultdict(list)

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()

        # start to evaluate
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            eval_time_info["reader_cost"].update(reader_cost)

            # auto compute batch size
            batch_size = self.guess_batch_size(batch_data, dataloader)
            # update training state
            self.state.step_in_eval_epoch += 1

            self.callback_handler.on_eval_step_begin(
                self.config, self.state, self.control
            )

            result = self.model(batch_data)

            loss_dict = result.get("loss_dict", {})
            # update loss and metric for log
            for key in loss_dict:
                if key not in eval_loss_info:
                    eval_loss_info[key] = AverageMeter(key)
                eval_loss_info[key].update(loss_dict[key], batch_size)

            pred_dict = result.get("pred_dict", {})

            if self.compute_metric_func_dict is not None:
                if self.metric_strategy_during_eval == "step":
                    for (
                        key,
                        compute_metric_func,
                    ) in self.compute_metric_func_dict.items():
                        pred = pred_dict[key]
                        label = batch_data[key]
                        metric = compute_metric_func(pred, label)
                        if key not in eval_metric_info:
                            eval_metric_info[key] = AverageMeter(key)
                        eval_metric_info[key].update(metric, batch_size)
                else:
                    for key, pred in pred_dict.items():
                        pred = pred.detach() if hasattr(pred, "detach") else pred
                        if self.world_size > 1:
                            pred = misc.all_gather(pred)
                        all_pred_dict[key].append(pred)
                    label_keys = self.compute_metric_func_dict.keys()
                    for key in label_keys:
                        label = batch_data[key]
                        label = label.detach() if hasattr(label, "detach") else label
                        if self.world_size > 1:
                            label = misc.all_gather(label)
                        all_label_dict[key].append(label)

            batch_cost = time.perf_counter() - batch_tic
            eval_time_info["batch_cost"].update(batch_cost)
            self.callback_handler.on_eval_step_end(
                self.config, self.state, self.control
            )
            self._maybe_log(eval_time_info, eval_loss_info, eval_metric_info)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        if (
            self.metric_strategy_during_eval == "epoch"
            and self.compute_metric_func_dict is not None
        ):
            for key, compute_metric_func in self.compute_metric_func_dict.items():
                pred = paddle.concat(all_pred_dict[key])[:num_eval_samples]
                label = paddle.concat(all_label_dict[key])[:num_eval_samples]
                metric = compute_metric_func(pred, label)
                if key not in eval_metric_info:
                    eval_metric_info[key] = AverageMeter(key)
                eval_metric_info[key].update(metric, num_eval_samples)

        self.callback_handler.on_eval_epoch_end(self.config, self.state, self.control)
        self._maybe_log(eval_time_info, eval_loss_info, eval_metric_info)
        return eval_loss_info, eval_metric_info

    def guess_batch_size(self, input_data, dataloader):
        try:
            batch_size = compute_batch_size(input_data, dataloader)
            return batch_size
        except Exception as e:
            logger.debug(
                f"Failed to calculate batch size due to error: {e}. Falling back "
                "to default batch size by dataloader."
            )
        try:
            batch_size = dataloader.batch_sampler.batch_size
            return batch_size
        except Exception as e:
            logger.debug(
                f"Failed to calculate batch size due to error: {e}. Falling back "
                "to default batch size of 1."
            )
            batch_size = 1

        return batch_size

    def train_epoch(self, dataloader: paddle.io.DataLoader):
        """Train program for one epoch.
        Args:
            dataloader (paddle.io.DataLoader): Dataloader of current epoch
            epoch_id (int): Epoch id.
        """
        # set model to train mode
        self.model.train()
        # initialize train loss, metric, cost info
        train_loss_info = {}
        train_metric_info = {}
        train_time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # update training state
        # self.callback_handler.train_dataloader = dataloader
        self.state.max_steps_in_train_epoch = len(dataloader)
        self.state.epoch += 1
        self.state.mode = "train"
        self.state.step_in_train_epoch = 0
        # call on_train_epoch_begin
        self.callback_handler.on_train_epoch_begin(
            self.config, self.state, self.control
        )

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        # start training loop
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            train_time_info["reader_cost"].update(reader_cost)
            # auto compute batch size
            batch_size = self.guess_batch_size(batch_data, dataloader)

            # update training state
            self.state.step_in_train_epoch += 1
            self.state.global_step += 1

            self.callback_handler.on_train_step_begin(
                self.config, self.state, self.control
            )
            # run forward, maybe use amp
            with self.autocast_context_manager(self.whether_use_amp, self.amp_level):
                result = self.model(batch_data)
                loss_dict = result["loss_dict"]
                loss = loss_dict["loss"]

            # run backward, maybe use amp
            if self.whether_use_amp:
                loss_scaled = self.scaler.scale(loss)
                loss_scaled.backward()
            else:
                loss.backward()

            if self.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP + no_sync
                hpu.fused_allreduce_gradients(list(self.model.parameters()), None)
            self.callback_handler.on_pre_optimizer_step(
                self.config, self.state, self.control
            )
            # update parameters
            if self.whether_use_amp:
                self.scaler.minimize(self.optimizer, loss_scaled)
            else:
                self.optimizer.step()
            self.callback_handler.on_optimizer_step(
                self.config, self.state, self.control
            )
            self.optimizer.clear_grad()

            # update loss and metric for log
            for key in loss_dict:
                if key not in train_loss_info:
                    train_loss_info[key] = AverageMeter(key)
                train_loss_info[key].update(loss_dict[key], batch_size)

            if self.whether_compute_metric_during_train:
                pred_dict = result.get("pred_dict", {})
                for key, compute_metric_func in self.compute_metric_func_dict.items():
                    pred = pred_dict[key]
                    label = batch_data[key]
                    metric = compute_metric_func(pred, label)
                    if key not in train_metric_info:
                        train_metric_info[key] = AverageMeter(key)
                    train_metric_info[key].update(metric, batch_size)

            batch_cost = time.perf_counter() - batch_tic
            train_time_info["batch_cost"].update(batch_cost)

            self.callback_handler.on_train_step_end(
                self.config, self.state, self.control
            )
            self._maybe_log(train_time_info, train_loss_info, train_metric_info)

            # update learning rate by epoch
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()
        self.callback_handler.on_train_epoch_end(self.config, self.state, self.control)
        self._maybe_log(train_time_info, train_loss_info, train_metric_info)
        return train_loss_info, train_metric_info

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Training."""

        # load model checkpoint, usually used for resume training
        resume_from_checkpoint = (
            resume_from_checkpoint
            if resume_from_checkpoint is not None
            else self.resume_from_checkpoint
        )
        if resume_from_checkpoint is not None:
            if self.pretrained_model_path is not None:
                logger.warning(
                    "Detected 'pretrained_model_path' is given, weights in which might"
                    " be overridden by weights loaded from given 'checkpoint_path'."
                )
            loaded_state = save_load.load_checkpoint(
                self.resume_from_checkpoint,
                self.model,
                self.optimizer,
                self.scaler,
            )
            self.state = TrainerState.from_dict(loaded_state)

        self.control = self.callback_handler.on_train_begin(
            self.config, self.state, self.control
        )
        start_epoch = self.state.epoch + 1
        for epoch_id in range(start_epoch, self.max_epochs + 1):
            train_loss_info, train_metric_info = self.train_epoch(self.train_dataloader)
            eval_loss_info, eval_metric_info = self._maybe_eval()

            self._determine_best_metric(
                train_loss_info, train_metric_info, eval_loss_info, eval_metric_info
            )

            if self.state.epoch % self.config["save_freq"] == 0:
                self.state.should_save_by_freq = True

            self._maybe_save()

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                if isinstance(self.lr_scheduler, ReduceOnPlateau):
                    if self.lr_scheduler.indicator == "train_loss":
                        indicator_value = train_loss_info[
                            self.lr_scheduler.indicator_name
                        ].avg
                    elif self.lr_scheduler.indicator == "train_metric":
                        indicator_value = train_metric_info[
                            self.lr_scheduler.indicator_name
                        ].avg
                    elif self.lr_scheduler.indicator == "eval_loss":
                        indicator_value = eval_loss_info[
                            self.lr_scheduler.indicator_name
                        ].avg
                    elif self.lr_scheduler.indicator == "eval_metric":
                        indicator_value = eval_metric_info[
                            self.lr_scheduler.indicator_name
                        ].avg
                    else:
                        raise ValueError(
                            "Unsupported lr scheduler indicator: "
                            f"{self.lr_scheduler.indicator}"
                        )
                    self.lr_scheduler.step(metrics=indicator_value)
                else:
                    self.lr_scheduler.step()

    def _determine_best_metric(
        self, train_loss_info, train_metric_info, eval_loss_info, eval_metric_info
    ):
        best_metric_indicator = self.config.get("best_metric_indicator", None)
        name_for_best_metric = self.config.get("name_for_best_metric", None)
        if best_metric_indicator is not None:
            assert (
                name_for_best_metric is not None
            ), "name_for_best_metric must be specified when best_metric_indicator is "
            "specified."

        greater_is_better = self.config["greater_is_better"]
        if best_metric_indicator == "train_loss":
            self.state.cur_metric = train_loss_info[name_for_best_metric].avg
        elif best_metric_indicator == "train_metric":
            self.state.cur_metric = train_metric_info[name_for_best_metric].avg
        elif best_metric_indicator == "eval_loss" and eval_loss_info is not None:
            self.state.cur_metric = eval_loss_info[name_for_best_metric].avg
        elif best_metric_indicator == "eval_metric" and eval_metric_info is not None:
            self.state.cur_metric = eval_metric_info[name_for_best_metric].avg

        if self.state.best_metric is None:
            self.state.best_metric = self.state.cur_metric
            self.state.best_epoch = self.state.epoch
            self.control.should_save_best = True
            return True
        elif greater_is_better:
            if self.state.cur_metric > self.state.best_metric:
                self.state.best_metric = self.state.cur_metric
                self.state.best_epoch = self.state.epoch
                self.control.should_save_best = True
                return True
        else:
            if self.state.cur_metric < self.state.best_metric:
                self.state.best_metric = self.state.cur_metric
                self.state.best_epoch = self.state.epoch
                self.control.should_save_best = True
                return True
        return False

    def eval(self):
        eval_loss_info, eval_metric_info = self.eval_epoch(self.eval_dataloader)
        return eval_loss_info, eval_metric_info

    def test(self):
        test_loss_info, test_metric_info = self.eval_epoch(self.test_dataloader)
        return test_loss_info, test_metric_info

    def _maybe_log(self, time_info, loss_info, metric_info):
        if self.control.should_log_step:
            logs: OrderedDict[str, float] = {}
            for name, average_meter in time_info.items():
                logs[name] = average_meter.val
            for name, average_meter in loss_info.items():
                logs[name] = average_meter.val
            for name, average_meter in metric_info.items():
                logs[name] = average_meter.val
            self.control = self.callback_handler.on_log_step(
                self.config, self.state, self.control, logs=logs
            )
        if self.control.should_log_epoch:
            logs: OrderedDict[str, float] = {}
            for name, average_meter in time_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in loss_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in metric_info.items():
                logs[name] = average_meter.avg
            self.control = self.callback_handler.on_log_epoch(
                self.config, self.state, self.control, logs=logs
            )

    def _maybe_eval(
        self,
    ):
        if self.control.should_evaluate:
            return self.eval_epoch(self.eval_dataloader)
        else:
            return None, None

    def _maybe_save(
        self,
    ):
        if self.control.should_save_by_freq:
            self.control.should_save_by_freq = False
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                self.state.to_dict(),
                self.scaler,
                output_dir=self.output_dir,
                prefix=f"epoch_{self.state.epoch}",
            )
        if self.control.should_save_latest:
            # Always save latest when training begins
            # self.control.should_save_latest = False
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                self.state.to_dict(),
                self.scaler,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(self.state.epoch == 1),
            )
        if self.control.should_save_best:
            self.control.should_save_best = False
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                self.state.to_dict(),
                self.scaler,
                output_dir=self.output_dir,
                prefix="best",
            )
