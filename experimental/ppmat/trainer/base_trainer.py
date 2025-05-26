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
import functools
import os.path as osp
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

from ppmat.trainer.trainer_state import TrainerState
from ppmat.trainer.utils import compute_batch_size
from ppmat.trainer.utils import log_paddle_version
from ppmat.utils import AverageMeter
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils import save_load


class BaseTrainer:
    """Base Trainer for training model. A simple but feature-complete training and
    eval loop for model training.

    Args:
        config (Dict): Training configuration.
        model (nn.Layer): Model to be trained, which should inherit `paddle.nn.Layer`
            class.
        train_dataloader (Optional[paddle.io.DataLoader], optional): Training
            dataloader for training. Defaults to None.
        val_dataloader (Optional[paddle.io.DataLoader], optional): Validation
            dataloader for evaluation. Defaults to None.
        optimizer (Optional[optim.Optimizer], optional): Optimizer for training.
            Defaults to None.
        lr_scheduler (Optional[optim.lr.LRScheduler], optional): Learning Rate
            Scheduler. Defaults to None.
        compute_metric_func_dict (Optional[Dict], optional): Compute metric function
            dictionary. Defaults to None.
    """

    def __init__(
        self,
        config: Dict,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
        compute_metric_func_dict: Optional[Dict] = None,
    ):
        # 1. initialize arguments
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.compute_metric_func_dict = compute_metric_func_dict

        self.config = config

        if optimizer is None:
            self.use_amp = False
            logger.info("Optimizer is None, AMP is disabled.")

        # 2. get config from config file, and set default values if not provided.
        self.max_epochs = config["max_epochs"]
        self.output_dir = config["output_dir"]
        self.save_freq = config["save_freq"]
        self.log_freq = config["log_freq"]
        self.start_eval_epoch = config["start_eval_epoch"]
        self.eval_freq = config["eval_freq"]
        self.seed = config["seed"]
        self.pretrained_model_path = config.get("pretrained_model_path", None)
        self.pretrained_weight_name = config.get("pretrained_weight_name", None)
        self.resume_from_checkpoint = config.get("resume_from_checkpoint", None)
        self.compute_metric_during_train = config["compute_metric_during_train"]
        self.use_amp = config.get("use_amp", False)
        self.amp_level = config.get("amp_level", "O1")
        self.metric_strategy_during_eval = config.get(
            "metric_strategy_during_eval", "step"
        )
        self.eval_with_no_grad = config.get("eval_with_no_grad", True)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        self.use_visualdl = config.get("use_visualdl", False)
        self.use_wandb = config.get("use_wandb", False)
        self.wandb_config = config.get("wandb_config", {})
        self.use_tensorboard = config.get("use_tensorboard", False)

        if self.use_amp:
            logger.info(f"Using AMP with level {self.amp_level}.")

        # 3. set distributed environment, if world_size > 1, initialize distributed
        # environment
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

        # 4. load pretrained model, usually used for transfer learning
        if self.pretrained_model_path is not None:
            save_load.load_pretrain(
                self.model, self.pretrained_model_path, self.pretrained_weight_name
            )

        # 5. set automatic mixed precision(AMP) configuration
        self.scaler = paddle.amp.GradScaler(True) if self.use_amp else None
        if self.use_amp:
            self.model, self.optimizer = amp.decorate(
                self.model,
                self.optimizer,
                self.amp_level,
                save_dtype="float32",
            )

        # 6. wrap model and optimizer to parallel object
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

        # 7. set VisualDL tool
        self.visualdl_writer = None
        if self.use_visualdl:
            try:
                import visualdl as vdl
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install 'visualdl' with `pip install visualdl` first."
                )
            with misc.RankZeroOnly(self.rank) as is_master:
                if is_master:
                    self.visualdl_writer = vdl.LogWriter(
                        osp.join(self.output_dir, "vdl")
                    )
            logger.info(
                "VisualDL is enabled for logging, you can view it by running:\n"
                f"visualdl --logdir {self.visualdl_writer._logdir} --port 8080"
                "\n For more information about how to use VisualDL, please refer to:"
                "https://www.paddlepaddle.org.cn/paddle/visualdl"
            )

        # 8. set WandB tool
        self.wandb_writer = None
        if self.use_wandb:
            try:
                import wandb
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install 'wandb' with `pip install wandb` first."
                )
            with misc.RankZeroOnly(self.rank) as is_master:
                if is_master:
                    self.wandb_writer = wandb.init(**self.wandb_config)

        # 9. set TensorBoardX tool
        self.tensorboard_writer = None
        if self.use_tensorboard:
            try:
                import tensorboardX
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Please install 'tensorboardX' with `pip install tensorboardX` "
                    "first."
                )
            with misc.RankZeroOnly(self.rank) as is_master:
                if is_master:
                    self.tensorboard_writer = tensorboardX.SummaryWriter(
                        osp.join(self.output_dir, "tensorboard")
                    )
            logger.message(
                "TensorboardX is enabled for logging, you can view it by "
                f"running:\ntensorboard --logdir {self.tensorboard_writer.logdir}"
            )

        # 10. log paddle version
        log_paddle_version()

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

    def no_sync_context_manager(
        self,
        enable: bool,
        ddp_model: paddle.DataParallel,
    ) -> contextlib.AbstractContextManager:
        """Smart no_sync context manager for given model.
        NOTE: Only `paddle.DataParallel` object has `no_sync` interface.

        Args:
            enable (bool): Enable no_sync.

        Returns:
            contextlib.AbstractContextManager: Smart no_sync context manager.
        """
        if enable:
            if not isinstance(self.model, paddle.DataParallel):
                raise TypeError(
                    "no_sync interface is only for model with type "
                    f"paddle.DataParallel, but got type {misc.typename(ddp_model)}"
                )
            ctx_manager = ddp_model.no_sync()
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    @functools.lru_cache()
    def no_grad_context_manager(
        self, enable: bool
    ) -> contextlib.AbstractContextManager:
        """Smart no_grad context manager.

        Args:
            enable (bool): Enable no_grad.

        Returns:
            contextlib.AbstractContextManager: Smart no_grad context manager.
        """
        if enable:
            ctx_manager = paddle.no_grad()
        else:
            ctx_manager = (
                contextlib.nullcontext()
                if sys.version_info >= (3, 7)
                else contextlib.suppress()
            )
        return ctx_manager

    def eval_epoch(self, dataloader: paddle.io.DataLoader):
        """Evaluate model on a dataset.

        Args:
            dataloader (paddle.io.DataLoader): Dataloader for evaluation.
        """
        # set model to eval mode
        self.model.eval()
        # initialize eval loss, metric, cost info
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # update training state
        self.state.max_steps_in_eval_epoch = len(dataloader)
        self.state.step_in_eval_epoch = 0

        num_eval_samples = len(dataloader.dataset)

        # initialize all_pred_dict and all_label_dict
        all_pred_dict = defaultdict(list)
        all_label_dict = defaultdict(list)

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()

        # start to evaluate
        for _, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            time_info["reader_cost"].update(reader_cost)

            # auto compute batch size
            batch_size = self.guess_batch_size(batch_data, dataloader)
            # update training state
            self.state.step_in_eval_epoch += 1

            # forward model
            with self.autocast_context_manager(
                self.use_amp, self.amp_level
            ), self.no_grad_context_manager(self.eval_with_no_grad):
                result = self.model(batch_data)
            loss_dict = result.get("loss_dict", {})

            # update loss and metric for log
            for key in loss_dict:
                if key not in loss_info:
                    loss_info[key] = AverageMeter(key)
                loss_info[key].update(float(loss_dict[key]), batch_size)

            # get prediction
            pred_dict = result.get("pred_dict", {})

            # compute metric during evaluation or gathering all predictions and labels
            if self.compute_metric_func_dict is not None:
                if self.metric_strategy_during_eval == "step":
                    # compute metric for each step
                    for (
                        key,
                        compute_metric_func,
                    ) in self.compute_metric_func_dict.items():
                        pred = pred_dict[key]
                        label = batch_data[key]
                        metric = compute_metric_func(pred, label)
                        if key not in metric_info:
                            metric_info[key] = AverageMeter(key)
                        metric_info[key].update(float(metric), batch_size)
                else:
                    # gather all predictions and labels
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
            time_info["batch_cost"].update(batch_cost)

            # log the current step
            if (
                self.state.step_in_eval_epoch % self.config["log_freq"] == 0
                or self.state.step_in_eval_epoch == self.state.max_steps_in_eval_epoch
                or self.state.step_in_eval_epoch == 1
            ):

                logs: OrderedDict[str, float] = {}
                for name, average_meter in time_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in loss_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in metric_info.items():
                    logs[name] = average_meter.val

                msg = f"Eval: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
                msg += (
                    f" | Step: [{self.state.step_in_eval_epoch}/"
                    + f"{self.state.max_steps_in_eval_epoch}]"
                )
                if logs is not None:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)
            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        # compute metric for whole epoch
        if (
            self.metric_strategy_during_eval == "epoch"
            and self.compute_metric_func_dict is not None
        ):
            for key, compute_metric_func in self.compute_metric_func_dict.items():
                pred = paddle.concat(all_pred_dict[key])[:num_eval_samples]
                label = paddle.concat(all_label_dict[key])[:num_eval_samples]
                metric = compute_metric_func(pred, label)
                if key not in metric_info:
                    metric_info[key] = AverageMeter(key)
                metric_info[key].update(float(metric), num_eval_samples)
        return time_info, loss_info, metric_info

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
            dataloader (paddle.io.DataLoader): The dataloader used for training.
        """
        # set model to train mode
        self.model.train()
        # initialize train loss, metric, cost info
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }

        # update training state
        self.state.max_steps_in_train_epoch = (
            len(dataloader) // self.gradient_accumulation_steps
        )
        if len(dataloader) % self.gradient_accumulation_steps != 0:
            self.state.max_steps_in_train_epoch += 1
        self.state.step_in_train_epoch = 0

        # Start timing for reading data and the entire batch
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        # start training loop
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.perf_counter() - reader_tic
            time_info["reader_cost"].update(reader_cost)
            # auto compute batch size
            batch_size = self.guess_batch_size(batch_data, dataloader)

            # run forward, maybe use amp
            with self.no_sync_context_manager(self.world_size > 1, self.model):
                with self.autocast_context_manager(self.use_amp, self.amp_level):
                    result = self.model(batch_data)
                    loss_dict = result["loss_dict"]
                    loss = loss_dict["loss"]

                # run backward, maybe use amp
                if self.use_amp:
                    loss_scaled = self.scaler.scale(loss)
                    loss_scaled.backward()
                else:
                    loss.backward()

            # when the number of iterations is multiple of gradient_accumulation_steps,
            # we need to update parameters
            if (iter_id + 1) % self.gradient_accumulation_steps != 0 and (
                iter_id + 1
            ) != len(dataloader):
                continue

            # update training state
            self.state.step_in_train_epoch += 1
            self.state.global_step += 1

            if self.world_size > 1:
                # fuse + allreduce manually before optimization if use DDP + no_sync
                hpu.fused_allreduce_gradients(list(self.model.parameters()), None)

            # update parameters
            if self.use_amp:
                self.scaler.minimize(self.optimizer, loss_scaled)
            else:
                self.optimizer.step()
            self.optimizer.clear_grad()

            # update loss and metric for log
            for key in loss_dict:
                if key not in loss_info:
                    loss_info[key] = AverageMeter(key)
                loss_info[key].update(float(loss_dict[key]), batch_size)

            if self.compute_metric_during_train:
                pred_dict = result.get("pred_dict", {})
                for key, compute_metric_func in self.compute_metric_func_dict.items():
                    pred = pred_dict[key]
                    label = batch_data[key]
                    metric = compute_metric_func(pred, label)
                    if key not in metric_info:
                        metric_info[key] = AverageMeter(key)
                    metric_info[key].update(float(metric), batch_size)

            batch_cost = time.perf_counter() - batch_tic
            time_info["batch_cost"].update(batch_cost)

            # log training info
            if (
                self.state.step_in_train_epoch % self.config["log_freq"] == 0
                or self.state.step_in_train_epoch == self.state.max_steps_in_train_epoch
                or self.state.step_in_train_epoch == 1
            ):

                logs: OrderedDict[str, float] = {}
                if self.optimizer is not None:
                    logs["lr"] = self.optimizer.get_lr()
                for name, average_meter in time_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in loss_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in metric_info.items():
                    logs[name] = average_meter.val

                msg = f"Train: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
                msg += (
                    f" | Step: [{self.state.step_in_train_epoch}/"
                    + f"{self.state.max_steps_in_train_epoch}]"
                )
                if logs is not None:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)
                # log training info to visualdl, wandb, tensorboard
                logger.scalar(
                    tag="train(step)",
                    metric_dict=logs,
                    step=self.state.global_step,
                    visualdl_writer=self.visualdl_writer,
                    wandb_writer=self.wandb_writer,
                    tensorboard_writer=self.tensorboard_writer,
                )

            # update learning rate by epoch
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()
        return time_info, loss_info, metric_info

    def train(
        self,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Start a new training process."""

        if train_dataloader is None:
            assert (
                self.train_dataloader is not None
            ), "train_dataloader is None, please set it or pass to the constructor."
            train_dataloader = self.train_dataloader
        else:
            self.train_dataloader = train_dataloader
        if val_dataloader is None:
            assert (
                self.val_dataloader is not None
            ), "val_dataloader is None, please set it or pass to the constructor."
            val_dataloader = self.val_dataloader
        else:
            self.val_dataloader = val_dataloader
        
        if hasattr(self.model, 'before_train'):
            self.model.before_train(self)

        self.state = TrainerState()
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

        logger.info("Training start...")
        trainable_params = self.get_num_trainable_parameters()
        logger.info(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")

        # train loop
        for _ in range(self.state.epoch, self.max_epochs):
            self.state.epoch += 1
            # train one epoch
            train_time_info, train_loss_info, train_metric_info = self.train_epoch(
                train_dataloader
            )
            # log training info
            logs: OrderedDict[str, float] = {}
            for name, average_meter in train_time_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in train_loss_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in train_metric_info.items():
                logs[name] = average_meter.avg
            msg = f"Train: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
            if logs is not None:
                for key, val in logs.items():
                    msg += f" | {key}: {val:.6f}"
            logger.info(msg)
            # Temporary disable wandb_writer, since it is not support step less the
            # current step(self.state.global_step)
            logger.scalar(
                tag="train(epoch)",
                metric_dict=logs,
                step=self.state.epoch,
                visualdl_writer=self.visualdl_writer,
                # wandb_writer=self.wandb_writer,
                tensorboard_writer=self.tensorboard_writer,
            )

            # save checkpoint when epoch is divisible by save_freq
            if (
                self.state.epoch % self.config["save_freq"] == 0
                or self.state.epoch == self.config["max_epochs"]
                or self.state.epoch == 1
            ):
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.state.to_dict(),
                    self.scaler,
                    output_dir=self.output_dir,
                    prefix=f"epoch_{self.state.epoch}",
                )

            # Always save latest when training begins
            save_load.save_checkpoint(
                self.model,
                self.optimizer,
                self.state.to_dict(),
                self.scaler,
                output_dir=self.output_dir,
                prefix="latest",
                print_log=(self.state.epoch == 1),
            )

            # evaluate model when epoch is divisible by eval_freq
            if (
                self.state.epoch % self.config["eval_freq"] == 0
                or self.state.epoch == self.config["max_epochs"]
                or self.state.epoch == 1
            ):

                eval_time_info, eval_loss_info, eval_metric_info = self.eval_epoch(
                    val_dataloader
                )

                logs: OrderedDict[str, float] = {}
                for name, average_meter in eval_time_info.items():
                    logs[name] = average_meter.avg
                for name, average_meter in eval_loss_info.items():
                    logs[name] = average_meter.avg
                for name, average_meter in eval_metric_info.items():
                    logs[name] = average_meter.avg

                msg = f"Eval: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
                if logs is not None:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)
                # Temporary disable wandb_writer, since it is not support step less the
                # current step(self.state.global_step)
                logger.scalar(
                    tag="eval(epoch)",
                    metric_dict=logs,
                    step=self.state.epoch,
                    visualdl_writer=self.visualdl_writer,
                    # wandb_writer=self.wandb_writer,
                    tensorboard_writer=self.tensorboard_writer,
                )

            # save best model when best_metric is better than previous best_metric
            save_best_flag = self._determine_best_metric(
                train_loss_info, train_metric_info, eval_loss_info, eval_metric_info
            )
            if save_best_flag:
                save_load.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.state.to_dict(),
                    self.scaler,
                    output_dir=self.output_dir,
                    prefix="best",
                )
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
        if best_metric_indicator is None:
            return False
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
        else:
            raise ValueError(
                f"Unsupported best_metric_indicator: {best_metric_indicator}"
            )

        if self.state.best_metric is None:
            self.state.best_metric = self.state.cur_metric
            self.state.best_epoch = self.state.epoch
            return True
        elif greater_is_better:
            if self.state.cur_metric > self.state.best_metric:
                self.state.best_metric = self.state.cur_metric
                self.state.best_epoch = self.state.epoch
                return True
        else:
            if self.state.cur_metric < self.state.best_metric:
                self.state.best_metric = self.state.cur_metric
                self.state.best_epoch = self.state.epoch
                return True
        return False

    def eval(self, dataloader: paddle.io.DataLoader):
        assert dataloader is not None, "dataloader is None, please set it first"
        self.state = TrainerState()
        time_info, loss_info, metric_info = self.eval_epoch(dataloader)
        logs: OrderedDict[str, float] = {}
        for name, average_meter in time_info.items():
            logs[name] = average_meter.avg
        for name, average_meter in loss_info.items():
            logs[name] = average_meter.avg
        for name, average_meter in metric_info.items():
            logs[name] = average_meter.avg

        msg = "Eval:"
        if logs is not None:
            for key, val in logs.items():
                msg += f" | {key}: {val:.6f}"
        logger.info(msg)

        return time_info, loss_info, metric_info
