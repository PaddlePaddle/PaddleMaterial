from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable
from typing import Optional

import paddle
import paddle.distributed as dist
from packaging import version
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet

from mattergen.utils import logger
from mattergen.utils import save_load


def scale_shared_grads(model):
    """Divide the gradients of the layers that are shared across multiple
    blocks
    by the number the weights are shared for
    """
    with paddle.no_grad():

        def scale_grad(param, scale_factor):
            if param.grad is None:
                return
            g_data = param.grad
            new_grads = g_data / scale_factor
            param.grad = new_grads  # .copy_(new_grads)

        if isinstance(model, paddle.distributed.parallel.DataParallel):
            model = model._layers
        for layer, num_blocks in model.shared_parameters:
            scale_grad(layer, num_blocks)


class TrainerDiffusion:
    """Class used to handle training a diffusion model"""

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
        self.epochs = config.trainer.max_epochs
        self.output_dir = config.trainer.output_dir
        self.save_freq = config.trainer.save_freq
        self.log_freq = config.trainer.log_freq
        self.start_eval_epoch = config.trainer.start_eval_epoch
        self.eval_freq = config.trainer.eval_freq
        self.seed = config.trainer.seed
        self.pretrained_model_path = config.trainer.pretrained_model_path
        self.checkpoint_path = config.trainer.checkpoint_path
        self.scale_grad = config.trainer.scale_grad
        self.is_save_traj = config.trainer.is_save_traj
        self.step_lr = config.trainer.step_lr

        self.iters_per_epoch = len(self.train_dataloader)

        # if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
        #     if (
        #         self.config["Optimizer"]["lr"].get("indicator", "train_loss")
        #         == "eval_loss"
        #     ):
        #         assert self.eval_freq == 1, (
        #             "ReduceOnPlateau only support eval_freq==1 when indicator="
        #             "'eval_loss'"
        #         )
        #         assert self.lr_scheduler.by_epoch is True, (
        #             "ReduceOnPlateau only support by_epoch=True, when indicator="
        #             "'eval_loss"
        #         )

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
            if version.Version(paddle.__version__) < version.Version("3.0.0"):
                logger.warning(
                    f"Detected paddlepaddle version is '{paddle_version}', "
                    "currently it is recommended to use release 3.0 or develop version."
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

            loss, loss_dict = self.model(batch_data)
            loss_dict["loss"] = loss

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
                    if k == "loss":
                        msg += f" | {k}: {v:.5f}"
                    else:
                        msg += f" | {k}(loss): {v:.5f}"
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
            loss, loss_dict = self.model(batch_data)
            loss_dict["loss"] = loss

            loss.backward()
            # if self.scale_grad:
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
                        self.config.lightning_module.optimizer_partial.cfg.lr.get(
                            "indicator", "train_loss"
                        )
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
                    if k == "loss":
                        msg += f" | {k}: {v:.5f}"
                    else:
                        msg += f" | {k}(loss): {v:.5f}"
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
                    if k == "loss":
                        msg += f" | {k}: {v:.5f}"
                    else:
                        msg += f" | {k}(loss): {v:.5f}"
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
                        self.config.lightning_module.optimizer_partial.cfg.lr.get(
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
        loss_dict = self.eval_epoch(self.test_dataloader, epoch_id=1)
        msg = "Test: "
        for k, v in loss_dict.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
        logger.info(msg)
        return loss_dict
