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
import argparse
import os
import time

import paddle
from omegaconf import OmegaConf

from ppmat.datasets import DensityCollator
from ppmat.datasets import DensityVoxelCollator
from ppmat.datasets import build_dataloader
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.models.infgcn.paddle_utils import *  # noqa: F403
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.trainer.trainer_state import TrainerState
from ppmat.utils import logger
from ppmat.utils import misc

# Based on step not epoch


class InfGCNTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_voxel = self.config.get("use_voxel", False)
        self.criterion = paddle.nn.MSELoss()

    def eval_epoch(self, dataloader):
        self.model.eval()
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": misc.AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": misc.AverageMeter(name="batch_cost", postfix="s"),
        }

        self.state.max_steps_in_eval_epoch = len(dataloader)
        self.state.step_in_eval_epoch = 0

        reader_tic = time.time()
        batch_tic = time.time()

        device = paddle.get_device()

        for _, batch_data in enumerate(dataloader):
            reader_cost = time.time() - reader_tic
            time_info["reader_cost"].update(reader_cost)

            batch_size = self.guess_batch_size(batch_data, dataloader)
            self.state.step_in_eval_epoch += 1

            with paddle.no_grad():
                g, density, grid_coord, infos = batch_data
                g = g.to(device)
                density, grid_coord = density.to(device), grid_coord.to(device)
                for i, info in enumerate(infos):
                    if "cell" in info and hasattr(info["cell"], "to"):
                        infos[i]["cell"] = info["cell"].to(device)

                pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

                if self.use_voxel:
                    mask = (density > 0).astype(dtype="float32")
                    pred = pred * mask
                    density = density * mask

                loss = self.criterion(pred, density)
                mae = paddle.abs(pred - density).sum() / density.sum()

            if "loss" not in loss_info:
                loss_info["loss"] = misc.AverageMeter("loss")
            loss_info["loss"].update(float(loss), batch_size)

            if "mae" not in loss_info:
                loss_info["mae"] = misc.AverageMeter("mae")
            loss_info["mae"].update(float(mae), batch_size)

            if self.compute_metric_func_dict is not None:
                for key, compute_metric_func in self.compute_metric_func_dict.items():
                    metric = compute_metric_func(pred, density)
                    if key not in metric_info:
                        metric_info[key] = misc.AverageMeter(key)
                    metric_info[key].update(float(metric), batch_size)

            batch_cost = time.time() - batch_tic
            time_info["batch_cost"].update(batch_cost)

            if (
                self.state.step_in_eval_epoch % self.config["log_freq"] == 0
                or self.state.step_in_eval_epoch == self.state.max_steps_in_eval_epoch
                or self.state.step_in_eval_epoch == 1
            ):
                logs = {}
                for name, average_meter in time_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in loss_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in metric_info.items():
                    logs[name] = average_meter.val

                msg = f"Eval: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
                msg += f" | Step: [{self.state.step_in_eval_epoch} / {self.state.max_steps_in_eval_epoch}]"
                if logs:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)

            batch_tic = time.time()
            reader_tic = time.time()

        return time_info, loss_info, metric_info

    # eval_batch acually, for keep consistance
    def test_epoch(self, dataloader, num_infer=None, num_vis=2, inf_samples=None):
        self.model.eval()
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": misc.AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": misc.AverageMeter(name="batch_cost", postfix="s"),
        }

        self.state.max_steps_in_eval_epoch = num_infer or len(dataloader)
        self.state.step_in_eval_epoch = 0

        reader_tic = time.time()
        batch_tic = time.time()
        device = paddle.get_device()

        for idx, batch_data in enumerate(dataloader):
            if num_infer is not None and idx >= num_infer:
                break

            reader_cost = time.time() - reader_tic
            time_info["reader_cost"].update(reader_cost)

            batch_size = self.guess_batch_size(batch_data, dataloader)
            self.state.step_in_eval_epoch += 1

            with paddle.no_grad():
                g, density, grid_coord, infos = batch_data
                g = g.to(device)
                density, grid_coord = density.to(device), grid_coord.to(device)

                for i, info in enumerate(infos):
                    if "cell" in info and hasattr(info["cell"], "to"):
                        infos[i]["cell"] = info["cell"].to(device)

                grid_batch_size = inf_samples or self.config.get("inf_samples", 4096)
                if grid_batch_size is None:
                    preds = self.model(g.x, g.pos, grid_coord, g.batch, infos)
                else:
                    preds = []
                    for grid in grid_coord.split(grid_batch_size, dim=1):
                        preds.append(self.model(g.x, g.pos, grid, g.batch, infos))
                    preds = paddle.concat(x=preds, axis=1)

                mask = (density > 0).astype(dtype="float32")
                preds = preds * mask
                density = density * mask
                diff = paddle.abs(x=preds - density)
                sum_idx = tuple(range(1, density.dim()))
                loss = diff.pow(y=2).sum(axis=sum_idx) / mask.sum(axis=sum_idx)
                mae = diff.sum(axis=sum_idx) / density.sum(axis=sum_idx)

                if idx == 0 and num_vis > 0:
                    for vis_idx, (p, d, info) in enumerate(zip(preds, density, infos)):
                        if vis_idx >= num_vis:
                            break

            loss = paddle.mean(loss)
            mae = paddle.mean(mae)
            if "loss" not in loss_info:
                loss_info["loss"] = misc.AverageMeter("loss")
            loss_info["loss"].update(float(loss), batch_size)

            if "mae" not in loss_info:
                loss_info["mae"] = misc.AverageMeter("mae")
            loss_info["mae"].update(float(mae), batch_size)

            if self.compute_metric_func_dict is not None:
                for key, compute_metric_func in self.compute_metric_func_dict.items():
                    metric = compute_metric_func(preds, density)
                    if key not in metric_info:
                        metric_info[key] = misc.AverageMeter(key)
                    metric_info[key].update(float(metric), batch_size)

            batch_cost = time.time() - batch_tic
            time_info["batch_cost"].update(batch_cost)

            if (
                self.state.step_in_eval_epoch % self.config["log_freq"] == 0
                or self.state.step_in_eval_epoch == self.state.max_steps_in_eval_epoch
                or self.state.step_in_eval_epoch == 1
            ):
                logs = {}
                for name, average_meter in time_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in loss_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in metric_info.items():
                    logs[name] = average_meter.val

                msg = f"Test: Step [{self.state.step_in_eval_epoch}/{self.state.max_steps_in_eval_epoch}]"
                if logs:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)

            batch_tic = time.time()
            reader_tic = time.time()

        return time_info, loss_info, metric_info

    def train_epoch(self, dataloader):
        self.model.train()
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": misc.AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": misc.AverageMeter(name="batch_cost", postfix="s"),
        }

        self.state.max_steps_in_train_epoch = len(dataloader)
        self.state.step_in_train_epoch = 0

        reader_tic = time.time()
        batch_tic = time.time()
        device = paddle.get_device()
        for iter_id, batch_data in enumerate(dataloader):
            reader_cost = time.time() - reader_tic
            time_info["reader_cost"].update(reader_cost)

            batch_size = self.guess_batch_size(batch_data, dataloader)

            g, density, grid_coord, infos = batch_data

            device = paddle.get_device()
            print(
                "device",
            )
            g = g.to(device)
            density, grid_coord = density.to(device), grid_coord.to(device)
            for i, info in enumerate(infos):
                if "cell" in info and hasattr(info["cell"], "to"):
                    infos[i]["cell"] = info["cell"].to(device)

            pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

            if self.use_voxel:
                mask = (density > 0).astype(dtype="float32")
                pred = pred * mask
                density = density * mask

            loss = self.criterion(pred, density)
            mae = paddle.abs(pred - density).sum() / density.sum()
            loss.backward()

            self.optimizer.step()
            self.optimizer.clear_grad()

            self.state.step_in_train_epoch += 1
            self.state.global_step += 1

            if "loss" not in loss_info:
                loss_info["loss"] = misc.AverageMeter("loss")
            loss_info["loss"].update(float(loss), batch_size)

            if "mae" not in loss_info:
                loss_info["mae"] = misc.AverageMeter("mae")
            loss_info["mae"].update(float(mae), batch_size)

            if (
                self.compute_metric_during_train
                and self.compute_metric_func_dict is not None
            ):
                for key, compute_metric_func in self.compute_metric_func_dict.items():
                    metric = compute_metric_func(pred, density)
                    if key not in metric_info:
                        metric_info[key] = misc.AverageMeter(key)
                    metric_info[key].update(float(metric), batch_size)

            batch_cost = time.time() - batch_tic
            time_info["batch_cost"].update(batch_cost)

            if (
                self.state.step_in_train_epoch % self.config["log_freq"] == 0
                or self.state.step_in_train_epoch == self.state.max_steps_in_train_epoch
                or self.state.step_in_train_epoch == 1
            ):
                logs = {}
                logs["lr"] = self.optimizer.get_lr()
                for name, average_meter in time_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in loss_info.items():
                    logs[name] = average_meter.val
                for name, average_meter in metric_info.items():
                    logs[name] = average_meter.val

                msg = f"Train: Epoch [{self.state.epoch}/{self.config['max_epochs']}]"
                msg += f" | Step: [{self.state.step_in_train_epoch}/{self.state.max_steps_in_train_epoch}]"
                if logs:
                    for key, val in logs.items():
                        msg += f" | {key}: {val:.6f}"
                logger.info(msg)

            batch_tic = time.time()
            reader_tic = time.time()

        return time_info, loss_info, metric_info

    def forward(self, batch_data):
        g, density, grid_coord, infos = batch_data

        device = paddle.get_device()
        g = g.to(device)
        density, grid_coord = density.to(device), grid_coord.to(device)
        for i, info in enumerate(infos):
            if "cell" in info and hasattr(info["cell"], "to"):
                infos[i]["cell"] = info["cell"].to(device)

        pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

        if self.use_voxel:
            mask = (density > 0).astype(dtype="float32")
            pred = pred * mask
            density = density * mask

        loss = self.criterion(pred, density)
        mae = paddle.abs(pred - density).sum() / density.sum()

        return {
            "loss_dict": {"loss": loss, "mae": mae},
            "pred_dict": {"density_mae": pred},
        }

    def train_step(self, batch_data):
        g, density, grid_coord, infos = batch_data

        device = paddle.get_device()
        g = g.to(device)
        density, grid_coord = density.to(device), grid_coord.to(device)
        for i, info in enumerate(infos):
            if "cell" in info and hasattr(info["cell"], "to"):
                infos[i]["cell"] = info["cell"].to(device)

        pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

        if self.use_voxel:
            mask = (density > 0).astype(dtype="float32")
            pred = pred * mask
            density = density * mask

        loss = self.criterion(pred, density)
        mae = paddle.abs(pred - density).sum() / density.sum()

        return {"loss": loss, "mae": mae}

    def eval_step(self, batch_data):
        g, density, grid_coord, infos = batch_data

        device = paddle.get_device()
        g = g.to(device)
        density, grid_coord = density.to(device), grid_coord.to(device)
        for i, info in enumerate(infos):
            if "cell" in info and hasattr(info["cell"], "to"):
                infos[i]["cell"] = info["cell"].to(device)

        pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

        if self.use_voxel:
            mask = (density > 0).astype(dtype="float32")
            pred = pred * mask
            density = density * mask

        loss = self.criterion(pred, density)
        mae = paddle.abs(pred - density).sum() / density.sum()

        return {"loss": loss, "mae": mae, "pred": pred, "density": density}

    def train(self):
        max_iter = self.config.get("max_iter", None)
        if max_iter is None:
            if self.config.get("max_epochs") is None:
                raise ValueError("set max_iter or max_epochs")
            max_iter = self.config["max_epochs"] * len(self.train_dataloader)
            logger.info(
                f"Based on max_epochs={self.config['max_epochs']} and dataset size, max_iter={max_iter}"
            )

        # Based on step not epoch
        self.state = TrainerState()
        logger.info("Training start...")
        trainable_params = self.get_num_trainable_parameters()
        logger.info(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")

        global_step = 0
        epoch = 0

        save_freq = self.config.get("save_freq", 1000)
        eval_freq = self.config.get("eval_freq", 1000)
        start_eval_step = self.config.get("start_eval_step", 0)

        while global_step < max_iter:
            self.state.epoch = epoch
            self.model.train()
            loss_info = {}
            metric_info = {}
            time_info = {
                "reader_cost": misc.AverageMeter(name="reader_cost", postfix="s"),
                "batch_cost": misc.AverageMeter(name="batch_cost", postfix="s"),
            }

            reader_tic = time.time()
            batch_tic = time.time()

            for iter_id, batch_data in enumerate(self.train_dataloader):
                reader_cost = time.time() - reader_tic
                time_info["reader_cost"].update(reader_cost)
                batch_size = self.guess_batch_size(batch_data, self.train_dataloader)
                g, density, grid_coord, infos = batch_data
                device = paddle.get_device()
                g = g.to(device)
                density, grid_coord = density.to(device), grid_coord.to(device)
                for i, info in enumerate(infos):
                    if "cell" in info and hasattr(info["cell"], "to"):
                        infos[i]["cell"] = info["cell"].to(device)

                pred = self.model(g.x, g.pos, grid_coord, g.batch, infos)

                if self.use_voxel:
                    mask = (density > 0).astype(dtype="float32")
                    pred = pred * mask
                    density = density * mask

                loss = self.criterion(pred, density)
                mae = paddle.abs(pred - density).sum() / density.sum()
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                global_step += 1
                if "loss" not in loss_info:
                    loss_info["loss"] = misc.AverageMeter("loss")
                loss_info["loss"].update(float(loss), batch_size)

                if "mae" not in loss_info:
                    loss_info["mae"] = misc.AverageMeter("mae")
                loss_info["mae"].update(float(mae), batch_size)

                if (
                    self.compute_metric_during_train
                    and self.compute_metric_func_dict is not None
                ):
                    for (
                        key,
                        compute_metric_func,
                    ) in self.compute_metric_func_dict.items():
                        metric = compute_metric_func(pred, density)
                        if key not in metric_info:
                            metric_info[key] = misc.AverageMeter(key)
                        metric_info[key].update(float(metric), batch_size)

                batch_cost = time.time() - batch_tic
                time_info["batch_cost"].update(batch_cost)

                if global_step % self.config["log_freq"] == 0 or global_step == 1:
                    logs = {}
                    logs["lr"] = self.optimizer.get_lr()
                    for name, average_meter in time_info.items():
                        logs[name] = average_meter.val
                    for name, average_meter in loss_info.items():
                        logs[name] = average_meter.val
                    for name, average_meter in metric_info.items():
                        logs[name] = average_meter.val

                    msg = f"Train: Epoch [{epoch}] | Global Step [{global_step}/{max_iter}]"
                    if logs:
                        for key, val in logs.items():
                            msg += f" | {key}: {val:.6f}"
                    logger.info(msg)

                if global_step % save_freq == 0 or global_step >= max_iter:
                    save_path = os.path.join(
                        self.output_dir, f"step_{global_step}.pdparams"
                    )
                    paddle.save(
                        obj={
                            "model": self.model.state_dict(),
                            "epoch": epoch,
                            "step": global_step,
                        },
                        path=save_path,
                    )
                    logger.info(f"Model saved to {save_path}")

                if (
                    global_step >= start_eval_step
                    and global_step % eval_freq == 0
                    and self.val_dataloader is not None
                ):
                    eval_time_info, eval_loss_info, eval_metric_info = self.eval_epoch(
                        self.val_dataloader
                    )

                    logs = {}
                    for name, average_meter in eval_time_info.items():
                        logs[name] = average_meter.avg
                    for name, average_meter in eval_loss_info.items():
                        logs[name] = average_meter.avg
                    for name, average_meter in eval_metric_info.items():
                        logs[name] = average_meter.avg

                    msg = f"Eval: Epoch [{epoch}] | Global Step [{global_step}/{max_iter}]"
                    if logs:
                        for key, val in logs.items():
                            msg += f" | {key}: {val:.6f}"
                    logger.info(msg)

                    self.model.train()

                    if self.lr_scheduler is not None:
                        avg_val_loss = (
                            eval_loss_info["loss"].avg
                            if "loss" in eval_loss_info
                            else None
                        )
                        scheduler_type = type(self.lr_scheduler).__name__
                        if "ReduceOnPlateau" in scheduler_type or (
                            hasattr(self.lr_scheduler, "_learning_rate")
                            and type(self.lr_scheduler._learning_rate).__name__
                            == "ReduceOnPlateau"
                        ):
                            self.lr_scheduler.step(avg_val_loss)
                        else:
                            self.lr_scheduler.step()
                if global_step >= max_iter:
                    logger.info(
                        f"Reached maximum iterations {max_iter}, stopping training"
                    )
                    return

                batch_tic = time.time()
                reader_tic = time.time()

            logs = {}
            for name, average_meter in time_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in loss_info.items():
                logs[name] = average_meter.avg
            for name, average_meter in metric_info.items():
                logs[name] = average_meter.avg

            msg = f"Train: Epoch [{epoch}] completed | Global Step [{global_step}/{max_iter}]"
            if logs:
                for key, val in logs.items():
                    msg += f" | {key}: {val:.6f}"
            logger.info(msg)

            epoch += 1

    def test(
        self,
        dataloader: paddle.io.DataLoader,
        num_infer=None,
        num_vis=2,
        inf_samples=None,
    ):
        assert dataloader is not None, "dataloader is None, please set it first"
        self.state = TrainerState()

        num_infer = num_infer or len(dataloader)
        time_info, loss_info, metric_info = self.test_epoch(
            dataloader, num_infer, num_vis, inf_samples
        )
        logs = {}
        for name, average_meter in time_info.items():
            logs[name] = average_meter.avg
        for name, average_meter in loss_info.items():
            logs[name] = average_meter.avg
        for name, average_meter in metric_info.items():
            logs[name] = average_meter.avg

        msg = "Test:"
        if logs is not None:
            for key, val in logs.items():
                msg += f" | {key}: {val:.6f}"
        logger.info(msg)

        return time_info, loss_info, metric_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./electron_density_prediction/configs/infgcn_md.yaml",
        help="Path to config file",
    )

    parser.add_argument("-d", "--device", type=str, default="gpu:0", help="cuda or cpu")

    args, dynamic_args = parser.parse_known_args()
    paddle.set_device(args.device)

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    os.makedirs(config["Trainer"]["output_dir"], exist_ok=True)
    config_name = os.path.basename(args.config)
    OmegaConf.save(config, os.path.join(config["Trainer"]["output_dir"], config_name))

    config = OmegaConf.to_container(config, resolve=True)
    logger_path = os.path.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")

    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    model_cfg = config["Model"]
    model = build_model(model_cfg)

    use_voxel = config["Global"].get("use_voxel", False)
    train_samples = config["Trainer"].get("train_samples", None)
    val_samples = config["Trainer"].get("val_samples", None)

    train_data_cfg = config["Dataset"].get("train")
    if train_data_cfg is None:
        logger.warning("Training dataset is not defined in the config")
        train_loader = None
    else:
        train_loader = build_dataloader(train_data_cfg)
        if train_loader is not None:
            if use_voxel:
                train_loader.collate_fn = DensityVoxelCollator()
            else:
                train_loader.collate_fn = DensityCollator(train_samples)

    val_data_cfg = config["Dataset"].get("val")
    if val_data_cfg is None:
        logger.warning("Validation dataset is not defined in the configuration")
        val_loader = None
    else:
        val_loader = build_dataloader(val_data_cfg)
        if val_loader is not None:
            if use_voxel:
                val_loader.collate_fn = DensityVoxelCollator()
            else:
                val_loader.collate_fn = DensityCollator(val_samples)

    test_data_cfg = config["Dataset"].get("test")
    if test_data_cfg is None:
        logger.warning("Test dataset is not defined in the configuration")
        test_loader = None
    else:
        test_loader = build_dataloader(test_data_cfg)
        if test_loader is not None:
            if use_voxel:
                test_loader.collate_fn = DensityVoxelCollator()
            else:
                test_loader.collate_fn = DensityCollator(None)

    if config.get("Optimizer") is not None:
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            config["Trainer"]["max_epochs"],
            len(train_loader),
        )
    else:
        optimizer, lr_scheduler = None, None

    metric_cfg = config.get("Metric")
    if metric_cfg is not None:
        metric_func = build_metric(metric_cfg)
    else:
        metric_func = None

    trainer = InfGCNTrainer(
        config["Trainer"],
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        compute_metric_func_dict=metric_func,
    )

    trainer.use_voxel = use_voxel

    if config["Global"].get("do_train", True):
        trainer.train()
    if config["Global"].get("do_eval", False):
        logger.info("Evaluate on the validation set")
        time_info, loss_info, metric_info = trainer.eval(val_loader)
    if config["Global"].get("do_test", False):
        logger.info("Evaluate on the test set")
        if "Predict" not in config:
            num_infer = None
            num_vis = 2
            inf_samples = 4096
        else:
            num_infer = config["Predict"].get("num_infer", None)
            num_vis = config["Predict"].get("num_vis", 2)
            inf_samples = config["Predict"].get("inf_samples", 4096)
        time_info, loss_info, metric_info = trainer.test(
            test_loader, num_infer, num_vis, inf_samples
        )
