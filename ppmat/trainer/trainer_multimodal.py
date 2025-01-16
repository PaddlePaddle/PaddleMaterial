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
from rdkit import Chem

from ppmat.metrics.abstract_metrics import NLL
from ppmat.metrics.abstract_metrics import SumExceptBatchKL
from ppmat.metrics.abstract_metrics import SumExceptBatchMetric
from ppmat.metrics.train_metrics import TrainLossDiscrete
from ppmat.models.digress import diffusion_utils
from ppmat.models.digress.base_model import ConditionGraphTransformer
from ppmat.models.digress.base_model import ContrastGraphTransformer

# from ppmat.models.digress.base_model import MolecularGraphTransformer
from ppmat.models.digress.noise_schedule import DiscreteUniformTransition
from ppmat.models.digress.noise_schedule import MarginalUniformTransition
from ppmat.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete
from ppmat.models.digress.utils import digressutils as utils

# from ppmat.trainer.trainer_diffusion import TrainerDiffusion
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.io import read_json  # noqa
from ppmat.utils.io import write_json


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
        self.epochs = config["Global"]["epochs"]
        self.output_dir = config["Global"]["output_dir"]
        self.save_freq = config["Global"]["save_freq"]
        self.log_freq = config["Global"]["log_freq"]
        self.start_eval_epoch = config["Global"]["start_eval_epoch"]
        self.eval_freq = config["Global"]["eval_freq"]
        self.seed = config["Global"]["seed"]
        self.pretrained_model_path = config["Global"].get("pretrained_model_path", None)
        self.checkpoint_path = config["Global"].get("checkpoint_path", None)
        self.scale_grad = config["Global"].get("scale_grad", False)
        self.is_save_traj = config["Global"].get("is_save_traj", False)
        self.step_lr = config["Global"].get("step_lr", 0.000005)

        self.iters_per_epoch = len(self.train_dataloader)

        if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            if (
                self.config["Optimizer"]["lr"].get("indicator", "train_loss")
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
                        self.config["Optimizer"]["lr"].get("indicator", "train_loss")
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
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = "float32"  # 原先是torch.float32
        self.T = cfg.model.diffusion_steps

        # 以下是一些网络结构相关超参
        self.enc_voc_size = 5450
        self.max_len = 256
        self.d_model = 256
        self.ffn_hidden = 1024
        self.n_head = 8
        self.n_layers_TE = 3
        self.drop_prob = 0.0

        # Paddle 不需要手动获取 device，一般通过 paddle.set_device("gpu") 或 "cpu"
        # self.device = paddle.device.get_device()

        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.tem = 2  # 温度/缩放参数
        self.val_loss = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # 构造 backbone (contrastGT)
        self.model = ContrastGraphTransformer(
            n_layers_GT=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            enc_voc_size=self.enc_voc_size,
            max_len=self.max_len,
            d_model=self.d_model,
            ffn_hidden=self.ffn_hidden,
            n_head=self.n_head,
            n_layers_TE=self.n_layers_TE,
            drop_prob=self.drop_prob
            # device=self.device  # 在 Paddle 下通常不需显式传
        )

        # 噪声日程表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps
        )

        # Transition Model
        if cfg.model.transition == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == "marginal":
            node_types = self.dataset_info.node_types.astype("float32")
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype("float32")
            e_marginals = edge_types / paddle.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, "
                f"{e_marginals} for edges"
            )

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output,
            )

        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256

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


class TrainerMultiModal:
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = "float32"
        self.T = cfg.model.diffusion_steps

        self.enc_voc_size = 5450
        self.max_len = 256
        self.d_model = 256
        self.ffn_hidden = 1024
        self.n_head = 8
        self.n_layers_TE = 3
        self.drop_prob = 0.0
        self.device = "gpu"  # Paddle 通常用 paddle.set_device("gpu") 来设置设备

        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # 训练损失 & 一些指标
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # 构建 ConditionGT 模型
        self.model = ConditionGraphTransformer(
            n_layers_GT=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            enc_voc_size=self.enc_voc_size,
            max_len=self.max_len,
            d_model=self.d_model,
            ffn_hidden=self.ffn_hidden,
            n_head=self.n_head,
            n_layers_TE=self.n_layers_TE,
            drop_prob=self.drop_prob,
            device=self.device,
        )

        # 噪声日程表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps
        )

        # Transition Model
        if cfg.model.transition == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = (
                paddle.ones([self.Xdim_output], dtype="float32") / self.Xdim_output
            )
            e_limit = (
                paddle.ones([self.Edim_output], dtype="float32") / self.Edim_output
            )
            y_limit = (
                paddle.ones([self.ydim_output], dtype="float32") / self.ydim_output
            )
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == "marginal":
            node_types = paddle.to_tensor(self.dataset_info.node_types, dtype="float32")
            x_marginals = node_types / paddle.sum(node_types)
            edge_types = paddle.to_tensor(self.dataset_info.edge_types, dtype="float32")
            e_marginals = edge_types / paddle.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, "
                f"{e_marginals} for edges"
            )
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            y_limit = (
                paddle.ones([self.ydim_output], dtype="float32") / self.ydim_output
            )
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=y_limit)

        # 其余属性
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256

    # -------------------------
    # 优化器 (可选)
    # -------------------------
    def configure_optimizers(self):
        return paddle.optimizer.AdamW(
            parameters=self.parameters(),
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

    def train_step(self, data, i):

        if data.edge_index.size(1) == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        # to_dense
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

        # forward
        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)
        # compute loss
        loss_val = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=(i % self.log_every_steps == 0),
        )
        if i % 80 == 0:
            print(f"train_loss: {loss_val.numpy()[0]}")
        # train_metrics
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=(i % self.log_every_steps == 0),
        )
        sys.stdout.flush()
        return {"loss": loss_val}

    # -------------------------
    # 验证循环 => val_step
    # -------------------------
    def val_step(self, data, i):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # apply noise
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)

        self.val_y_collection.append(data.conditionVec)
        self.val_atomCount.append(data.atom_count)
        self.val_x.append(X)
        self.val_e.append(E)

        # 计算 training loss
        loss_val = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=(i % self.log_every_steps == 0),
        )
        if i % 10 == 0:
            print(f"val_loss: {loss_val.numpy()[0]}")

        # 进一步计算NLL
        nll = self.compute_val_loss(
            pred,
            noisy_data,
            X,
            E,
            data.y,
            node_mask,
            condition=conditionAll,
            test=False,
        )
        return {"loss": nll}

    # -------------------------
    # 测试循环 => test_step
    # -------------------------
    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)

        self.test_y_collection.append(data.conditionVec)
        self.test_atomCount.append(data.atom_count)
        self.test_x.append(X)
        self.test_e.append(E)

        nll = self.compute_val_loss(
            pred, noisy_data, X, E, data.y, node_mask, condition=conditionAll, test=True
        )
        return {"loss": nll}

    # -------------------------
    # 噪声 & Q
    # -------------------------
    def apply_noise(self, X, E, y, node_mask):

        bs = X.shape[0]
        lowest_t = 1
        t_int = paddle.randint(
            lowest_t, self.T + 1, shape=[bs, 1], dtype="int64"
        ).astype("float32")
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        # probX = X @ Qtb.X => paddle.matmul
        probX = paddle.matmul(X, Qtb.X)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))

        sampled_t = diffusion_utils.sample_discrete_features(probX, probE, node_mask)
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

    def compute_val_loss(
        self, pred, noisy_data, X, E, y, node_mask, condition, test=False
    ):

        t = noisy_data["t"]
        N = paddle.sum(node_mask, axis=1).astype("int64")
        log_pN = self.node_dist.log_prob(N)

        # kl_prior => uniform
        kl_prior_ = self.kl_prior(X, E, node_mask)

        # diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # reconstruction
        prob0 = self.reconstruction_logp(t, X, E, node_mask, condition)
        # val_X_logp, val_E_logp
        loss_term_0 = self.val_X_logp(
            X * paddle.log(prob0.X + 1e-10)
        ) + self.val_E_logp(E * paddle.log(prob0.E + 1e-10))

        nlls = -log_pN + kl_prior_ + loss_all_t - loss_term_0
        nll = (self.test_nll if test else self.val_nll)(nlls)

        # 在 Paddle 下若需 wandb，需要 import wandb 并保证 wandb.run != None
        # ...
        return nll

    # -------------------------
    # forward => 组装拼接 + 调用 ConditionGT
    # -------------------------
    def forward(self, noisy_data, extra_data, node_mask, condition):

        X_ = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype("float32")
        E_ = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype("float32")
        y_ = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype("float32")

        condition_t = paddle.to_tensor(condition, dtype="int64")
        return self.model(X_, E_, y_, node_mask, condition_t)

    def forward_sample(self, noisy_data, extra_data, node_mask, condition):

        X_ = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype("float32")
        E_ = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype("float32")
        y_ = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype("float32")
        condition_t = paddle.to_tensor(condition, dtype="int64")
        return self.model(X_, E_, y_, node_mask, condition_t)

    # -------------------------
    # KL prior
    # -------------------------
    def kl_prior(self, X, E, node_mask):
        bs = X.shape[0]
        ones = paddle.ones([bs, 1], dtype="float32")
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        probX = paddle.matmul(X, Qtb.X)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))
        # limit
        limit_X = paddle.expand(
            self.limit_dist.X.unsqueeze(0).unsqueeze(0), [bs, X.shape[1], -1]
        )
        limit_E = paddle.expand(
            self.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            [bs, E.shape[1], E.shape[2], -1],
        )
        # mask
        (
            limit_dist_X,
            limit_dist_E,
            probX_m,
            probE_m,
        ) = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
        )
        kl_distance_X = F.kl_div(
            paddle.log(probX_m + 1e-10), limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            paddle.log(probE_m + 1e-10), limit_dist_E, reduction="none"
        )

        return diffusion_utils.sum_except_batch(
            kl_distance_X
        ) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):

        pred_probs_X = F.softmax(pred.X, axis=-1)
        pred_probs_E = F.softmax(pred.E, axis=-1)
        pred_probs_y = F.softmax(pred.y, axis=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

        bs, n, _ = X.shape
        prob_true = diffusion_utils.posterior_distributions(
            X=X,
            E=E,
            y=y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_true.E = paddle.reshape(prob_true.E, [bs, n, n, -1])

        prob_pred = diffusion_utils.posterior_distributions(
            X=pred_probs_X,
            E=pred_probs_E,
            y=pred_probs_y,
            X_t=noisy_data["X_t"],
            E_t=noisy_data["E_t"],
            y_t=noisy_data["y_t"],
            Qt=Qt,
            Qsb=Qsb,
            Qtb=Qtb,
        )
        prob_pred.E = paddle.reshape(prob_pred.E, [bs, n, n, -1])

        # mask
        (
            prob_true_X,
            prob_true_E,
            prob_pred_X,
            prob_pred_E,
        ) = diffusion_utils.mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask,
        )

        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true_X, paddle.log(prob_pred_X + 1e-10)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true_E, paddle.log(prob_pred_E + 1e-10)
        )
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, condition):

        t_zeros = paddle.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = paddle.matmul(X, Q0.X)
        probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

        sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).astype("float32")
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).astype("float32")
        y0 = sampled0.y

        # forward
        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)
        zeros_t = paddle.zeros([X0.shape[0], 1], dtype="float32")
        noised_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t": zeros_t,
        }
        extra_data = self.compute_extra_data(noised_data)
        pred0 = self.forward(noised_data, extra_data, node_mask, condition)

        probX0 = F.softmax(pred0.X, axis=-1)
        probE0 = F.softmax(pred0.E, axis=-1)
        proby0 = F.softmax(pred0.y, axis=-1)

        # 屏蔽无效节点
        probX0 = paddle.where(
            ~node_mask.unsqueeze(-1), paddle.ones_like(probX0) * 1.0, probX0
        )
        # E
        node_mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0 = paddle.where(
            ~node_mask_2d.unsqueeze(-1), paddle.ones_like(probE0) * 1.0, probE0
        )

        diag_mask = paddle.eye(probE0.shape[1], dtype="bool")
        diag_mask = diag_mask.unsqueeze(0)
        diag_mask = diag_mask.expand([probE0.shape[0], -1, -1])
        probE0 = paddle.where(
            diag_mask.unsqueeze(-1), paddle.ones_like(probE0) * 1.0, probE0
        )

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    # -------------------------
    # 采样 => sample_batch
    # -------------------------
    @paddle.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        batch_condition,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        batch_X,
        batch_E,
        num_nodes=None,
    ):

        # 这里是反向扩散采样逻辑
        # 与Lightning下相同，只需把 torch.* -> paddle.* 并注意张量形状
        ...

    def mol_from_graphs(self, node_list, adjacency_matrix):

        atom_decoder = self.dataset_info.atom_decoder
        mol = Chem.RWMol()

        node_to_idx = {}
        for i, nd in enumerate(node_list):
            if nd == -1:
                continue
            a = Chem.Atom(atom_decoder[int(nd)])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    # -------------------------
    # 如果想仿Lightning的回调
    # -------------------------
    def on_train_epoch_start(self):
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self):
        to_log = self.train_loss.log_epoch_metrics()
        print(
            f"Epoch XX: X_CE: {to_log['train_epoch/x_CE'] :.3f}, "
            f"E_CE: {to_log['train_epoch/E_CE'] :.3f}, "
            f"y_CE: {to_log['train_epoch/y_CE'] :.3f}, "
            f"Time: {time.time() - self.start_epoch_time:.1f}s"
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        print(f"Train epoch end: {epoch_at_metrics} -- {epoch_bond_metrics}")

    def on_validation_epoch_start(self):
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e = []

    def on_validation_epoch_end(self):
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute(),
        ]
        print(
            f"Val NLL {metrics[0]:.2f} | Val Atom KL {metrics[1]:.2f} | "
            f"Val Edge KL {metrics[2]:.2f}"
        )

    def on_test_epoch_start(self):
        print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []

    def on_test_epoch_end(self):
        print("Done testing.")
