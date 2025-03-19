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
import time
from collections import defaultdict
from typing import Callable
from typing import Optional

import paddle
import paddle.distributed as dist
from packaging import version
from paddle import DataParallel as DP
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet

from ppmat.models.denmr.utils import diffgraphformer_utils as utils
from ppmat.models.denmr.utils import model_utils as m_utils
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.io import read_json  # noqa
from ppmat.utils.io import write_json


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


class TrainerDiffGraphFormer:
    def __init__(
        self,
        config,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        sample_dataloader: Optional[paddle.io.DataLoader] = None,
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
        self.sample_dataloader = sample_dataloader
        self.test_dataloader = test_dataloader
        self.metric_class = metric_class
        self.lr_scheduler = lr_scheduler

        # get trainer config
        self.epochs = config["Trainer"]["epochs"]
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

        # get sample config
        self.samp_per_val = self.config["Sampler"]["sample_every_val"]
        self.visual_num = self.config["Sampler"]["visual_num"]
        self.chains_left_to_save = self.config["Sampler"]["chains_to_save"]
        self.number_chain_steps = self.config["Sampler"]["number_chain_steps"]
        self.sample_batch_iters = self.config["Sampler"]["sample_batch_iters"]

        # get tracker config
        self.output_dir = config["Tracker"]["save"]["output_dir"]
        self.save_freq = config["Tracker"]["save"]["save_freq"]
        self.log_freq = config["Tracker"]["log"]["log_freq"]

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

        # other
        self.vocabDim = self.config["Trainer"]["vocab_dim"]

    def train(self) -> None:
        """Training."""
        self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = self.best_metric["epoch"] + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            # train epoch
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch_id)

            # log train epoch info
            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for k, v in train_loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
            logger.info(msg)

            # eval epoch
            save_metric_dict = {"epoch": epoch_id}
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_dict, metric_dict = self.eval_epoch(
                    self.val_dataloader, epoch_id
                )
                save_metric_dict.update(eval_loss_dict)

                # log eval epoch loss & metric info
                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                for k, v in eval_loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
                for k, v in metric_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}"
                        if k == "metric"
                        else f" | {k}(metric): {v:.5f}"
                    )
                logger.info(msg)

                # update best metric
                if eval_loss_dict["train_loss"] <= self.best_metric["loss"]:
                    self.best_metric.update(eval_loss_dict)
                    self.best_metric["epoch"] = epoch_id

                    save_load.save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.best_metric,
                        output_dir=self.output_dir,
                        prefix="best",
                    )

            # sample epoch
            if (
                epoch_id % (self.samp_per_val * self.eval_freq) == 0
                and dist.get_rank() == 0
            ):
                start = time.time()

                # eval sample epoch
                metric_dict = self.sample_epoch(self.sample_dataloader, epoch_id)

                # log eval sample metric info
                msg = f"Sample: Epoch [{epoch_id}/{self.epochs}] "
                msg += f" | sample_metric cost: {time.time() - start:.5f}s"
                for k, v in metric_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item() if v.numel() == 1 else v.tolist()
                    msg += (
                        f" | {k}(metric): {', '.join(f'{x:.5f}' for x in v)}"
                        if isinstance(v, (list, tuple))
                        else f" | {k}(metric): {v:.5f}"
                    )
                logger.info(msg)

            # update learning rate by epoch
            if self.lr_scheduler is not None and self.lr_scheduler.by_epoch:
                if isinstance(self.lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                    if (
                        self.config["Trainer"]["Optimizer"]["lr"].get(
                            "indicator", "train_loss"
                        )
                        == "train_loss"
                    ):
                        train_loss = train_loss_dict["train_loss"]
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
            if iter_id == 1:  # TODO: for debug
                break
            reader_cost = time.perf_counter() - reader_tic

            loss_dict, metric_dict = self.model(batch_data, mode="train")

            loss = loss_dict["train_loss"]
            loss.backward()
            if self.scale_grad:
                scale_shared_grads(self.model)

            self.optimizer.step()
            self.optimizer.clear_grad()

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            # TODO: distrubuted
            if self.world_size > 1:
                fleet.utils.hybrid_parallel_util.fused_allreduce_gradients(
                    list(self.model.parameters()), None
                )

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
                for k, v in metric_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    msg += (
                        f" | {k}: {v:.5f}"
                        if k == "metric"
                        else f" | {k}(metric): {v:.5f}"
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
        total_metric = defaultdict(list)

        data_length = len(dataloader)
        for iter_id, batch_data in enumerate(dataloader):
            if iter_id == 1:  # TODO: for debug
                break
            reader_cost = time.perf_counter() - reader_tic

            loss_dict, metric_dict = self.model(batch_data, mode="eval")

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)
            for key, value in metric_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_metric[key].append(value)

            batch_cost = time.perf_counter() - batch_tic
            if paddle.distributed.get_rank() == 0 and (
                iter_id % self.log_freq == 0 or iter_id == data_length - 1
            ):
                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
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
        total_metric_avg = {k: sum(v) / len(v) for k, v in total_metric.items()}

        return total_loss_avg, total_metric_avg

    @paddle.no_grad()
    def sample_epoch(self, dataloader, epoch_id: int):
        self.model.eval()
        iters = self.sample_batch_iters  # TODO: use iterdataset for sampling

        # init samples dict for transfer into calculate metrics
        samples = dict()
        samples["pred"] = list()
        samples["true"] = list()
        samples["n_all"] = 0
        if isinstance(self.model, DP):
            samples["dict"] = self.model._layers.dataset_info.atom_decoder
        else:
            samples["dict"] = self.model.dataset_info.atom_decoder

        for iter_id, batch_data in enumerate(dataloader):

            # prepare the batch data
            batch_graph, other_data = batch_data
            dense_data, node_mask = utils.to_dense(
                batch_graph.node_feat["feat"],
                batch_graph.edges.T,
                batch_graph.edge_feat["feat"],
                batch_graph.graph_node_id,
            )
            dense_data = dense_data.mask(node_mask)
            batch_nmr = other_data["conditionVec"]
            batch_atomCount = other_data["atom_count"]
            batch_y = other_data["y"]
            batch_X, batch_E = dense_data.X, dense_data.E
            bs = len(batch_y)

            # sample from the model
            molecule_list, molecule_list_True = m_utils.sample_batch(
                self.model._layers if isinstance(self.model, DP) else self.model,
                batch_id=iter_id,
                num_nodes=batch_atomCount,
                batch_condition=batch_nmr,
                batch_X=batch_X,
                batch_E=batch_E,
                batch_y=batch_y,
                batch_size=bs,
                visual_num=self.visual_num,
                keep_chain=self.chains_left_to_save,
                number_chain_steps=self.number_chain_steps,
            )

            # save the samples for calculate metrics
            samples["pred"].extend(molecule_list)
            samples["true"].extend(molecule_list_True)
            samples["n_all"] += len(batch_y)

            # contral iters
            iters -= 1
            if iters == 0:
                break

        # sampled molecules to compute metrics
        if isinstance(self.model, DP):
            metric_dict = self.model._layers.sampling_metrics.forward(
                samples,
                epoch_id,
                val_counter=-1,
                test=True,
                local_rank=self.rank,
                output_dir=self.output_dir,
            )
        else:
            metric_dict = self.model.sampling_metrics.forward(
                samples,
                epoch_id,
                val_counter=-1,
                test=True,
                local_rank=self.rank,
                output_dir=self.output_dir,
            )
        return metric_dict

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

    def train(self) -> None:
        # TODO:self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
        self.max_steps = self.epochs * self.iters_per_epoch

        start_epoch = 0 + 1

        for epoch_id in range(start_epoch, self.epochs + 1):
            # train epoch
            train_loss_dict = self.train_epoch(self.train_dataloader, epoch_id)

            # log train epoch info
            msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
            for k, v in train_loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
            logger.info(msg)

            # eval epoch
            save_metric_dict = {"epoch": epoch_id}
            if (
                epoch_id >= self.start_eval_epoch
                and self.eval_freq > 0
                and epoch_id % self.eval_freq == 0
                and dist.get_rank() == 0
            ):
                eval_loss_dict = self.eval_epoch(self.val_dataloader, epoch_id)

                # log eval epoch info
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

            loss_dict = self.model(batch_data)
            loss = loss_dict["loss"]
            loss.backward()

            self.optimizer.step()
            self.optimizer.clear_grad()

            for key, value in loss_dict.items():
                if isinstance(value, paddle.Tensor):
                    value = value.item()
                total_loss[key].append(value)

            # TODO
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


class TrainerDiffPrior:
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

        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

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
            loaded_metric = save_load.load_checkpoint(
                self.checkpoint_path,
                self.model,
                self.optimizer,
            )
            if isinstance(loaded_metric, dict):
                self.best_metric.update(loaded_metric)

        # Obtain rank information of the current process
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
        # TODO:self.global_step = self.best_metric["epoch"] * self.iters_per_epoch
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

            # TODO: if solver.world_size > 1:
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
        """Gradient update, scheduler step"""
        self.optimizer.step()
        self.optimizer.clear_grad()

        # Update the scheduler based on the existence of a warmup scheduler
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.step()
        else:
            self.lr_scheduler.step()

        self.global_step += 1

    def eval(self):
        loss_dict = self.eval_epoch(self.val_dataloader, epoch_id=1)
        msg = "Eval: "
        for k, v in loss_dict.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
        logger.info(msg)
        return loss_dict


class TrainerMMDecoder(TrainerDiffGraphFormer):
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
