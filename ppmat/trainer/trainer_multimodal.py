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

import time
from collections import defaultdict
from typing import Callable, Dict, Union
from typing import Optional
import numpy as np

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
        self.loss_dict_train = config["Tracker"]["log"]["out_dict"]["loss"].get(
            "train", None
        )
        self.loss_dict_eval = config["Tracker"]["log"]["out_dict"]["loss"].get(
            "eval", None
        )
        self.metric_dict_train = config["Tracker"]["log"]["out_dict"]["metric"].get(
            "train", None
        )
        self.metric_dict_eval = config["Tracker"]["log"]["out_dict"]["metric"].get(
            "eval", None
        )
        self.metric_dict_sample = config["Tracker"]["log"]["out_dict"]["metric"].get(
            "sample", None
        )
        self.flag_train_step = config["Tracker"]["log"]["flag_train_step"]
        self.flag_eval_step = config["Tracker"]["log"]["flag_eval_step"]

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
        self.flag_use_formula = self.config["Trainer"].get("flag_use_formula", False)
        self.flag_keep_onehot = self.config["Trainer"].get("flag_keep_onehot", False)
        self.num_candidate: int = self.config["Trainer"].get("num_candidate", 1)

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
                if k in self.loss_dict_train or self.loss_dict_train is None:
                    msg += (
                        f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
                    )
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
                    if k in self.loss_dict_eval or self.loss_dict_eval is None:
                        msg += (
                            f" | {k}: {v:.5f}"
                            if k == "loss"
                            else f" | {k}(loss): {v:.5f}"
                        )
                for k, v in metric_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    if k in self.metric_dict_eval or self.metric_dict_eval is None:
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
                    if k in self.metric_dict_sample or self.metric_dict_sample is None:
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

    def test(self):

        self.model.eval()
        epoch_id = 0

        eval_loss_dict, metric_dict = self.eval_epoch(self.test_dataloader, epoch_id)

        # log eval epoch loss & metric info
        msg = f"Test: Epoch [{epoch_id+1}/1]"
        for k, v in eval_loss_dict.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            if k in self.loss_dict_eval or self.loss_dict_eval is None:
                msg += f" | {k}: {v:.5f}" if k == "loss" else f" | {k}(loss): {v:.5f}"
        for k, v in metric_dict.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            if k in self.metric_dict_eval or self.metric_dict_eval is None:
                msg += (
                    f" | {k}: {v:.5f}" if k == "metric" else f" | {k}(metric): {v:.5f}"
                )
        logger.info(msg)

        data_length = len(self.test_dataloader)
        logger.message(f"Start to sample ... | Total Batches: {data_length}")
        start = time.time()

        # sample epoch
        metric_dict = self.sample_epoch(
            self.test_dataloader, 
            epoch_id, 
            flag_sample=True, 
            keep_onehot=self.flag_keep_onehot, 
            num_candidate=self.num_candidate
        )

        # log eval sample metric info
        if paddle.distributed.get_rank() == 0:
            msg = "Test: Sample:"
            msg += f" | sample_metric cost: {time.time() - start:.5f}s"
            for k, v in metric_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item() if v.numel() == 1 else v.tolist()
                if self.metric_dict_sample is None or k in self.metric_dict_sample:
                    msg += (
                        f" | {k}(metric): {', '.join(f'{x:.5f}' for x in v)}"
                        if isinstance(v, (list, tuple))
                        else f" | {k}(metric): {v:.5f}"
                    )
            logger.info(msg)

    def train_epoch(self, dataloader, epoch_id: int):
        """Train one epoch (sample-weighted averaging).

        Args:
            epoch_id (int): Epoch id.
        Returns
        -------
        dict
            {loss_name: epoch_average}
        """
        reader_tic = batch_tic = time.perf_counter()

        self.model.train()

        # accumlators
        loss_sum = defaultdict(float)    # Σ (batch_mean × batch_size)
        total_samples = 0   # Σ batch_size

        data_length = len(dataloader)

        # ---------------------------------------------------
        for iter_id, batch_data in enumerate(dataloader):

            reader_cost = time.perf_counter() - reader_tic

            loss_dict, metric_dict = self.model(batch_data, mode="train")

            loss = loss_dict["train_loss"]
            loss.backward()

            if self.scale_grad:
                scale_shared_grads(self.model)

            self.optimizer.step()
            self.optimizer.clear_grad()

            # sample-weightd accumulate
            batch_graph, _ = batch_data
            bs = batch_graph.num_graph
            if isinstance(bs, paddle.Tensor):        # 0-dim Tensor → int
                bs = int(bs.item())
            elif isinstance(bs, (list, tuple)):      # list → get length
                bs = len(bs)
            else:                                    # already int
                bs = int(bs)

            for k, v in loss_dict.items():
                v = v.item() if isinstance(v, paddle.Tensor) else float(v)
                loss_sum[k] += v * bs

            total_samples += bs

            # distributed grad all-reduce
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

            # update and log training information
            batch_cost = time.perf_counter() - batch_tic
            self.global_step += 1

            if (
                paddle.distributed.get_rank() == 0
                and (iter_id % self.log_freq == 0 or iter_id == data_length - 1)
                and self.flag_train_step is True
            ):
                lr_val = self.optimizer._learning_rate
                lr_val = lr_val() if callable(lr_val) else lr_val
                msg = f"Train: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | lr: {lr_val:.6f}".rstrip("0")
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"

                for k, v in loss_dict.items():
                    v = v.item() if isinstance(v, paddle.Tensor) else v
                    if self.loss_dict_train is None or k in self.loss_dict_train:
                        tag = k if k == "loss" else f"{k}(loss)"
                        msg += f" | {tag}: {v:.5f}"
                for k, v in metric_dict.items():
                    v = v.item() if isinstance(v, paddle.Tensor) else v
                    if self.metric_dict_train is None or k in self.metric_dict_train:
                        tag = k if k == "metric" else f"{k}(metric)"
                        msg += f" | {tag}: {v:.5f}"

                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        # epoch-average
        total_loss_avg = {k: sum(v) / len(v) for k, v in loss_sum.items()}

        return total_loss_avg

    @paddle.no_grad()
    def eval_epoch(self, dataloader, epoch_id: int):
        """Run one validation/test epoch and return averaged metrics.

        The averaging is **sample-weighted**, so the final result is correct
        even when the last mini-batch is smaller than others.
        """

        reader_tic = batch_tic = time.perf_counter()

        self.model.eval()
    
        # Accumulators (sum over all samples, not over batches)
        loss_sum = defaultdict(float)
        metric_sum = defaultdict(float)
        total_samples = 0 # record total samples processed

        data_length = len(dataloader)

        for iter_id, batch_data in enumerate(dataloader):

            reader_cost = time.perf_counter() - reader_tic

            loss_dict, metric_dict = self.model(batch_data, mode="eval")

            batch_graph, _ = batch_data
            bs = batch_graph.num_graph
            if isinstance(bs, paddle.Tensor):        # 0-dim Tensor → int
                bs = int(bs.item())
            elif isinstance(bs, (list, tuple)):      # list → get length
                bs = len(bs)
            else:                                    # already int
                bs = int(bs)

            # sample-weighted accumulate
            for k, v in loss_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                    loss_sum[k] += v * bs

            for k, v in metric_dict.items():
                if isinstance(v, paddle.Tensor):
                    v = v.item()
                metric_sum[k] += v * bs
            
            total_samples += bs

            # ----------------------------------------------------------
            batch_cost = time.perf_counter() - batch_tic
            if (
                paddle.distributed.get_rank() == 0
                and (iter_id % self.log_freq == 0 or iter_id == data_length - 1)
                and self.flag_eval_step is True
            ):
                msg = f"Eval: Epoch [{epoch_id}/{self.epochs}]"
                msg += f" | Step: [{iter_id+1}/{data_length}]"
                msg += f" | reader cost: {reader_cost:.5f}s"
                msg += f" | batch cost: {batch_cost:.5f}s"
                
                # print desired loss and metric
                for k, v in loss_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v = v.item()
                    if k in self.loss_dict_eval or self.loss_dict_eval is None:
                        msg += (
                            f" | {k}: {v:.5f}"
                            if k == "loss"
                            else f" | {k}(loss): {v:.5f}"
                        )
                logger.info(msg)

            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()

        loss_avg = {k: v / total_samples for k, v in loss_sum.items()}
        metric_avg = {k: v / total_samples for k, v in metric_sum.items()}

        return loss_avg, metric_avg

    @paddle.no_grad()
    def sample_epoch(
        self, 
        dataloader: paddle.io.DataLoader, 
        epoch_id: int, 
        num_candidate: int = 1,
        keep_onehot: bool = False,
        flag_sample:bool = False
    ):
        """Run **one full sampling pass** over ``dataloader`` and collect metrics.

        This wrapper repeatedly calls :func:`sample_batch` to generate *multiple*  
        candidate molecules for every ground‑truth graph in the batch. The first  
        candidate of each batch is treated as the *default prediction* used for  
        classical metrics (validity, novelty, etc.). All *num_candidate* variants  
        can optionally be forwarded to *retrieval‑based* metrics that compare  
        `molVec` embeddings against an NMR‑condition embedding.

        Parameters
        ----------
        self : TrainerLike
            Trainer / Runner object that holds the diffusion ``model``, runtime
            configs, logging utilities, etc.
        dataloader : paddle.io.DataLoader
            Yields tuples ``(graph, aux_data)`` where `graph` is a *pgl* style
            MiniBatchGraph and `aux_data` is a dict containing scalar labels,
            condition vectors and atom counts. TODO: recheck details.
        epoch_id : int
            Current epoch index – propagated to the metric logger so that saved
            artefacts (csv / images) are grouped by epoch.
        flag_sample : bool, default ``False``
            *True*  – run through the *entire* dataloader regardless of
            ``self.sample_batch_iters`` (used for final sampling).
            *False* – stop early after ``self.sample_batch_iters`` iterations.
        num_candidate : int, default 1
            How many independent candidate graphs to sample **per ground‑truth**
            (first one is *pred*, remained serve retrieval evaluation).
        keep_onehot : bool, default ``True``
            If *True* each candidate also returns padded one‑hot tensors
            ``X_hot / E_hot`` that are later required by the `molVec` encoder. If
            retrieval metrics are disabled you can set this to *False* to save
            memory.

        Returns
        -------
        dict
            A flattened dictionary of scalar metrics produced by
            :class:`SamplingMolecularMetrics` (top‑k accuracy, RDKit validity,
            histogram MAE, etc.).

        Workflow
        --------
        1. Initialise an empty ``samples`` dict – this will be the *single* payload
        passed to :pyclass:`SamplingMolecularMetrics`.
        2. Iterate over the dataloader
        • convert sparse PGL graph → dense tensors (node/edge one‑hot).
        • build four‑branch NMR condition vector.
        • call :func:`sample_batch` ``num_candidate`` times.
        • aggregate predictions, ground‑truth and (optionally) one‑hot tensors.
        3. Early‑exit when ``iters_left`` hits zero (unless *flag_sample* forces a
        full pass).
        4. Call the metric layer *once* – avoids repeated RDKit initialisation and
        keeps logging atomic.
        """
        # Put the model in eval‑mode so layers like dropout / batch‑norm are frozen
        self.model.eval()
        
        # used for early‑stopping a long dataloader when we only need a subset.
        # When `flag_sample=True` the entire dataloader will be exhausted regardless
        # of this limit.
        max_iters: int = self.sample_batch_iters

        # 1. pre‑allocate the data structure that SamplingMolecularMetrics expects
        samples: Dict[str, Union[list, int]] = {
            "pred"  : [],   # first candidate of each ground‑truth
            "true"  : [],   # ground‑truth graphs
            "n_all" : 0,    # total number of GT molecules processed
            "node_mask_meta": [], # node mask metadata for each batch
            "batch_condition": [ None for _ in range(4) ], # 4‑branch NMR condition
            "dict"  : (
                self.model._layers if isinstance(self.model, paddle.DataParallel)
                else self.model
            ).dataset_info.atom_decoder,   # id → element symbol
        }
        if keep_onehot:
            # For retrieval metrics we need to keep *all* candidates & their one‑hot
            samples["candidates"]   = [[] for _ in range(num_candidate)]
            samples["candidates_X"] = [[] for _ in range(num_candidate)]
            samples["candidates_E"] = [[] for _ in range(num_candidate)]

        # 2. main loop over DataLoader
        for iter_id, (batch_graph, aux) in enumerate(dataloader):
            # 2.a convert sparse graph to dense (one‑hot padded) representation
            dense_data, node_mask = utils.to_dense(
                batch_graph.node_feat["feat"],
                batch_graph.edges.T,
                batch_graph.edge_feat["feat"],
                batch_graph.graph_node_id,
            )
            dense_data = dense_data.mask(node_mask) # remove padding rows

            # basic batch tensors
            batch_atomCount = aux["atom_count"]           # [B] number of atoms
            batch_y = aux["y"]                            # labels (unused here)
            batch_X, batch_E = dense_data.X, dense_data.E # one‑hot Node / Edge
            bs = len(batch_y)                             # batch size

            # 2.b build four‑branch NMR condition tensor list
            cond_H = aux["conditionVec"]["H_nmr"].reshape(bs, self.model.seq_len_H1, -1)
            cond_C = aux["conditionVec"]["C_nmr"].reshape(bs, self.model.seq_len_C13)
            num_H_peak = aux["conditionVec"]["num_H_peak"]
            num_C_peak = aux["conditionVec"]["num_C_peak"]
            batch_nmr = [cond_H, num_H_peak, cond_C, num_C_peak]

            # 2.c call `sample_batch` `num_candidate` times
            for c_idx in range(num_candidate):
                res = m_utils.sample_batch(
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
                    return_onehot=keep_onehot,
                    flag_useformula=self.flag_use_formula,
                    iter_idx = c_idx,
                )
                
                if keep_onehot:
                    mol_pred, mol_true, X_hot, E_hot= res
                    samples["candidates"  ][c_idx].extend(mol_pred)
                    samples["candidates_X"][c_idx].extend(X_hot)
                    samples["candidates_E"][c_idx].extend(E_hot)
                else:
                    mol_pred, mol_true = res  # only discrete tensors

                # first candidate → default prediction for classical metrics
                if c_idx == 0:
                    samples["pred"].extend(mol_pred)
                    samples["true"].extend(mol_true)
                # samples["n_all"] += len(batch_y) # TODO right?

            # 2‑d) meta‑info used by retrieval metrics
            for i, t in enumerate(batch_nmr):
                if samples["batch_condition"][i] is None:
                    samples["batch_condition"][i] = paddle.to_tensor(t)
                else:
                    samples["batch_condition"][i] = paddle.concat(
                        [samples["batch_condition"][i], paddle.to_tensor(t)], axis=0
                    )
            samples["node_mask_meta"].extend(batch_atomCount)
            samples["n_all"] += bs

            # 2‑e) Early‑stop check
            # We exit the loop once the number of processed mini‑batches reaches
            # `max_iters`, unless the caller has explicitly set `flag_sample=True`
            # to force a full‑dataset sweep.
            if (not flag_sample) and (iter_id + 1 >= max_iters):
                break

        # 3. Pass everything to SamplingMolecularMetrics (single call)
        metric_layer = (
            self.model._layers if isinstance(self.model, paddle.DataParallel)
            else self.model
        ).sampling_metrics

        metric_dict = metric_layer(
            samples,
            current_epoch = epoch_id,
            local_rank = self.rank,
            output_dir = self.output_dir,
            flag_test = True, # enable file saving
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
        sample_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        metric_class: Optional[Callable] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
    ):
        super().__init__(
            config,
            model,
            train_dataloader,
            val_dataloader,
            sample_dataloader,
            test_dataloader,
            optimizer,
            metric_class,
            lr_scheduler,
        )
