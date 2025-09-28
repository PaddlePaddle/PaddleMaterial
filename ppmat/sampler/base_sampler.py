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

import copy
import os
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from ppmat.datasets import build_dataloader
from ppmat.datasets import build_dataset_infos
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.msd_nmr_dataset import DataLoaderCollection
from ppmat.datasets.transform import build_post_transforms
from ppmat.metrics import DiffNMRStreamingAdapter
from ppmat.metrics import build_metric
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.models.diffnmr.extra_features_graph import DummyExtraFeatures
from ppmat.models.diffnmr.extra_features_graph import ExtraFeatures
from ppmat.models.diffnmr.extra_features_molecular_graph import ExtraMolecularFeatures
from ppmat.models.diffnmr.utils import diffgraphformer_utils
from ppmat.schedulers import scheduling_diffnmr
from ppmat.utils import logger
from ppmat.utils import save_load
from ppmat.utils.visualization import MolecularVisualization


class MolecularSampler:
    """Molecular Sampler.

    This class provides an interface for sampling structures using pre-trained deep
    learning models. Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        # if model_name is not None, then config_path and checkpoint_path must be
        # provided
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is "
            "None."

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."

            # TODO: optimize in the future
            set_signal_handlers()
            train_data_cfg = config["Dataset"].get("train")
            train_loader = build_dataloader(train_data_cfg)

            val_data_cfg = config["Dataset"].get("val")
            val_loader = build_dataloader(val_data_cfg)

            test_data_cfg = config["Dataset"].get("test")
            test_loader = build_dataloader(test_data_cfg)

            # build datasetinfo
            dataloaders = DataLoaderCollection(train_loader, val_loader, test_loader)
            dataset_infos = build_dataset_infos(
                dataloaders=dataloaders, cfg=config, recompute_statistics=False
            )
            train_smiles = dataset_infos.train_smiles

            # extra features
            if (
                config["Model"]["__init_params__"]["diffmodel_cfg"]["extra_features"]
                is not None
            ):
                extra_features = ExtraFeatures(
                    config["Model"]["__init_params__"]["diffmodel_cfg"][
                        "extra_features"
                    ],
                    dataset_infos=dataset_infos,
                )
                domain_features = ExtraMolecularFeatures(
                    dataset_infos=dataset_infos,
                )
            else:
                extra_features = DummyExtraFeatures()
                domain_features = DummyExtraFeatures()
            fallback_loader = train_loader or val_loader or test_loader
            dataset_infos.compute_input_output_dims(
                dataloader=fallback_loader,
                extra_features=extra_features,
                domain_features=domain_features,
                conditionDim=config["Model"]["__init_params__"]["diffmodel_cfg"][
                    "conditdim"
                ],
            )

            # CLIP for sample metric
            model_cfg = config["CLIP"]
            self.clip = build_model(
                model_cfg,
                extra_features=extra_features,
                domain_features=domain_features,
                dataset_infos=dataset_infos,
            )

            # visualization tools
            self.visualization_tools = MolecularVisualization(
                dataset_infos=dataset_infos,
                output_dir=config["Trainer"]["output_dir"],
            )

            model_cfg = config["Model"]
            model = build_model(
                model_cfg,
                extra_features=extra_features,
                domain_features=domain_features,
                dataset_infos=dataset_infos,
                visualization_tools=self.visualization_tools,
                clip=self.clip,
            )

            self.pretrained_model_path = (
                checkpoint_path
                if checkpoint_path is not None
                else config.get("pretrained_model_path", None)
            )
            self.pretrained_weight_name = (
                weights_name
                if weights_name is not None
                else config.get("pretrained_weight_name", None)
            )
            save_load.load_pretrain(
                model, self.pretrained_model_path, self.pretrained_weight_name
            )

        else:
            logger.info("Since model_name is given, downloading it...")
            model, config = build_model_from_name(model_name, weights_name)

        self.model = model
        self.config = config

        self.model.eval()

        # sample config
        sample_config = config.get("Sampler", None)
        self.sample_config = sample_config
        self.samp_per_val = sample_config["sample_every_val"]
        self.visual_num = sample_config["visual_num"]
        self.chains_left_to_save = sample_config["chains_to_save"]
        self.number_chain_steps = sample_config["number_chain_steps"]
        self.sample_batch_iters = sample_config["sample_batch_iters"]
        self.metric_dict_sample = sample_config.get("out_dict", None)
        self.flag_retrival_sampling = sample_config.get("flag_retrival_sampling", False)
        self.flag_use_formula = sample_config.get("flag_use_formula", False)
        self.flag_retrival_initilization = sample_config.get(
            "flag_retrival_initilization", False
        )
        self.num_candidates = sample_config.get("num_candidates", 1)

        self.post_transforms_cfg = self.sample_config.get("post_transforms", None)
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

        # runtime info
        self.rank = (
            int(paddle.distributed.get_rank())
            if paddle.distributed.is_initialized()
            else 0
        )
        self.output_dir = self.config.get("Sampler", {}).get("output_dir", "./outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        if self.clip is not None:
            setattr(self.model, "clip", self.clip)

        self.molecular_vectors, self.smiles_list = self._init_retrieval_bank(
            self.sample_config,
        )

        self.streaming = DiffNMRStreamingAdapter(
            t_scale=float(self.sample_config.get("t_scale", 1.0)),
            dataset_infos=dataset_infos,
        )
        self.streaming.bind(
            model=self.model,
            dataset_infos=dataset_infos,
            clip=self.clip,
            train_smiles=train_smiles,
            num_candidate=self.num_candidates,
        )
        setattr(self.model, "streaming_adapter", self.streaming)

    def compute_metric(
        self,
        save_path=None,
    ):
        self.output_dir = save_path if save_path is not None else self.output_dir
        metrics_cfg = self.sample_config.get("metrics")
        assert metrics_cfg is not None, "metrics config must be provided."
        metrics_fn = build_metric(metrics_cfg)

        total_results = self.sample_by_dataloader(
            self.output_dir,
        )

        metric = metrics_fn(total_results)
        return metric

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def sample(self, data, sample_params=None):
        if sample_params is None:
            sample_params = {}
        assert isinstance(sample_params, dict), "sample_params must be a dict or None."
        pred_data = self.model.sample(data, **sample_params)
        pred_data = self.post_process(pred_data)
        return pred_data

    def sample_by_dataloader(
        self,
        save_path=None,
    ):
        self.output_dir = save_path if save_path is not None else self.output_dir
        dataset_cfg = self.sample_config["data"]
        data_loader = build_dataloader(dataset_cfg)

        # build_molecule_cfg = self.sample_config["build_molecule_cfg"]
        # molecule_converter = BuildMolecule(**build_molecule_cfg)

        logger.info(f"Total iterations: {len(data_loader)}")
        logger.info("Start sampling process...")

        self.model.eval()
        epoch_id = 0

        data_length = len(data_loader)
        logger.message(f"Start to sample ... | Total Batches: {data_length}")
        start = time.time()

        # sample epoch
        metric_dict = self.sample_epoch(
            data_loader,
            epoch_id,
            keep_onehot=self.flag_retrival_sampling,
            num_candidates=self.num_candidates,
        )

        # log eval sample metric info
        if paddle.distributed.get_rank() == 0:
            msg = "Sample:"
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

    @paddle.no_grad()
    def sample_epoch(
        self,
        dataloader: paddle.io.DataLoader,
        epoch_id: int,
        num_candidates: int = 1,
        keep_onehot: bool = False,
    ):
        """Run **one full sampling pass** over ``dataloader`` and collect metrics.

        This wrapper repeatedly calls :func:`sample_batch` to generate *multiple*
        candidate molecules for every ground‑truth graph in the batch. The first
        candidate of each batch is treated as the *default prediction* used for
        classical metrics (validity, novelty, etc.). All *num_candidates* variants
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
        num_candidates : int, default 1
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
        • call :func:`sample_batch` ``num_candidates`` times.
        • aggregate predictions, ground‑truth and (optionally) one‑hot tensors.
        3. Early‑exit when ``iters_left`` hits zero
        4. Call the metric layer *once* – avoids repeated RDKit initialisation and
        keeps logging atomic.
        """
        # Put the model in eval‑mode so layers like dropout / batch‑norm are frozen
        self.model.eval()

        # used for early‑stopping a long dataloader when we only need a subset.
        max_iters: int = self.sample_batch_iters

        # 1. pre‑allocate the data structure that SamplingMolecularMetrics expects
        samples: Dict[str, Union[list, int]] = {
            "pred": [],  # first candidate of each ground‑truth
            "true": [],  # ground‑truth graphs
            "n_all": 0,  # total number of GT molecules processed
            "node_mask_meta": [],  # node mask metadata for each batch
            "batch_condition": [],  # 4‑branch NMR condition
            "dict": (
                self.model._layers
                if isinstance(self.model, paddle.DataParallel)
                else self.model
            ).dataset_info.atom_decoder,  # id → element symbol
        }
        if keep_onehot:
            # For retrieval metrics we need to keep *all* candidates & their one‑hot
            samples["candidates"] = [[] for _ in range(num_candidates)]
            samples["candidates_X"] = [[] for _ in range(num_candidates)]
            samples["candidates_E"] = [[] for _ in range(num_candidates)]

        # 2. main loop over DataLoader
        for iter_id, batch_data in enumerate(dataloader):
            batch_graph = batch_data["graph"]
            batch_property = batch_data["property"]
            batch_spectrum = batch_data["spectrum"]

            # 2.a convert sparse graph to dense (one‑hot padded) representation
            dense_data, node_mask = diffgraphformer_utils.to_dense(
                paddle.to_tensor(batch_graph.node_feat["feat"]),
                paddle.to_tensor(batch_graph.edges.T),
                paddle.to_tensor(batch_graph.edge_feat["feat"]),
                paddle.to_tensor(batch_graph.graph_node_id),
            )
            dense_data = dense_data.mask(node_mask)  # remove padding rows

            # basic batch tensors
            batch_atomCount = paddle.to_tensor(
                batch_property["atom_count"]
            )  # [B] number of atoms
            batch_y = paddle.to_tensor(batch_property["y"])  # labels (unused here)
            batch_X, batch_E = dense_data.X, dense_data.E  # one‑hot Node / Edge
            bs = len(batch_y)  # batch size

            # 2.b build four‑branch NMR condition tensor list
            if hasattr(self.model, "seq_len_H1"):
                cond_H = paddle.to_tensor(batch_spectrum["H_nmr"])
                cond_C = paddle.to_tensor(batch_spectrum["C_nmr"])
                num_H_peak = paddle.to_tensor(batch_spectrum["num_H_peak"])
                num_C_peak = paddle.to_tensor(batch_spectrum["num_C_peak"])
                batch_nmr = [cond_H, num_H_peak, cond_C, num_C_peak]
            else:
                batch_nmr = None  # TODO: re‑implement for single‑branch condition

            # 2.c call `sample_batch` `num_candidates` times
            for c_idx in range(num_candidates):
                kwargs = dict(
                    model=self.model._layers
                    if isinstance(self.model, paddle.DataParallel)
                    else self.model,
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
                    iter_idx=c_idx,
                )
                if self.flag_retrival_initilization:
                    kwargs.update(
                        retrival_initilization=self.flag_retrival_initilization,
                        clip=self.clip,
                        molecular_vectors=self.molecular_vectors,
                        smiles_list=self.smiles_list,
                    )
                res = self.sample_batch(**kwargs)

                if keep_onehot:
                    mol_pred, mol_true, X_hot, E_hot = res
                    samples["candidates"][c_idx].extend(mol_pred)
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
            if batch_nmr is not None:
                samples["batch_condition"] = [None for _ in range(4)]
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
            # `max_iters`.
            if iter_id + 1 >= max_iters:
                break

        # 3. Pass everything to SamplingMolecularMetrics (single call)
        self.streaming.update_step(
            result={
                "samples": samples,
                "epoch_id": epoch_id,
                "local_rank": self.rank,
                "output_dir": self.output_dir,
            },
            batch=None,
            stage="sample",
        )
        metric_dict = self.streaming.compute_epoch(stage="sample")

        return metric_dict

    @paddle.no_grad()
    def sample_batch(
        self,
        model,
        batch_id: int,
        batch_size: int,
        batch_condition: List[paddle.Tensor],
        number_chain_steps: int,
        keep_chain: int,
        visual_num: int,
        batch_X: paddle.Tensor,
        batch_E: paddle.Tensor,
        batch_y: paddle.Tensor,
        iter_idx: int,
        num_nodes: Union[int, paddle.Tensor] = None,
        flag_useformula: bool = False,
        return_onehot: bool = False,
        retrival_initilization: bool = False,
        clip: paddle.nn.Layer = None,
        molecular_vectors: paddle.Tensor = None,
        smiles_list: List = None,
    ) -> Union[Tuple[List, List], Tuple[List, List, paddle.Tensor, paddle.Tensor],]:
        """Reverse–diffusion sampling in **Paddle dynamic graph**.

        Parameters
        ----------
        model : DiffusionModelLike
            The generator. Must expose attributes `T`, `node_dist`, `limit_dist`,
            and method `sample_p_zs_given_zt`.
        batch_id : int
            Index of the current batch – used only for logging / visualisation.
        batch_size : int
            Number of graphs to sample in this call.
        batch_condition : list[paddle.Tensor]
            Four‑branch conditioning vector (¹H‑NMR, ¹H peaks, ¹³C‑NMR, ¹³C peaks).
        number_chain_steps : int
            How many intermediate frames to keep for visualisation.
        keep_chain : int
            Number of graph chains to retain (B‑dim truncation).
        visual_num : int
            Number of final samples to render via `visualization_tools`.
        batch_X / batch_E : paddle.Tensor
            One‑hot ground‑truth node / edge feature tensors (used for guidance or
            as *oracle formula* when ``flag_useformula`` is True).
        batch_y : paddle.Tensor
            Additional labels (if any) required by the model.
        iter_idx : int
            Current iteration index for obtain candidates for retrival.
        num_nodes : int | paddle.Tensor | None
            Number of nodes per graph. When *None* the model samples from its own
            learned distribution.
        flag_useformula : bool
            If *True* force the sampled node features to exactly equal the
            provided one‑hot `batch_X` (for strict formula reconstruction).
        return_onehot : bool
            Whether to return the *padded* one‑hot tensors (`X_hot`, `E_hot`) in
            addition to discrete index lists – required by molVec retrieval.
        retrival_initilization : bool, default False
            Whether to enable **retrieval‑based initialization**.
            If True, the model will fetch the closest reference molecules
            (using `molecular_vectors`) and use them as the first step of
            the diffusion / sampling chain instead of pure noise.
        clip : paddle.nn.Layer | None, default None
            optional projection/clipping layer applied to latent features before
            retrieval.
        molecular_vectors : paddle.Tensor | None, default None
            2‑D tensor [N, D] containing embeddings of the reference molecule library
        smiles_list : list[str] | None, default None
            list of SMILES strings corresponding to those reference embeddings

        Returns
        -------
        If ``return_onehot`` is **False** (default):
            (molecule_list, molecule_list_true)
        If ``return_onehot`` is **True**:
            (molecule_list, molecule_list_true, X_hot, E_hot)

        Where
        ``molecule_list[i] == [atom_index_vector, bond_matrix]`` and
        ``molecule_list_true`` follows the same structure for ground‑truth.
        """

        # 1. Determine node counts and create a boolean mask for padded positions
        if num_nodes is None:
            # Sample number of nodes from the model's learned distribution
            n_nodes = model.node_dist.sample_n(batch_size)
        elif isinstance(num_nodes, int):
            n_nodes = paddle.full([batch_size], num_nodes, dtype="int64")
        else:
            n_nodes = paddle.to_tensor(num_nodes)  # assume Tensor

        n_max: int = int(paddle.max(n_nodes).item())  # ***largest graph size***

        # `node_mask[b, i] == True` if node *i* is real for graph *b*
        arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
        node_mask = arange < n_nodes.unsqueeze(1)

        # 2. Initialise z_T with (categorical) noise and prepare trajectory buffers
        # z(n_samples, n_nodes, n_features)
        z_T = scheduling_diffnmr.sample_discrete_feature_noise(
            limit_dist=model.limit_dist, node_mask=node_mask
        )
        X_t, E_t, y_t = z_T.X, z_T.E, z_T.y

        chain_X = paddle.zeros([number_chain_steps, keep_chain, n_max], dtype="int64")
        chain_E = paddle.zeros(
            [number_chain_steps, keep_chain, n_max, n_max], dtype="int64"
        )

        # 3. Retrieval Initialization(Optional)
        if retrival_initilization and batch_condition is not None:
            logger.info("Sampling Initializing using Retrieval Method.")
            output = clip.spectrum_encoder(batch_condition)

            similarities = self._batched_cosine_similarity(
                output, molecular_vectors, 128
            )
            top_k = 1
            top_k_values, top_k_indices = paddle.topk(similarities, k=top_k, axis=1)

            if iter_idx == 0:
                cols = 8
                lines = []
                for start in range(0, len(top_k_values), cols):
                    chunk = top_k_values[start : start + cols]
                    line = " | ".join(
                        f"{start+j} : {v.item():.3f}" for j, v in enumerate(chunk)
                    )
                    lines.append(line)
                logger.info(
                    "Highest Similarities (SampleID:Value)\n" + "\n".join(lines)
                )

            result_smiles = []
            for i in range(batch_size):
                idx = top_k_indices[i].item()
                result_smiles.append(smiles_list[idx])

            node_list = []
            adj_matrix_list = []

            for i in range(batch_size):
                smiles = result_smiles[i]
                current_node_mask = node_mask[i]
                node_tensor_onehot, adjacency_matrix_onehot = graphs_from_mol(
                    smiles, current_node_mask, i, X_t, E_t
                )
                node_list.append(node_tensor_onehot)
                adj_matrix_list.append(adjacency_matrix_onehot)

            X_t = paddle.stack(node_list, axis=0)
            E_t = paddle.stack(adj_matrix_list, axis=0)

            assert (E_t == paddle.transpose(E_t, [0, 2, 1])).all()
            assert number_chain_steps < model.T
        else:
            logger.info("Start Initializing using Random Method.")

        # 4. Main reverse‑diffusion loop: t = T → 1 (s = t‑1)
        for s_int in tqdm(
            range(model.T - 1, -1, -1),
            desc=f"Batch {batch_id} RepeatIter {iter_idx} sampling {model.T}→0",
            unit="step",
        ):
            s_arr = paddle.full([batch_size, 1], float(s_int))
            t_arr = s_arr + 1.0
            s_norm, t_norm = s_arr / model.T, t_arr / model.T

            # One reverse‑diffusion step
            sampled_s, discrete_sampled_s = scheduling_diffnmr.step(
                model,
                s=s_norm,
                t=t_norm,
                X_t=X_t,
                E_t=E_t,
                y_t=y_t,
                node_mask=node_mask,
                conditionVec=batch_condition,
                batch_X=batch_X,
                batch_E=batch_E,
                batch_y=batch_y,
            )
            X_t, E_t, y_t = sampled_s.X, sampled_s.E, sampled_s.y
            if flag_useformula is True:
                # Force atom types to match the provided formula (oracle guidance)
                X_t = batch_X

            # save intermediate frames for the first `keep_chain` graphs
            write_index = (s_int * number_chain_steps) // model.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # 5. Collapse padding → obtain discrete indices; optionally keep one‑hot
        # Make a *clone* of `sampled_s` so that collapsing will not overwrite the
        # one‑hot information we still need for molVec retrieval.
        sampled_copy = copy.deepcopy(sampled_s)

        # 5‑a. Get discrete indices from the cloned tensor (padding removed)
        sampled_collapse = sampled_copy.mask(node_mask, collapse=True)
        X_idx, E_idx = sampled_collapse.X, sampled_collapse.E  # [B, …]
        if flag_useformula:
            # Ensure indices follow the oracle molecular formula when required
            X_idx = paddle.argmax(batch_X, axis=-1)

        # 5‑b. Optionally obtain **un‑collapsed** one‑hot tensors for retrieval.
        if return_onehot:
            # Call mask *without* collapse on the ORIGINAL `sampled_s`, which still
            # contains one‑hot embeddings; shape stays [B, n_max, feat]
            X_hot = sampled_s.mask(node_mask).X.numpy()
            E_hot = sampled_s.mask(node_mask).E.numpy()
            if flag_useformula:
                # When formula guidance is enabled, the node one‑hot should exactly
                # match the provided ground‑truth.
                X_hot = batch_X.numpy()
            X_hot = [X_hot[i] for i in range(X_hot.shape[0])]
            E_hot = [E_hot[i] for i in range(E_hot.shape[0])]
        else:
            X_hot = E_hot = None

        # 6. Assemble Python lists for downstream RDKit / metrics
        mol_list, mol_true = [], []
        n_nodes_np = n_nodes.numpy()
        batch_X_idx = paddle.argmax(batch_X, axis=-1).numpy()
        batch_E_idx = paddle.argmax(batch_E, axis=-1).numpy()
        for i in range(batch_size):
            n = n_nodes_np[i]
            mol_list.append(
                [
                    X_idx[i, :n].numpy(),
                    E_idx[i, :n, :n].numpy(),
                ]
            )
            mol_true.append(
                [
                    batch_X_idx[i, :n],
                    batch_E_idx[i, :n, :n],
                ]
            )

        # 7. Optional visualisation via model.visualization_tools
        if self.visualization_tools is not None:
            # 7.a Prepare the chain for visualization and saving
            if keep_chain > 0:
                # pick the last frame of the chain add the top index of chain_X/E(index
                # 0)
                final_X_chain = X_idx[:keep_chain]
                final_E_chain = E_idx[:keep_chain]
                chain_X[
                    0
                ] = final_X_chain  # Overwrite last frame with the resulting X, E
                chain_E[0] = final_E_chain

                # revers time sequence for visualization
                chain_X = scheduling_diffnmr.reverse_tensor(chain_X)
                chain_E = scheduling_diffnmr.reverse_tensor(chain_E)

                # Repeat last frame to see final sample better
                chain_X = paddle.concat(
                    [chain_X, chain_X[-1:].tile([10, 1, 1])], axis=0
                )
                chain_E = paddle.concat(
                    [chain_E, chain_E[-1:].tile([10, 1, 1, 1])], axis=0
                )
                assert chain_X.shape[0] == (number_chain_steps + 10)

            # 7.b use visulize tools
            num_mols = chain_X.shape[1]
            # draw animation of diffusion process of generated molecules
            for i in range(num_mols):
                chain_X_np = chain_X[:, i, :].numpy()
                chain_E_np = chain_E[:, i, :, :].numpy()
                self.visualization_tools.visualize_chain(
                    batch_id, i, chain_X_np, chain_E_np
                )
            # draw picture of predicted and true molecules
            self.visualization_tools.visualizeNmr(
                batch_id,
                mol_list,
                mol_true,
                visual_num,
            )

        if return_onehot:
            return mol_list, mol_true, X_hot, E_hot
        return mol_list, mol_true

    def _init_retrieval_bank(self, cfg):
        """
        load the molecular vector library for retrieval initialization/evaluation
        from configuration
        """
        if not cfg:
            return None, None
        path = cfg.get("retrival_database_path", None)
        if path is None or not os.path.exists(path):
            logger.warning(f"[retrieval_bank] path missing or not found: {path}")
            return None, None

        ext = os.path.splitext(path)[1].lower()
        embs, smiles = None, None
        try:
            if ext == ".csv":
                data = pd.read_csv(path)
                data["molecularRep"] = data["molecularRep"].apply(
                    lambda x: np.fromstring(x.strip("[]"), sep=" ")
                )
                # to paddle tensor
                embs = paddle.to_tensor(
                    np.stack(data["molecularRep"].values), dtype="float32"
                )
                smiles = data["smiles"].tolist()
            else:
                raise ValueError(f"Unsupported retrieval bank ext: {ext}")
        except Exception as e:
            logger.warning(f"[retrieval_bank] load failed: {e}")
            return None, None

        return embs, smiles

    def _batched_cosine_similarity(self, output, molecular_vectors, batch_size_cut):
        similarities = []
        for i in range(0, molecular_vectors.shape[0], batch_size_cut):
            batch_vectors = molecular_vectors[i : i + batch_size_cut]
            sim = F.cosine_similarity(
                output.unsqueeze(1), batch_vectors.unsqueeze(0), axis=-1
            )  # [batch_size, batch_size_cut]
            similarities.append(sim)
        return paddle.concat(similarities, axis=1)  # [batch_size, N]


def graphs_from_mol(smiles, node_mask, i, X, E):
    """
    Convert an SMILES string into graph presentation (node features & adjacency).

    Parameters
    ----------
    smiles : str
        The molecule in SMILES format.

    node_mask : paddle.Tensor, shape [max_nodes]
        Boolean / 0‑1 mask telling how many node slots are valid
        for *this* molecule inside the batch tensor.

    i : int
        Index of the current sample in the mini‑batch (1st dimension of X/E).

    X : paddle.Tensor, shape [B, max_nodes, n_atom_types]
        Pre‑allocated batch buffer for node one‑hot features.
        Will be updated in‑place at index `i`.

    E : paddle.Tensor, shape [B, max_nodes, max_nodes, n_bond_types]
        Pre‑allocated batch buffer for adjacency one‑hot tensors.
        Will be updated in‑place at index `i`.


    Returns:
        node_list: A one-hot encoded list representing atom types.
        adjacency_matrix: A 3D numpy array (one-hot encoded) representing the adjacency
            matrix of the molecule.
    """
    from rdkit import Chem

    num_trueAtoms = paddle.sum(node_mask)

    # dictionary to map atom symbols to integer values
    atom_encoder = {
        "C": 0,
        "N": 1,
        "O": 2,
        "F": 3,
        "P": 4,
        "S": 5,
        "Cl": 6,
        "Br": 7,
        "I": 8,
    }
    atom_encoder_len = len(atom_encoder)  # Number of distinct atom types
    # print(f'graphs_from_mol_smiles{smiles}')
    # initialize the node list
    node_list = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES or parsing failed: {smiles}")
        return X[i], E[i]

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # print(f'symbol{symbol}')
        if symbol in atom_encoder:
            node_list.append(atom_encoder[symbol])
        else:
            raise ValueError(f"Atom symbol {symbol} not in atom_encoder")

    # initialize adjacency matrix
    num_atoms = len(node_list)
    node_tensor = paddle.to_tensor(node_list, dtype="int64")
    node_mask_len = node_mask.shape[0]
    padding = paddle.full((node_mask_len - num_atoms,), fill_value=-1, dtype="int64")
    node_tensor = paddle.concat((node_tensor, padding))
    num_atoms_max = len(node_tensor)

    # Convert node_tensor to one-hot
    node_tensor_onehot = F.one_hot(
        node_tensor.clip(min=0), num_classes=atom_encoder_len
    ).astype(
        "float32"
    )  # Ignore -1 for num_classes
    node_tensor_onehot[node_tensor == -1] = 0  # Set -1 positions to all-zero vectors
    if num_atoms >= num_trueAtoms:
        X[i][:num_trueAtoms] = node_tensor_onehot[:num_trueAtoms]
        node_tensor_onehot = X[i]
    else:
        X[i][:num_atoms] = node_tensor_onehot[:num_atoms]
        node_tensor_onehot = X[i]

    adjacency_matrix = np.full((num_atoms_max, num_atoms_max), -1, dtype="int")
    adjacency_matrix[:num_atoms, :num_atoms] = 0

    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # determine bond type
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_value = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_value = 2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_value = 3
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_value = 4
        else:
            bond_value = 0

        # populate adjacency matrix (symmetric)
        adjacency_matrix[start_idx, end_idx] = bond_value
        adjacency_matrix[end_idx, start_idx] = bond_value

    # Convert adjacency_matrix to one-hot
    max_bond_type = 4  # Maximum bond type value (single, double, triple, aromatic)
    adjacency_matrix_tensor = paddle.to_tensor(adjacency_matrix, dtype="int64")
    adjacency_matrix_onehot = F.one_hot(
        adjacency_matrix_tensor.clip(min=0), num_classes=max_bond_type + 1
    ).astype("float32")
    adjacency_matrix_onehot[
        adjacency_matrix_tensor == -1
    ] = 0  # Set -1 positions to all-zero vectors

    if num_atoms >= num_trueAtoms:
        E[i][:num_trueAtoms, :num_trueAtoms] = adjacency_matrix_onehot[
            :num_trueAtoms, :num_trueAtoms
        ]
        adjacency_matrix_onehot = E[i]
    else:
        E[i][:num_atoms, :num_atoms] = adjacency_matrix_onehot[:num_atoms, :num_atoms]
        adjacency_matrix_onehot = E[i]

    return node_tensor_onehot, adjacency_matrix_onehot
