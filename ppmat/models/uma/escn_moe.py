from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools
import logging
import warnings
from contextlib import nullcontext

import numpy as np
import paddle
from matplotlib import pyplot as plt

from ._compat.registry import registry
from ._compat.utils import conditional_grad
from ._compat.base import HeadInterface
from .escn_md import eSCNMDBackbone, resolve_dataset_mapping
from .nn.mole import MOLE, MOLEGlobals
from .nn.mole_utils import (
    MOLEInterface,
    convert_model_to_MOLE_model,
    model_search_and_replace,
    recursive_replace_all_linear,
    recursive_replace_so2_MOLE,
    replace_linear_with_MOLE,
    replace_MOLE_with_linear,
)

warnings.filterwarnings(
    "ignore", message=r"index_reduce\(\) is in beta", category=UserWarning
)


def _no_autocast():
    return nullcontext()


def _scatter_add_1d(index: paddle.Tensor, size: int, dtype: paddle.dtype) -> paddle.Tensor:
    out = index.new_zeros([size], dtype=dtype)
    ones = index.new_ones([index.shape[0]], dtype=dtype)
    out.index_add_(0, index.astype("int64"), ones)
    return out


def _one_hot(indices: paddle.Tensor, num_classes: int, dtype: paddle.dtype) -> paddle.Tensor:
    return paddle.nn.functional.one_hot(indices.astype("int64"), num_classes).astype(dtype)


@registry.register_model("escnmd_moe_backbone")
class eSCNMDMoeBackbone(eSCNMDBackbone, MOLEInterface):
    def __init__(
        self,
        num_experts: int = 8,
        moe_dropout: float = 0.0,
        use_global_embedding: bool = False,
        use_composition_embedding: bool = False,
        composition_dropout: float = 0.0,
        moe_expert_coefficient_norm: str = "softmax",
        act=paddle.nn.SiLU,
        layers_moe=None,
        moe_layer_type: str = "pytorch",
        moe_single: bool = False,
        moe_type: str = "so2",
        model_version: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parent_kwargs = kwargs
        self.num_experts = num_experts
        self.model_version = model_version
        if num_experts > 0:
            convert_model_to_MOLE_model(
                model=self,
                num_experts=num_experts,
                mole_dropout=moe_dropout,
                mole_expert_coefficient_norm=moe_expert_coefficient_norm,
                act=act,
                layers_mole=layers_moe,
                use_composition_embedding=use_composition_embedding,
                composition_dropout=composition_dropout,
                mole_layer_type=moe_layer_type,
                mole_single=moe_single,
                mole_type=moe_type,
            )

    def merge_MOLE_model(self, data):
        if self.num_experts == 0:
            return self
        csd_mixed_emb = self.csd_embedding(
            charge=data["charge"], spin=data["spin"], dataset=data["dataset"]
        )
        self.set_MOLE_coefficients(
            atomic_numbers_full=data["atomic_numbers"],
            batch_full=data["batch"],
            csd_mixed_emb=csd_mixed_emb,
        )
        if self.mole_type != "so2":
            raise ValueError("Only mole_type=so2 supported for merging")
        model_search_and_replace(
            self, recursive_replace_so2_MOLE, replace_MOLE_with_linear
        )
        self.routing_mlp = None
        self.composition_embedding = None
        self.num_experts = 0
        new_model = eSCNMDBackbone(**self.parent_kwargs)
        new_model.load_state_dict(self.state_dict())
        new_model.eval()
        return new_model

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
        if self.num_experts == 0:
            return
        with _no_autocast():
            embeddings = []
            if self.use_composition_embedding:
                effective_atomic_numbers_full = atomic_numbers_full
                effective_batch_full = batch_full

                if self.training and self.composition_dropout > 0.0:
                    mask = (
                        paddle.rand_like(atomic_numbers_full, dtype=paddle.float32)
                        > self.composition_dropout
                    )
                    effective_atomic_numbers_full = atomic_numbers_full[mask]
                    effective_batch_full = batch_full[mask]

                composition_by_atom = self.composition_embedding(
                    effective_atomic_numbers_full
                )

                composition = composition_by_atom.new_zeros(
                    [csd_mixed_emb.shape[0], self.sphere_channels],
                    dtype=composition_by_atom.dtype,
                )
                composition.index_add_(
                    0,
                    effective_batch_full.astype("int64"),
                    composition_by_atom,
                )

                counts = composition_by_atom.new_zeros(
                    [csd_mixed_emb.shape[0]], dtype=composition_by_atom.dtype
                )
                counts.index_add_(
                    0,
                    effective_batch_full.astype("int64"),
                    composition_by_atom.new_ones(
                        [effective_batch_full.shape[0]], dtype=composition_by_atom.dtype
                    ),
                )
                if np.isclose(self.model_version, 1.0).item():
                    counts = counts + 1.0
                counts = paddle.clip(counts, min=1.0)
                composition = composition / counts.unsqueeze(1)

                embeddings.append(composition.unsqueeze(0))

            embeddings.append(csd_mixed_emb.unsqueeze(0))
            expert_mixing_coefficients_before_norm = self.routing_mlp(
                paddle.vstack(embeddings)
                .transpose([1, 0, 2])
                .reshape([csd_mixed_emb.shape[0], -1])
            )
            self.global_mole_tensors.expert_mixing_coefficients = (
                self.mole_expert_coefficient_norm(
                    self.mole_dropout(expert_mixing_coefficients_before_norm)
                )
            )

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index):
        if self.num_experts == 0:
            return
        with _no_autocast():
            target_batch = batch_full[edge_index[1]]
            mole_sizes = _scatter_add_1d(
                target_batch,
                nsystems,
                dtype=paddle.int32,
            )
            self.global_mole_tensors.mole_sizes = mole_sizes.cpu()

    def log_MOLE_stats(self):
        if not self.training or self.num_experts == 0:
            return
        if not hasattr(self, "fig"):
            self.fig, self.axs = plt.subplots(2, 1)
        with paddle.no_grad():
            if self.counter % 500 == 0:
                logging.info(
                    f"{self.counter}: Expert variance: "
                    + ",".join(
                        [
                            f"{x:.2e}"
                            for x in self.global_mole_tensors.expert_mixing_coefficients.var(
                                axis=0
                            ).tolist()
                        ]
                    )
                )
                logging.info(
                    f"{self.counter}: Expert mean: "
                    + ",".join(
                        [
                            f"{x:.2e}"
                            for x in self.global_mole_tensors.expert_mixing_coefficients.mean(
                                axis=0
                            ).tolist()
                        ]
                    )
                )
                self.fig.tight_layout()
                self.plot_ready = True
        self.counter += 1


class DatasetSpecificMoEWrapper(paddle.nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        head_cls,
        wrap_property=True,
        head_kwargs=None,
        dataset_names: (list[str] | None) = None,
        dataset_mapping: (dict[str, str] | None) = None,
    ):
        """
        Initialize the DatasetSpecificMoEWrapper.

        Args:
            backbone: The backbone model providing embeddings.
            head_cls: Registry name of the head class to instantiate.
            wrap_property: If True, wrap output tensors in a dict with the key name.
            head_kwargs: Additional keyword arguments passed to the head constructor.
            dataset_names: Deprecated. Use dataset_mapping instead.
            dataset_mapping: A mapping from dataset names to output head identifiers.
                Allows multiple datasets to share the same head/expert by mapping
                them to the same identifier. Example:
                {"omol": "omol", "omat": "omat", "oc20": "oc20", "oc20_subset": "oc20"}
                Here omol and omat have their own heads while oc20 and oc20_subset
                share the same oc20 head. Dict values must be a subset of dict keys.
        """
        super().__init__()
        if head_kwargs is None:
            head_kwargs = {}
        self.regress_config = backbone.regress_config
        self.wrap_property = wrap_property
        self.dataset_names, self.dataset_name_to_exp = self._build_expert_mapping(
            dataset_names, dataset_mapping
        )
        self.head = registry.get_model_class(head_cls)(backbone, **head_kwargs)
        self.global_mole_tensors = MOLEGlobals(
            expert_mixing_coefficients=None, mole_sizes=None
        )
        replacement_factory = functools.partial(
            replace_linear_with_MOLE,
            num_experts=len(self.dataset_names),
            global_mole_tensors=self.global_mole_tensors,
            mole_layer_type="pytorch",
            cache=None,
        )
        recursive_replace_all_linear(self.head, replacement_factory)
        self.merged_on_dataset = None
        self.non_merged_dataset_names: list[str] = []

    @property
    def regress_forces(self) -> bool:
        return self.regress_config.forces

    @property
    def regress_stress(self) -> bool:
        return self.regress_config.stress

    @staticmethod
    def _build_expert_mapping(
        dataset_names: (list[str] | None), dataset_mapping: (dict[str, str] | None)
    ) -> tuple[list[str], dict[str, int]]:
        """
        Build the dataset-to-expert-index mapping.

        Args:
            dataset_names: Deprecated list of dataset names.
            dataset_mapping: Dict mapping dataset names to head identifiers.

        Returns:
            A tuple of (sorted dataset names list, dict mapping names to expert indices).
        """
        dataset_mapping = resolve_dataset_mapping(
            dataset_names, dataset_mapping, "dataset_names"
        )
        sorted_names = sorted(dataset_mapping.keys())
        unique_targets = sorted(set(dataset_mapping.values()))
        name_to_exp = {
            name: unique_targets.index(dataset_mapping[name]) for name in sorted_names
        }
        return sorted_names, name_to_exp

    def merge_MOLE_model(self, data):
        """
        Merge MOLE layers into single Linear for single-dataset inference.

        Sets one-hot expert coefficients and replaces all MOLE→Linear.
        """
        self.merged_on_dataset = data.dataset[0]
        expert_idx = self.dataset_name_to_exp[self.merged_on_dataset]
        expert_indices = paddle.to_tensor([expert_idx], dtype="int64", place=data.pos.place)
        self.global_mole_tensors.expert_mixing_coefficients = _one_hot(
            expert_indices,
            len(self.dataset_name_to_exp),
            data.pos.dtype,
        )

        def replace_mole(module):
            for name, child in list(module.named_children()):
                if isinstance(child, MOLE):
                    setattr(module, name, child.merged_linear_layer())
                else:
                    replace_mole(child)

        replace_mole(self.head)
        self.non_merged_dataset_names = [
            n for n in self.dataset_names if n != self.merged_on_dataset
        ]
        return self

    def prepare_for_inference(self, data, settings):
        """
        Prepare head for inference. Handles MOLE merging if needed.
        """
        if settings.merge_mole:
            return self.merge_MOLE_model(data)
        return self

    @conditional_grad(paddle.enable_grad())
    def forward(self, data, emb: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        if self.merged_on_dataset is not None:
            head_output = self.head(data, emb)
            full_output = {}
            for key in head_output:
                full_output[f"{self.merged_on_dataset}_{key}"] = (
                    {key: head_output[key]} if self.wrap_property else head_output[key]
                )
                nan_tensor = head_output[key].new_full(
                    head_output[key].shape, float("nan")
                )
                for dataset in self.non_merged_dataset_names:
                    full_output[f"{dataset}_{key}"] = (
                        {key: nan_tensor} if self.wrap_property else nan_tensor
                    )
            return full_output
        batch_index = emb["batch"].astype("int64")
        self.global_mole_tensors.mole_sizes = _scatter_add_1d(
            batch_index,
            int(data.natoms.shape[0]),
            dtype=paddle.int32,
        )
        self.global_mole_tensors.natoms = emb["batch"].shape[0]
        data_batch_full = data.batch_full.cpu()

        expert_indices = paddle.to_tensor(
            [self.dataset_name_to_exp[dataset_name] for dataset_name in data.dataset],
            dtype="int64",
            place=data.pos.place,
        )
        self.global_mole_tensors.expert_mixing_coefficients = _one_hot(
            expert_indices,
            len(self.dataset_name_to_exp),
            data.pos.dtype,
        )
        head_output = self.head(data, emb)
        np_dataset_names = np.array(data.dataset)
        full_output = {}
        for dataset_name in self.dataset_names:
            dataset_mask = np_dataset_names == dataset_name
            for key, mole_output_tensor in head_output.items():
                output_tensor = mole_output_tensor.new_zeros(mole_output_tensor.shape)
                if dataset_mask.any():
                    if output_tensor.shape[0] == dataset_mask.shape[0]:
                        output_tensor[dataset_mask] = mole_output_tensor[dataset_mask]
                    else:
                        atoms_mask = paddle.isin(
                            data_batch_full,
                            paddle.where(paddle.from_numpy(dataset_mask))[0],
                        )
                        output_tensor[atoms_mask] = mole_output_tensor[atoms_mask]
                full_output[f"{dataset_name}_{key}"] = (
                    {key: output_tensor} if self.wrap_property else output_tensor
                )
        return full_output


class DatasetSpecificSingleHeadWrapper(paddle.nn.Module, HeadInterface):
    def __init__(
        self, backbone, dataset_names, head_cls, wrap_property=True, head_kwargs=None
    ):
        super().__init__()
        if head_kwargs is None:
            head_kwargs = {}
        self.regress_config = backbone.regress_config
        self.wrap_property = wrap_property
        self.dataset_names = sorted(dataset_names)
        self.head = registry.get_model_class(head_cls)(backbone, **head_kwargs)
        self.merged_on_dataset = None

    @property
    def regress_forces(self) -> bool:
        return self.regress_config.forces

    @property
    def regress_stress(self) -> bool:
        return self.regress_config.stress

    def merge_MOLE_model(self, data):
        self.merged_on_dataset = data.dataset[0]
        self.non_merged_dataset_names = [
            name for name in self.dataset_names if name != self.merged_on_dataset
        ]
        return self

    def prepare_for_inference(self, data, settings):
        """
        Prepare head for inference. Handles MOLE merging if needed.
        """
        if settings.merge_mole:
            return self.merge_MOLE_model(data)
        return self

    @conditional_grad(paddle.enable_grad())
    def forward(self, data, emb: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        data_batch_full = data.batch_full.cpu()
        head_output = self.head(data, emb)
        if self.merged_on_dataset is not None:
            full_output = {}
            for key in head_output:
                full_output[f"{self.merged_on_dataset}_{key}"] = (
                    {key: head_output[key]} if self.wrap_property else head_output[key]
                )
                nan_tensor = head_output[key].new_full(
                    head_output[key].shape, float("nan")
                )
                for dataset in self.non_merged_dataset_names:
                    full_output[f"{dataset}_{key}"] = (
                        {key: nan_tensor} if self.wrap_property else nan_tensor
                    )
            return full_output
        assert set(data.dataset) <= set(
            self.dataset_names
        ), f"Input dataset names: {set(data.dataset)} must be a strict subset of model's valid datset names: {set(self.dataset_names)} "
        np_dataset_names = np.array(data.dataset)
        full_output = {}
        for dataset_name in self.dataset_names:
            dataset_mask = np_dataset_names == dataset_name
            for key, head_output_tensor in head_output.items():
                output_tensor = head_output_tensor.new_zeros(head_output_tensor.shape)
                if dataset_mask.any():
                    if output_tensor.shape[0] == dataset_mask.shape[0]:
                        output_tensor[dataset_mask] = head_output_tensor[dataset_mask]
                    else:
                        atoms_mask = paddle.isin(
                            data_batch_full,
                            paddle.where(paddle.from_numpy(dataset_mask))[0],
                        )
                        output_tensor[atoms_mask] = head_output_tensor[atoms_mask]
                full_output[f"{dataset_name}_{key}"] = (
                    {key: output_tensor} if self.wrap_property else output_tensor
                )
        return full_output
