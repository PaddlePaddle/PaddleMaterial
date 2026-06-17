# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import paddle

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from .unified_radial import UnifiedRadialMLP

if TYPE_CHECKING:
    from .._compat.inference import InferenceSettings
__all__ = [
    "ExecutionMode",
    "ExecutionBackend",
    "UMASFastPytorchBackend",
    "UMASFastGPUBackend",
    "get_execution_backend",
    "maybe_update_settings_backend",
]
_M0_COL_INDICES_L_ORDER = [0, 2, 6]


class ExecutionMode(str, Enum):
    """
    Execution mode for model inference.
    """

    GENERAL = "general"
    UMAS_FAST_PYTORCH = "umas_fast_pytorch"
    UMAS_FAST_GPU = "umas_fast_gpu"


class ExecutionBackend:
    """
    Parameterless function dispatch for execution modes.

    Provides default PyTorch implementations for rotation and scatter
    operations. Subclass and override methods with optimized kernels
    (e.g. Triton) for specific execution modes.

    All methods are static — backends carry no instance state.

    Methods (override for optimization):
        - node_to_edge_wigner_permute: Gather node features and rotate L->M
        - permute_wigner_inv_edge_to_node: Rotate M->L and scatter to nodes
        - edge_degree_scatter: Rotate radial and scatter to nodes
        - prepare_model_for_inference: Apply backend-specific model transforms
    """

    @staticmethod
    def validate(lmax: int, mmax: int, settings: InferenceSettings) -> None:
        """
        Validate that model parameters and settings are compatible with this backend.

        Called before first inference.

        Args:
            lmax: Maximum degree of spherical harmonics.
            mmax: Maximum order of spherical harmonics.
            settings: Inference settings.

        Raises:
            ValueError: If incompatible with this backend.
        """

    @staticmethod
    def prepare_model_for_inference(model: paddle.nn.Module) -> None:
        """
        Prepare a model for inference with backend-specific transforms.

        Called once during prepare_for_inference. Override in subclasses
        to apply model transformations (e.g. SO2 block conversion).

        Args:
            model: The backbone model to prepare.
        """

    @staticmethod
    def get_layer_radial_emb(
        x_edge: paddle.Tensor, model: paddle.nn.Module
    ) -> list[paddle.Tensor]:
        """
        Get edge embeddings for each layer.

        Default implementation returns the same raw x_edge for all layers.
        SO2_Convolution will compute rad_func(x_edge) internally.

        Override in fast backends to precompute radials.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model

        Returns:
            List of edge embeddings, one per layer
        """
        return [x_edge] * len(model.blocks)

    @staticmethod
    def prepare_wigner(
        wigner: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        mappingReduced,
        coefficient_index: (paddle.Tensor | None),
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Transform raw Wigner matrices for this backend.

        Default: Apply coefficient selection (if mmax != lmax) and
        pre-compose with M-mapping via einsum.

        Args:
            wigner: Raw Wigner matrices [E, L, L]
            wigner_inv: Raw inverse Wigner matrices [E, L, L]
            mappingReduced: CoefficientMapping with to_m matrix
            coefficient_index: Indices for mmax != lmax selection,
                or None if mmax == lmax.

        Returns:
            Transformed (wigner, wigner_inv) ready for this backend.
        """
        if coefficient_index is not None:
            wigner = wigner.index_select(1, coefficient_index)
            wigner_inv = wigner_inv.index_select(2, coefficient_index)
        wigner = paddle.einsum(
            "mk,nkj->nmj", mappingReduced.to_m.astype(wigner.dtype), wigner
        )
        wigner_inv = paddle.einsum(
            "njk,mk->njm", wigner_inv, mappingReduced.to_m.astype(wigner_inv.dtype)
        )
        return wigner, wigner_inv

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: paddle.Tensor, edge_index: paddle.Tensor, wigner: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Gather node features and rotate L->M.

        Default: PyTorch gather + BMM.

        Args:
            x_full: Node features [N, L, C]
            edge_index: Edge indices [2, E]
            wigner: Wigner rotation matrices [E, M, L] or [E, M, 2L]

        Returns:
            Rotated edge messages [E, M, 2C]
        """
        x_source = x_full[edge_index[0]]
        x_target = x_full[edge_index[1]]
        x_message = paddle.cat((x_source, x_target), dim=2)
        return paddle.bmm(wigner, x_message)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        edge_index: paddle.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> paddle.Tensor:
        """
        Rotate M->L and scatter edge messages to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x_message: Edge message features [E, M, C]
            wigner_inv: Inverse Wigner matrices [E, L, M]
            edge_index: Edge indices [2, E]
            num_nodes: Total number of nodes (output size)
            node_offset: Offset for node indices (for chunking)

        Returns:
            Node embeddings [N, L, C] accumulated from edge messages
        """
        x_rotated = paddle.bmm(wigner_inv, x_message)
        new_embedding = paddle.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            place=x_rotated.place,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(
        x: paddle.Tensor,
        radial_output: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        edge_index: paddle.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int = 0,
    ) -> paddle.Tensor:
        """
        Edge degree embedding: rotate radial and scatter to nodes.

        Default: PyTorch BMM + index_add.

        Args:
            x: Node features [N, L, C] to update
            radial_output: RadialMLP output [E, m0 * C]
            wigner_inv: Wigner inverse with envelope pre-fused
                [E, L, m0] or [E, L, L]
            edge_index: Edge indices [2, E]
            m_0_num_coefficients: Number of m=0 coefficients
                (3 for lmax=2)
            sphere_channels: Number of channels C
            rescale_factor: Aggregation rescale factor
            node_offset: Node offset for graph parallelism

        Returns:
            Updated node features [N, L, C]
        """
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        x_edge_embedding = paddle.bmm(wigner_inv_m0, radial)
        x_edge_embedding = x_edge_embedding.astype(x.dtype)
        return x.index_add(
            0, edge_index[1] - node_offset, x_edge_embedding / rescale_factor
        )


class UMASFastPytorchBackend(ExecutionBackend):
    """
    Optimized PyTorch backend using block-diagonal SO2 convolutions.

    Requires merge_mole=True and activation_checkpointing=False.
    """

    @staticmethod
    def validate(lmax: int, mmax: int, settings: InferenceSettings) -> None:
        """
        Validate that settings are compatible with fast pytorch mode.
        """
        if settings is not None and settings.activation_checkpointing:
            raise ValueError(
                "UMASFastPytorchBackend requires activation_checkpointing=False"
            )

    @staticmethod
    def prepare_model_for_inference(model: paddle.nn.Module) -> None:
        """
        Convert SO2_Convolution modules to block-diagonal GEMM variants
        and create unified radial MLP for batched computation.

        Replaces so2_conv_1 with SO2_Conv1_WithRadialBlock and
        so2_conv_2 with SO2_Conv2_InternalBlock in each block's
        Edgewise module. Then creates a UnifiedRadialMLP from all
        radial functions for efficient batched computation.
        """
        from .so2_layers import convert_so2_conv1, convert_so2_conv2

        for block in model.blocks:
            block.edge_wise.so2_conv_1 = convert_so2_conv1(block.edge_wise.so2_conv_1)
            block.edge_wise.so2_conv_2 = convert_so2_conv2(block.edge_wise.so2_conv_2)
        rad_funcs = [block.edge_wise.so2_conv_1.rad_func for block in model.blocks]
        model._unified_radial_mlp = UnifiedRadialMLP(rad_funcs)

    @staticmethod
    def get_layer_radial_emb(
        x_edge: paddle.Tensor, model: paddle.nn.Module
    ) -> list[paddle.Tensor]:
        """
        Compute radial embeddings for all layers using batched UnifiedRadialMLP.

        Args:
            x_edge: Edge embeddings [E, edge_features]
            model: The backbone model with _unified_radial_mlp

        Returns:
            List of radial embeddings, one per layer [E, radial_features]
        """
        return model._unified_radial_mlp(x_edge)


class UMASFastGPUBackend(UMASFastPytorchBackend):
    """
    GPU-optimized backend: SO2 block conversion + Triton kernels.

    Extends UMASFastPytorchBackend with Triton-accelerated
    node_to_edge_wigner_permute, permute_wigner_inv_edge_to_node, and edge_degree_scatter.
    Requires lmax==2, mmax==2, and merge_mole=True.

    Note: sphere_channels % 128 == 0 gives optimal GPU utilization.
    Smaller values work but with reduced efficiency.
    """

    @staticmethod
    def validate(lmax: int, mmax: int, settings: InferenceSettings) -> None:
        UMASFastPytorchBackend.validate(lmax, mmax, settings)
        if not paddle.cuda.is_available():
            raise ValueError("umas_fast_gpu requires CUDA")
        if lmax != 2 or mmax != 2:
            raise ValueError("umas_fast_gpu requires lmax==2 and mmax==2")
        if not settings.merge_mole:
            raise ValueError("umas_fast_gpu requires merge_mole=True")

    @staticmethod
    def prepare_wigner(
        wigner: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        mappingReduced,
        coefficient_index: (paddle.Tensor | None),
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        # Paddle fallback keeps correctness without requiring torch/triton custom ops.
        return ExecutionBackend.prepare_wigner(
            wigner, wigner_inv, mappingReduced, coefficient_index
        )

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: paddle.Tensor, edge_index: paddle.Tensor, wigner: paddle.Tensor
    ) -> paddle.Tensor:
        return ExecutionBackend.node_to_edge_wigner_permute(x_full, edge_index, wigner)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        edge_index: paddle.Tensor,
        num_nodes: int,
        node_offset: int = 0,
    ) -> paddle.Tensor:
        return ExecutionBackend.permute_wigner_inv_edge_to_node(
            x_message, wigner_inv, edge_index, num_nodes, node_offset
        )

    @staticmethod
    def edge_degree_scatter(
        x: paddle.Tensor,
        radial_output: paddle.Tensor,
        wigner_inv: paddle.Tensor,
        edge_index: paddle.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int = 0,
    ) -> paddle.Tensor:
        return ExecutionBackend.edge_degree_scatter(
            x,
            radial_output,
            wigner_inv,
            edge_index,
            m_0_num_coefficients,
            sphere_channels,
            rescale_factor,
            node_offset,
        )


_EXECUTION_BACKENDS: dict[ExecutionMode, type[ExecutionBackend]] = {
    ExecutionMode.GENERAL: ExecutionBackend,
    ExecutionMode.UMAS_FAST_PYTORCH: UMASFastPytorchBackend,
    ExecutionMode.UMAS_FAST_GPU: UMASFastGPUBackend,
}


def get_execution_backend(
    mode: (ExecutionMode | str) = ExecutionMode.GENERAL,
) -> ExecutionBackend:
    """
    Factory function to create the appropriate execution backend.

    Args:
        mode: Execution mode (enum or string). Defaults to GENERAL.

    Returns:
        Configured execution backend instance
    """
    if isinstance(mode, str):
        mode = ExecutionMode(mode)
    if mode not in _EXECUTION_BACKENDS:
        available = [m.value for m in _EXECUTION_BACKENDS]
        raise ValueError(f"Unknown execution mode: {mode}. Available: {available}")
    return _EXECUTION_BACKENDS[mode]()


def maybe_update_settings_backend(
    settings: InferenceSettings, model_config: dict
) -> InferenceSettings:
    """
    Update inference settings to use UMAS_FAST_GPU if conditions are met.

    Sets execution_mode to UMAS_FAST_GPU if:
    - execution_mode is not already set
    - UMASFastGPUBackend.validate passes for the model and settings

    Args:
        settings: Current inference settings.
        model_config: The model configuration dictionary to validate.

    Returns:
        Updated inference settings with the appropriate execution mode.
    """
    if settings.execution_mode is not None:
        return settings
    try:
        lmax = model_config["backbone"]["lmax"]
        mmax = model_config["backbone"]["mmax"]
        UMASFastGPUBackend.validate(lmax, mmax, settings)
        return replace(settings, execution_mode=ExecutionMode.UMAS_FAST_GPU)
    except (ValueError, KeyError):
        return settings
