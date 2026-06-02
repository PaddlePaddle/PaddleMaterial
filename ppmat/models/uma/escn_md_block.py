from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from contextlib import nullcontext
from typing import TYPE_CHECKING

import paddle
from typing_extensions import Literal

from .paddle_utils import *

from ._compat import gp_utils
from .nn.activation import (GateActivation,
                                                    SeparableS2Activation_M)
from .nn.layer_norm import get_normalization_layer
from .nn.mole import MOLE
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear

if TYPE_CHECKING:
    from .common.so3 import CoefficientMapping, SO3_Grid
    from .nn.execution_backends import ExecutionBackend


def _record_function(_name: str):
    """No-op profiler context for Paddle migration."""
    return nullcontext()


def set_mole_ac_start_index(module: paddle.nn.Module, index: int) -> None:
    for submodule in module.modules():
        if isinstance(submodule, MOLE):
            submodule.global_mole_tensors.ac_start_idx = index


class Edgewise(paddle.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        cutoff: float,
        activation_checkpoint_chunk_size: (int | None),
        backend: ExecutionBackend,
        act_type: Literal["gate", "s2"] = "gate",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.activation_checkpoint_chunk_size = activation_checkpoint_chunk_size
        self.backend = backend
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.act_type = act_type
        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            self.act = SeparableS2Activation_M(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")
        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=copy.deepcopy(edge_channels_list),
            extra_m0_output_channels=extra_m0_output_channels,
        )
        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

    def forward(
        self,
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks,
        node_offset: int = 0,
    ):
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                x, total_atoms_across_gp_ranks
            )
        else:
            x_full = x
        if self.activation_checkpoint_chunk_size is None:
            return self.forward_chunk(
                x_full,
                x.shape[0],
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                node_offset,
            )
        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner.split(self.activation_checkpoint_chunk_size, dim=0)
        wigner_inv_partitions = wigner_inv_envelope.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(self.activation_checkpoint_chunk_size, dim=0)
        new_embeddings = []
        ac_mole_start_idx = 0
        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                paddle.distributed.fleet.utils.recompute(
                    self.forward_chunk,
                    x_full,
                    x.shape[0],
                    x_edge_partitions[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    node_offset,
                    ac_mole_start_idx,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]
            if len(new_embeddings) > 8:
                new_embeddings = [paddle.stack(new_embeddings).sum(axis=0)]
        return paddle.stack(new_embeddings).sum(axis=0)

    def forward_chunk(
        self,
        x_full,
        x_original_shape,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        node_offset: int = 0,
        ac_mole_start_idx: int = 0,
    ):
        set_mole_ac_start_index(self, ac_mole_start_idx)
        with _record_function("SO2Conv"):
            x_message = self.backend.node_to_edge_wigner_permute(
                x_full, edge_index, wigner
            )
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message)
            new_embedding = self.backend.permute_wigner_inv_edge_to_node(
                x_message,
                wigner_inv_envelope,
                edge_index,
                x_original_shape,
                node_offset,
            )
        set_mole_ac_start_index(self, 0)
        return new_embedding


class SpectralAtomwise(paddle.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.scalar_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                self.sphere_channels, self.lmax * self.hidden_channels, bias=True
            ),
            paddle.nn.SiLU(),
        )
        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        x = self.so3_linear_2(x)
        return x


class GridAtomwise(paddle.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: SO3_Grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.grid_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                self.sphere_channels, self.hidden_channels, bias=False
            ),
            paddle.nn.SiLU(),
            paddle.nn.Linear(
                self.hidden_channels, self.hidden_channels, bias=False
            ),
            paddle.nn.SiLU(),
            paddle.nn.Linear(
                self.hidden_channels, self.sphere_channels, bias=False
            ),
        )

    def forward(self, x):
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        x_grid = self.grid_mlp(x_grid)
        x = self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)
        return x


class eSCNMD_Block(paddle.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        SO3_grid: SO3_Grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
        act_type: Literal["gate", "s2"],
        ff_type: Literal["spectral", "grid"],
        activation_checkpoint_chunk_size: (int | None),
        backend: ExecutionBackend,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )
        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=backend,
        )
        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )
        if ff_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif ff_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )

    def forward(
        self,
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks,
        sys_node_embedding=None,
        node_offset: int = 0,
    ):
        x_res = x
        x = self.norm_1(x)
        if sys_node_embedding is not None:
            x0 = x[:, 0, :] + sys_node_embedding
            if x.shape[1] > 1:
                x = paddle.concat([x0.unsqueeze(1), x[:, 1:, :]], axis=1)
            else:
                x = x0.unsqueeze(1)
        with _record_function("edgewise"):
            x = self.edge_wise(
                x,
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                total_atoms_across_gp_ranks=total_atoms_across_gp_ranks,
                node_offset=node_offset,
            )
            x = x + x_res
        x_res = x
        x = self.norm_2(x)
        with _record_function("atomwise"):
            x = self.atom_wise(x)
            x = x + x_res
        return x
