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

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from omegaconf import DictConfig, ListConfig

from .paddle_utils import *

from ._compat import gp_utils
from ._compat.registry import registry
from ._compat.utils import conditional_grad
from ._compat.graph import generate_graph
from ._compat.base import HeadInterface
from .common.quaternion.quaternion_wigner_utils import create_wigner_data_module
from .common.quaternion.wigner_d_hybrid import axis_angle_wigner_hybrid
from .common.rotation import eulers_to_wigner, init_edge_rot_euler_angles
from .common.so3 import CoefficientMapping, SO3_Grid
from .nn.embedding import ChgSpinEmbedding, DatasetEmbedding, EdgeDegreeEmbedding
from .nn.execution_backends import get_execution_backend
from .nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .nn.mole_utils import MOLEInterface
from .nn.radial import GaussianSmearing, PolynomialEnvelope
from .nn.so3_layers import SO3_Linear
from .outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    compute_hessian,
    get_l_component_range,
    reduce_node_to_system,
)
from ._compat.irreps import cg_change_mat, irreps_sum
from ._compat.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    UMATask,
)
from ._compat.inference import OutputSpec, Task, InferenceSettings

from .escn_md_block import eSCNMD_Block

if TYPE_CHECKING:
    from ase import Atoms

ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE = 1024 * 128
AUTO_EDGE_CHUNK_FRACTION = 0.05


def _resolve_jd_path() -> str:
    """Resolve the packaged Wigner-d coefficient tensor file."""
    override = os.getenv("PPMAT_UMA_JD_PATH")
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return str(override_path)
        raise FileNotFoundError(
            f"PPMAT_UMA_JD_PATH points to a missing file: {override_path}"
        )
    here = Path(__file__).resolve().parent
    jd_path = here / "Jd.pt"
    if jd_path.exists():
        return str(jd_path)
    raise FileNotFoundError(
        "UMA requires `Jd.pt`. Please place it under `ppmat/models/uma/` "
        "or set PPMAT_UMA_JD_PATH to its location."
    )


def _load_jd_tensors() -> list[paddle.Tensor]:
    """Load `Jd.pt` from either Paddle or Torch serialization format."""
    jd_path = _resolve_jd_path()
    try:
        jd_list = paddle.load(path=jd_path)
        return [paddle.to_tensor(x) for x in jd_list]
    except Exception:
        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                "Failed to load `Jd.pt` via `paddle.load`, and `torch` is unavailable "
                "for fallback deserialization."
            ) from e
        jd_list_torch = torch.load(jd_path, map_location="cpu")
        return [
            paddle.to_tensor(
                x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            )
            for x in jd_list_torch
        ]


def _record_function(_name: str):
    """No-op profiler context for Paddle migration."""
    return nullcontext()


def _all_reduce_with_grad(tensor: paddle.Tensor) -> paddle.Tensor:
    """GP all-reduce with autograd support via gp_utils."""
    if not gp_utils.initialized():
        return tensor
    return gp_utils.reduce_from_model_parallel_region(tensor)


@dataclass
class GradRegressConfig:
    """
    Configuration for gradient-based computation of forces and stress.
    """

    direct_forces: bool = False
    direct_stress: bool = False
    forces: bool = False
    stress: bool = False
    hessian: bool = False
    hessian_vmap: bool = True


def add_n_empty_edges(
    graph_dict: dict, edges_to_add: int, cutoff: float, node_offset: int = 0
):
    graph_dict["edge_index"] = paddle.cat(
        (
            graph_dict["edge_index"].new_ones(2, edges_to_add) * node_offset,
            graph_dict["edge_index"],
        ),
        dim=1,
    )
    self_edge_distance_vec = graph_dict["edge_distance_vec"].new_ones(1, 3) + cutoff
    graph_dict["edge_distance_vec"] = paddle.cat(
        (
            self_edge_distance_vec.expand(edges_to_add, 3),
            graph_dict["edge_distance_vec"],
        ),
        dim=0,
    )
    edge_distance = paddle.linalg.norm(self_edge_distance_vec, dim=-1, keepdim=False)
    graph_dict["edge_distance"] = paddle.cat(
        (edge_distance.expand(edges_to_add), graph_dict["edge_distance"]), dim=0
    )


def validate_contiguous_channels(channels: list[int], name: str) -> tuple[int, int]:
    """Validate channels are contiguous, return (start, end) slice indices.

    Args:
        channels: List of channel indices to validate
        name: Name of the channel list for error messages

    Returns:
        Tuple of (start_idx, end_idx) for slicing. Returns (0, 0) if channels is empty.

    Raises:
        ValueError: If channels are not contiguous
    """
    if not channels:
        return 0, 0
    sorted_channels = sorted(channels)
    expected = list(range(sorted_channels[0], sorted_channels[-1] + 1))
    if sorted_channels != expected:
        raise ValueError(f"{name} must be contiguous (e.g., [0, 1, 2]). Got {channels}")
    return sorted_channels[0], sorted_channels[-1] + 1


def balance_channels_batched(
    emb: paddle.Tensor,
    target: paddle.Tensor,
    natoms: paddle.Tensor,
    batch: paddle.Tensor,
    start_idx: int,
    end_idx: int,
    target_offset: float = 0.0,
) -> paddle.Tensor:
    """Balance a contiguous range of channels to target sum per system.

    This batched version processes all channels in a contiguous range in a single
    call, which is more efficient than processing each channel individually.

    Args:
        emb: Node embeddings of shape [num_atoms, sph_features, channels]
        target: Target sum per system of shape [num_systems]
        natoms: Number of atoms per system of shape [num_systems]
        batch: Batch indices mapping atoms to systems of shape [num_atoms]
        start_idx: Start index of channel range (inclusive)
        end_idx: End index of channel range (exclusive)
        target_offset: Offset to subtract from target (e.g., 1.0 for spin)

    Returns:
        Modified embeddings with the specified channel range balanced to sum to target.

    Supports graph parallel (GP) mode using gp_utils reduction which
    provides correct gradients in both forward and backward passes.
    """
    out_emb = emb.clone()
    num_systems = len(natoms)
    n_channels = end_idx - start_idx
    channels_to_balance = emb[:, 0, start_idx:end_idx]
    system_sums = paddle.zeros([num_systems, n_channels], dtype=emb.dtype)
    system_sums.index_add_(0, batch, channels_to_balance)
    if gp_utils.initialized():
        system_sums = _all_reduce_with_grad(system_sums)
    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    corrections = (system_sums - target_sums) / natoms.unsqueeze(1)
    out_emb[:, 0, start_idx:end_idx] = channels_to_balance - corrections[batch]
    return out_emb


def resolve_dataset_mapping(
    deprecated_list: (list[str] | None),
    dataset_mapping: (dict[str, str] | None),
    deprecated_param_name: str = "dataset_list",
) -> dict[str, str]:
    """
    Validate and resolve dataset mapping from either a deprecated list or a mapping dict.

    Args:
        deprecated_list: Deprecated list of dataset names. If provided, it is
            converted to a mapping where each name maps to itself.
        dataset_mapping: Mapping from the config dataset name to desired dataset name for embeddings and heads.
            Allows multiple subsets to share the same dataset embedding and/or output head by mapping
            them to the same identifier.
        deprecated_param_name: Name of the deprecated parameter, used in
            warning/error messages.

    Returns:
        The resolved dataset mapping dict.

    Raises:
        ValueError: If both or neither arguments are provided, if the mapping
            is not a non-empty dict, or if mapping values are not a subset of
            mapping keys.
    """
    if deprecated_list is not None and dataset_mapping is not None:
        msg = f"Both '{deprecated_param_name}' (={deprecated_list}) and 'dataset_mapping' (={dataset_mapping}) have been provided. Please provide 'dataset_mapping' only in the config as '{deprecated_param_name}' is deprecated."
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is None and dataset_mapping is None:
        msg = "'dataset_mapping' must be provided in the config to use dataset embeddings."
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if deprecated_list is not None:
        if not isinstance(deprecated_list, (list, ListConfig)):
            msg = f"If '{deprecated_param_name}' is provided in the config, it must be a list of dataset names. Got: {deprecated_list!r}"
            logging.error(msg, stack_info=True)
            raise ValueError(msg)
        dataset_mapping = {name: name for name in deprecated_list}
        logging.warning(
            f"If '{deprecated_param_name}' is provided in the config, the code assumes that each dataset maps to itself. Please use 'dataset_mapping' as '{deprecated_param_name}' is deprecated and will be removed in the future."
        )
    if not isinstance(dataset_mapping, (dict, DictConfig)) or not dataset_mapping:
        msg = f"'dataset_mapping' must be a non-empty dictionary, got: {dataset_mapping!r}"
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    if not set(dataset_mapping.values()) <= set(dataset_mapping.keys()):
        missing = set(dataset_mapping.values()) - set(dataset_mapping.keys())
        msg = f"dataset_mapping values {missing} are not present in dataset_mapping keys {set(dataset_mapping.keys())}. Values must be a subset of keys. Full mapping provided: {dataset_mapping}"
        logging.error(msg, stack_info=True)
        raise ValueError(msg)
    return dataset_mapping


@registry.register_model("escnmd_backbone")
class eSCNMDBackbone(paddle.nn.Module, MOLEInterface):
    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: (int | None) = None,
        num_sphere_samples: int = 128,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = True,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: Literal["gaussian"] = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        direct_stress: bool = False,
        regress_stress: bool = False,
        regress_hessian: bool = False,
        hessian_vmap: bool = True,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: (list[str] | None) = None,
        dataset_mapping: (dict[str, str] | None) = None,
        use_dataset_embedding: bool = True,
        use_cuda_graph_wigner: bool = False,
        use_quaternion_wigner: bool = True,
        radius_pbc_version: int = 2,
        always_use_pbc: bool = True,
        charge_balanced_channels: (list[int] | None) = None,
        spin_balanced_channels: (list[int] | None) = None,
        edge_chunk_size: int = 1,
        execution_mode: str = "general",
    ) -> None:
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.num_sphere_samples = num_sphere_samples
        self.always_use_pbc = always_use_pbc
        self.regress_config = GradRegressConfig(
            direct_forces=direct_forces,
            forces=regress_forces,
            stress=regress_stress,
            direct_stress=direct_stress,
            hessian=regress_hessian,
            hessian_vmap=hessian_vmap,
        )
        charge_channels = (
            list(charge_balanced_channels) if charge_balanced_channels else []
        )
        spin_channels = list(spin_balanced_channels) if spin_balanced_channels else []
        (
            self.charge_channel_start,
            self.charge_channel_end,
        ) = validate_contiguous_channels(charge_channels, "charge_balanced_channels")
        self.spin_channel_start, self.spin_channel_end = validate_contiguous_channels(
            spin_channels, "spin_balanced_channels"
        )
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.use_quaternion_wigner = use_quaternion_wigner
        self.enforce_max_neighbors_strictly = False
        activation_checkpoint_chunk_size = None
        if activation_checkpointing:
            activation_checkpoint_chunk_size = (
                ESCNMD_DEFAULT_EDGE_ACTIVATION_CHECKPOINT_CHUNK_SIZE
            )
        self.edge_chunk_size = edge_chunk_size
        self.backend = get_execution_backend(execution_mode)
        self.chg_spin_emb_type = chg_spin_emb_type
        self.cs_emb_grad = cs_emb_grad
        self.dataset_emb_grad = dataset_emb_grad
        self.dataset_mapping = dataset_mapping
        self.dataset_list = dataset_list
        self.use_dataset_embedding = use_dataset_embedding
        if self.use_dataset_embedding:
            self.dataset_mapping = resolve_dataset_mapping(
                self.dataset_list, dataset_mapping, "dataset_list"
            )
        Jd_list = _load_jd_tensors()
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        if self.use_quaternion_wigner:
            self.wigner_data = create_wigner_data_module(lmax=self.lmax, lmin=5)
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.SO3_grid = paddle.nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )
        self.sphere_embedding = paddle.nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )
        self.charge_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type,
            "charge",
            self.sphere_channels,
            grad=self.cs_emb_grad,
        )
        self.spin_embedding = ChgSpinEmbedding(
            self.chg_spin_emb_type, "spin", self.sphere_channels, grad=self.cs_emb_grad
        )
        if self.use_dataset_embedding:
            self.dataset_embedding = DatasetEmbedding(
                self.sphere_channels,
                enable_grad=self.dataset_emb_grad,
                dataset_mapping=self.dataset_mapping,
            )
            self.mix_csd = paddle.nn.Linear(
                3 * self.sphere_channels, self.sphere_channels
            )
        else:
            self.mix_csd = paddle.nn.Linear(
                2 * self.sphere_channels, self.sphere_channels
            )
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0, self.cutoff, self.num_distance_basis, 2.0
            )
        else:
            raise ValueError("Unknown distance function")
        self.source_embedding = paddle.nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        self.target_embedding = paddle.nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        paddle.nn.initializer.Uniform(-0.001, 0.001)(self.source_embedding.weight)
        paddle.nn.initializer.Uniform(-0.001, 0.001)(self.target_embedding.weight)
        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=self.backend,
        )
        self.envelope = PolynomialEnvelope(exponent=5)
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type
        self.blocks = paddle.nn.ModuleList()
        for _ in range(self.num_layers):
            block = eSCNMD_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.ff_type,
                activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
                backend=self.backend,
            )
            self.blocks.append(block)
        self.norm = get_normalization_layer(
            self.norm_type, lmax=self.lmax, num_channels=self.sphere_channels
        )
        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    @property
    def direct_forces(self) -> bool:
        return self.regress_config.direct_forces

    @property
    def regress_forces(self) -> bool:
        return self.regress_config.forces

    @property
    def regress_stress(self) -> bool:
        return self.regress_config.stress

    def balance_channels(
        self,
        x_message_prime: paddle.Tensor,
        charge: paddle.Tensor,
        spin: paddle.Tensor,
        natoms: paddle.Tensor,
        batch: paddle.Tensor,
    ) -> paddle.Tensor:
        if self.charge_channel_end > self.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms=natoms,
                batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end,
                target_offset=0.0,
            )
        if self.spin_channel_end > self.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms=natoms,
                batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end,
                target_offset=1.0,
            )
        return x_message_prime

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if self.use_quaternion_wigner:
            with _record_function("obtain rotmat wigner quaternion"):
                wigner, wigner_inv = axis_angle_wigner_hybrid(
                    edge_distance_vecs,
                    self.lmax,
                    coeffs=self.wigner_data.coeffs,
                    U_blocks=self.wigner_data.U_blocks,
                    custom_kernels=self.wigner_data.custom_kernels,
                )
        else:
            Jd_buffers = [
                getattr(self, f"Jd_{l}").astype(edge_distance_vecs.dtype)
                for l in range(self.lmax + 1)
            ]
            with _record_function("obtain rotmat wigner original"):
                euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
                wigner = eulers_to_wigner(euler_angles, 0, self.lmax, Jd_buffers)
                wigner_inv = paddle.transpose(wigner, [0, 2, 1])
        return wigner, wigner_inv

    def csd_embedding(self, charge, spin, dataset):
        with _record_function("charge spin dataset embeddings"):
            chg_emb = self.charge_embedding(charge)
            spin_emb = self.spin_embedding(spin)
            if self.use_dataset_embedding:
                assert dataset is not None
                dataset_emb = self.dataset_embedding(dataset)
                return paddle.nn.SiLU()(
                    self.mix_csd(paddle.cat((chg_emb, spin_emb, dataset_emb), dim=1))
                )
            return paddle.nn.SiLU()(
                self.mix_csd(paddle.cat((chg_emb, spin_emb), dim=1))
            )

    def _generate_graph(self, data_dict):
        data_dict["gp_node_offset"] = 0
        node_partition = None
        if gp_utils.initialized():
            atomic_numbers_full = data_dict["atomic_numbers_full"]
            node_partition = paddle.tensor_split(
                paddle.arange(len(atomic_numbers_full)),
                gp_utils.get_gp_world_size(),
            )[gp_utils.get_gp_rank()]
            assert (
                node_partition.numel() > 0
            ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"
        if self.otf_graph:
            pbc = None
            if self.always_use_pbc:
                # dict/Batch containers do not guarantee len() == num_graphs.
                pbc = paddle.ones([len(data_dict["natoms"]), 3], dtype=paddle.bool)
            else:
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            assert (
                pbc.all() or (~pbc).all()
            ), "We can only accept pbc that is all true or all false"
            graph_dict = generate_graph(
                data_dict,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                pbc=pbc,
                node_partition=node_partition,
            )
        else:
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"
            if len(data_dict["natoms"]) == 1:
                shifts = data_dict["cell_offsets"].astype(
                    data_dict["cell"].dtype
                ) @ data_dict["cell"].squeeze(0)
            else:
                cell_per_edge = data_dict["cell"].repeat_interleave(
                    data_dict["nedges"], dim=0
                )
                shifts = paddle.einsum(
                    "ij,ijk->ik",
                    data_dict["cell_offsets"].astype(cell_per_edge.dtype),
                    cell_per_edge,
                )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )
            edge_distance = paddle.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)
            graph_dict = {
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
            }
        if gp_utils.initialized():
            data_dict["atomic_numbers"] = data_dict["atomic_numbers_full"][
                node_partition
            ]
            data_dict["batch"] = data_dict["batch_full"][node_partition]
            data_dict["gp_node_offset"] = paddle.min(node_partition).item()
        if graph_dict["edge_index"].shape[1] == 0:
            add_n_empty_edges(
                graph_dict, 1, self.cutoff, data_dict.get("gp_node_offset", 0)
            )
        return graph_dict

    @conditional_grad(paddle.enable_grad())
    def forward(self, data_dict: AtomicData) -> dict[str, paddle.Tensor]:
        data_dict["atomic_numbers"] = data_dict["atomic_numbers"].long()
        data_dict["atomic_numbers_full"] = data_dict["atomic_numbers"]
        data_dict["batch_full"] = data_dict["batch"]
        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict["dataset"] if "dataset" in data_dict else None,
        )
        self.set_MOLE_coefficients(
            atomic_numbers_full=data_dict["atomic_numbers_full"],
            batch_full=data_dict["batch_full"],
            csd_mixed_emb=csd_mixed_emb,
        )
        if not self.regress_config.direct_forces:
            if self.regress_config.forces or self.regress_config.stress:
                data_dict["pos"].requires_grad_(True)
            if self.regress_config.stress:
                data_dict["cell"].requires_grad_(True)
        with _record_function("generate_graph"):
            graph_dict = self._generate_graph(data_dict)
        with _record_function("obtain wigner"):
            wigner, wigner_inv = self._get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"]
            )
            coefficient_index = (
                self.coefficient_index if self.mmax != self.lmax else None
            )
            wigner, wigner_inv = self.backend.prepare_wigner(
                wigner, wigner_inv, self.mappingReduced, coefficient_index
            )
        with _record_function("atom embedding"):
            num_atoms = data_dict["atomic_numbers"].shape[0]
            x_message = paddle.zeros(
                [num_atoms, self.sph_feature_size, self.sphere_channels],
                dtype=data_dict["pos"].dtype,
            )
            x_scalar = self.sphere_embedding(data_dict["atomic_numbers"])
        sys_node_embedding = csd_mixed_emb[data_dict["batch"]]
        x_scalar = x_scalar + sys_node_embedding
        if self.sph_feature_size > 1:
            x_message = paddle.concat(
                [x_scalar.unsqueeze(1), x_message[:, 1:, :]], axis=1
            )
        else:
            x_message = x_scalar.unsqueeze(1)
        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=data_dict["batch_full"],
            edge_index=graph_dict["edge_index"],
        )
        self.log_MOLE_stats()
        with _record_function("edge embedding"):
            dist_scaled = graph_dict["edge_distance"] / self.cutoff
            edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
            edge_distance_embedding = self.distance_expansion(
                graph_dict["edge_distance"]
            )
            source_embedding = self.source_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][0]]
            )
            target_embedding = self.target_embedding(
                data_dict["atomic_numbers_full"][graph_dict["edge_index"][1]]
            )
            x_edge = paddle.cat(
                (edge_distance_embedding, source_embedding, target_embedding), dim=1
            )
            wigner_inv_envelope = wigner_inv * edge_envelope
            x_message = self.edge_degree_embedding(
                x_message,
                x_edge,
                graph_dict["edge_index"],
                wigner_inv_envelope,
                data_dict["gp_node_offset"],
            )
        with _record_function("layer_radial_emb"):
            x_edge_per_layer = self.backend.get_layer_radial_emb(x_edge, self)
        for i in range(self.num_layers):
            with _record_function(f"message passing {i}"):
                x_message = self.blocks[i](
                    x_message,
                    x_edge_per_layer[i],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv_envelope,
                    total_atoms_across_gp_ranks=data_dict["atomic_numbers_full"].shape[
                        0
                    ],
                    sys_node_embedding=sys_node_embedding,
                    node_offset=data_dict["gp_node_offset"],
                )
                x_message = self.balance_channels(
                    x_message,
                    charge=data_dict["charge"],
                    spin=data_dict["spin"],
                    natoms=data_dict["natoms"],
                    batch=data_dict["batch"],
                )
        x_message = self.norm(x_message)
        out = {"node_embedding": x_message, "batch": data_dict["batch"]}
        return out

    @property
    def num_params(self) -> int:
        return int(sum(np.prod(p.shape) for p in self.parameters()))

    @paddle.jit.not_to_static
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    paddle.nn.Linear,
                    SO3_Linear,
                    paddle.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (paddle.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)

    @classmethod
    def build_inference_settings(cls, settings: InferenceSettings) -> dict:
        """Build backbone config overrides from inference settings."""
        overrides = {}
        overrides["always_use_pbc"] = False
        if settings.activation_checkpointing is not None:
            overrides["activation_checkpointing"] = settings.activation_checkpointing
        if settings.edge_chunk_size is not None:
            overrides["edge_chunk_size"] = settings.edge_chunk_size
        if settings.external_graph_gen is not None:
            overrides["otf_graph"] = not settings.external_graph_gen
        if settings.internal_graph_gen_version is not None:
            overrides["radius_pbc_version"] = settings.internal_graph_gen_version
        if settings.use_quaternion_wigner is not None:
            overrides["use_quaternion_wigner"] = settings.use_quaternion_wigner
        if settings.execution_mode is not None:
            overrides["execution_mode"] = settings.execution_mode
        return overrides

    def get_default_untrained_tasks(
        self, checkpoint_tasks: dict[str, Task], inference_settings: InferenceSettings
    ) -> list[Task]:
        """
        Return default untrained tasks for eSCNMDBackbone.

        For this backbone, we add stress tasks for all energy datasets
        that don't already have stress (either trained or explicitly requested).
        Stress can be computed via autograd from energy predictions.

        Returns empty list if the model uses direct forces, since autograd-based
        stress computation requires energy-conserving force computation.
        """
        if self.direct_forces:
            return []
        tasks = []
        energy_datasets = set()
        stress_datasets = set()
        energy_task_by_dataset = {}
        for task in checkpoint_tasks.values():
            if task.property == "energy":
                for dataset in task.datasets:
                    energy_datasets.add(dataset)
                    energy_task_by_dataset[dataset] = task
            elif task.property == "stress":
                stress_datasets.update(task.datasets)
        stress_datasets.update(inference_settings.predict_untrained_stress)
        missing_stress_datasets = energy_datasets - stress_datasets
        for dataset in missing_stress_datasets:
            energy_task = energy_task_by_dataset[dataset]
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            tasks.append(
                Task(
                    name=f"{task_prefix}stress",
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(
                        dim=[1, 9], dtype=inference_settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )
        return tasks

    def validate_tasks(self, dataset_to_tasks: dict[str, list]) -> None:
        """
        Validate that task datasets are compatible with this backbone.
        """
        if self.use_dataset_embedding:
            assert set(dataset_to_tasks.keys()).issubset(
                set(self.dataset_mapping.keys())
            ), "Datasets in tasks is not a strict subset of datasets in backbone."

    def prepare_for_inference(self, data: AtomicData, settings: InferenceSettings):
        """
        Prepare model for inference. Called once on first prediction.

        For UMA: handles MOLE merging if settings.merge_mole is True.
        Stores initial composition for consistency checking.

        Returns:
            self or a new merged backbone if MOLE merging was performed. We return
            because type could have changed due to merging MOLE.
        """
        self._inference_settings = settings
        self._merged_composition = None
        self.backend.validate(self.lmax, self.mmax, settings)
        if settings.merge_mole:
            assert (
                data.natoms.numel() == 1
            ), "Cannot merge model with multiple systems in batch"
            self._merged_composition = self._get_composition_info(data)
            new_backbone = self.merge_MOLE_model(data)
            new_backbone._inference_settings = settings
            new_backbone._merged_composition = self._merged_composition
            self.backend.prepare_model_for_inference(new_backbone)
            return new_backbone
        self.backend.prepare_model_for_inference(self)
        return self

    def on_predict_check(self, data: AtomicData) -> None:
        """
        Called before each prediction. UMA checks MOLE consistency here.
        """
        if not getattr(self, "_inference_settings", None):
            return
        if self._inference_settings.merge_mole and self._merged_composition is not None:
            assert (
                data.natoms.numel() == 1
            ), "Cannot run merged model on batch with multiple systems"
            current = self._get_composition_info(data)
            self._assert_composition_matches(current)

    def _get_composition_info(self, data) -> tuple:
        """
        Get composition info for MOLE consistency checking.
        """
        composition = data.atomic_numbers.new_zeros(
            self.max_num_elements, dtype=paddle.int32
        ).index_add(
            0,
            data.atomic_numbers.astype(paddle.int32),
            data.atomic_numbers.new_ones(len(data.atomic_numbers), dtype=paddle.int32),
        )
        return (
            composition,
            getattr(data, "charge", None),
            getattr(data, "spin", None),
            getattr(data, "dataset", [None]),
        )

    def _assert_composition_matches(self, current: tuple) -> None:
        """
        Assert current composition matches what model was merged on.
        """
        merged = self._merged_composition
        merged_norm = merged[0].float() / merged[0].sum()
        curr_norm = current[0].float() / current[0].sum()
        assert bool(
            paddle.all(paddle.isclose(merged_norm, curr_norm, rtol=1e-05))
        ), "Compositions differ from merged model"
        merged_charge = merged[1]
        curr_charge = (
            current[1] if isinstance(current[1], paddle.Tensor) else current[1]
        )
        assert (
            bool(paddle.all(merged_charge == curr_charge))
            if isinstance(merged_charge, paddle.Tensor)
            else merged_charge == curr_charge
        ), f"Charge differs: {merged_charge} vs {current[1]}"
        merged_spin = merged[2]
        curr_spin = current[2] if isinstance(current[2], paddle.Tensor) else current[2]
        assert (
            bool(paddle.all(merged_spin == curr_spin))
            if isinstance(merged_spin, paddle.Tensor)
            else merged_spin == curr_spin
        ), f"Spin differs: {merged_spin} vs {current[2]}"
        assert merged[3] == current[3], f"Dataset differs: {merged[3]} vs {current[3]}"

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        UMA-specific validation: handle charge/spin for OMOL task.

        Sets default values for charge and spin in atoms.info and validates
        they are within acceptable ranges.
        """
        if "charge" not in atoms.info:
            if task_name == UMATask.OMOL.value:
                logging.warning(
                    "task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0. Ensure charge is an integer representing the total charge on the system and is within the range -100 to 100."
                )
            atoms.info["charge"] = DEFAULT_CHARGE
        if "spin" not in atoms.info:
            if task_name == UMATask.OMOL.value:
                atoms.info["spin"] = DEFAULT_SPIN_OMOL
                logging.warning(
                    "task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=1. Ensure spin is an integer representing the spin multiplicity from 0 to 100."
                )
            else:
                atoms.info["spin"] = DEFAULT_SPIN
        charge = atoms.info["charge"]
        if not isinstance(charge, (int, np.integer)):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. Charge must be an integer representing the total charge on the system."
            )
        if not CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]:
            raise ValueError(
                f"Invalid value for charge: {charge}. Charge must be within the range {CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )
        spin = atoms.info["spin"]
        if not isinstance(spin, (int, np.integer)):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. Spin must be an integer representing the spin multiplicity."
            )
        if not SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]:
            raise ValueError(
                f"Invalid value for spin: {spin}. Spin must be within the range {SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )


class MLP_EFS_Head(paddle.nn.Module, HeadInterface):
    """MLP head for predicting energy, forces, and stress using autograd derivatives.

    This head computes forces and stress by taking gradients of the energy with respect to
    atomic positions and cell displacement.
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: (str | None) = None,
        wrap_property: bool = True,
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.prefix = prefix
        self.wrap_property = wrap_property
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = paddle.nn.Sequential(
            paddle.nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(self.hidden_channels, 1, bias=True),
        )
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_config = backbone.regress_config

    @property
    def regress_forces(self) -> bool:
        return self.regress_config.forces

    @property
    def regress_stress(self) -> bool:
        return self.regress_config.stress

    @conditional_grad(paddle.enable_grad())
    def forward(
        self, data: AtomicData, emb: dict[str, paddle.Tensor]
    ) -> dict[str, paddle.Tensor]:
        energy_key = f"{self.prefix}_energy" if self.prefix else "energy"
        forces_key = f"{self.prefix}_forces" if self.prefix else "forces"
        stress_key = f"{self.prefix}_stress" if self.prefix else "stress"
        hessian_key = f"{self.prefix}_hessian" if self.prefix else "hessian"
        outputs = {}
        energy, energy_part = compute_energy(
            emb,
            self.energy_block,
            data["batch"],
            len(data["natoms"]),
            natoms=data["natoms"],
            reduce=self.reduce,
        )
        outputs[energy_key] = {"energy": energy} if self.wrap_property else energy
        if not gp_utils.initialized():
            embeddings = emb["node_embedding"].detach()
            outputs["embeddings"] = (
                {"embeddings": embeddings} if self.wrap_property else embeddings
            )
        create_graph = self.training or self.regress_config.hessian
        if self.regress_config.stress and not self.regress_config.direct_stress:
            forces, stress = compute_forces_and_stress(
                energy_part,
                data["pos"],
                data["cell"],
                batch=data["batch_full"],
                training=create_graph,
            )
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
            outputs[stress_key] = {"stress": stress} if self.wrap_property else stress
        elif self.regress_config.forces and not self.regress_config.direct_forces:
            forces = compute_forces(energy_part, data["pos"], training=self.training)
            outputs[forces_key] = {"forces": forces} if self.wrap_property else forces
        else:
            forces = None
        if self.regress_config.hessian:
            if forces is None:
                raise ValueError(
                    "Hessian computation requires forces. Please enable regress_forces or regress_stress."
                )
            if data["natoms"].numel() != 1:
                raise ValueError(
                    f"Hessian computation requires exactly 1 system in batch, found {data['natoms'].numel()}"
                )
            hessian = compute_hessian(
                forces,
                data["pos"],
                vmap=self.regress_config.hessian_vmap,
                training=create_graph,
            )
            outputs[hessian_key] = (
                {"hessian": hessian} if self.wrap_property else hessian
            )
        return outputs


class MLP_Energy_Head(MLP_EFS_Head):
    """MLP head for predicting energy."""

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "sum",
        prefix: (str | None) = None,
        wrap_property: bool = False,
    ) -> None:
        super().__init__(backbone, reduce, prefix, wrap_property)
        assert (
            backbone.regress_forces is False
            and backbone.regress_stress is False
            or (backbone.direct_forces is True or backbone.direct_stress is True)
        ), "regress_forces and regress_stress must be False for MLP_Energy_Head or direct_forces must be True.Use MLP_EFS_Head if you want to predict gradient forces and stress."


class Linear_Energy_Head(paddle.nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = paddle.nn.Linear(backbone.sphere_channels, 1, bias=True)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, paddle.Tensor]
    ) -> dict[str, paddle.Tensor]:
        energy, _ = compute_energy(
            emb,
            self.energy_block,
            data_dict["batch"],
            len(data_dict["natoms"]),
            natoms=data_dict["natoms"],
            reduce=self.reduce,
        )
        return {"energy": energy}


class Linear_Force_Head(paddle.nn.Module, HeadInterface):
    def __init__(self, backbone: eSCNMDBackbone) -> None:
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, data_dict: AtomicData, emb: dict[str, paddle.Tensor]):
        l0_l1_embedding = get_l_component_range(emb["node_embedding"], l_min=0, l_max=1)
        forces_output = self.linear(l0_l1_embedding)
        forces = get_l_component_range(forces_output, l_min=1, l_max=1)
        forces = forces.view([-1, 3])
        if gp_utils.initialized():
            forces = gp_utils.gather_from_model_parallel_region(
                forces, data_dict["atomic_numbers_full"].shape[0]
            )
        return {"forces": forces}


def compose_tensor(trace: paddle.Tensor, l2_symmetric: paddle.Tensor) -> paddle.Tensor:
    """Re-compose a tensor from its decomposition

    Args:
        trace: a tensor with scalar part of the decomposition of r2 tensors in the batch
        l2_symmetric: tensor with the symmetric/traceless part of decomposition

    Returns:
        tensor: rank 2 tensor
    """
    if trace.shape[1] != 1:
        raise ValueError("batch of traces must be shape (batch size, 1)")
    if l2_symmetric.shape[1] != 5:
        raise ValueError("batch of l2_symmetric tensors must be shape (batch size, 5)")
    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError(
            "Shape missmatch between trace and l2_symmetric parts. The first dimension is the batch dimension"
        )
    batch_size = trace.shape[0]
    middle_size = irreps_sum(1) - irreps_sum(0)
    parts = [trace]
    if middle_size > 0:
        parts.append(paddle.zeros([batch_size, middle_size], dtype=trace.dtype))
    parts.append(l2_symmetric)
    decomposed_preds = paddle.concat(parts, axis=1)
    r2_tensor = paddle.einsum(
        "ba, cb->ca",
        cg_change_mat(2, place=trace.place, dtype=trace.dtype),
        decomposed_preds,
    )
    return r2_tensor


class MLP_Stress_Head(paddle.nn.Module, HeadInterface):
    """MLP head for predicting the stress tensor.

    Predicts the isotropic (L=0) and anisotropic (L=2) parts of the stress tensor
    separately to ensure symmetry, then recomposes back to the full stress tensor.
    """

    def __init__(self, backbone: eSCNMDBackbone, reduce: str = "mean") -> None:
        super().__init__()
        self.reduce = reduce
        assert reduce in ["sum", "mean"]
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.scalar_block = paddle.nn.Sequential(
            paddle.nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            paddle.nn.SiLU(),
            paddle.nn.Linear(self.hidden_channels, 1, bias=True),
        )
        self.l2_linear = SO3_Linear(backbone.sphere_channels, 1, lmax=2)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, paddle.Tensor]
    ) -> dict[str, paddle.Tensor]:
        num_systems = len(data_dict["natoms"])
        batch = data_dict["batch"]
        scalar_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=0
        ).squeeze(1)
        node_scalar = self.scalar_block(scalar_embedding).view(-1)
        iso_stress, _ = reduce_node_to_system(node_scalar, batch, num_systems)
        if self.reduce == "mean":
            iso_stress = iso_stress / data_dict["natoms"].astype(iso_stress.dtype)
        l0l1l2_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=2
        )
        l2_output = self.l2_linear(l0l1l2_embedding)
        node_l2 = get_l_component_range(l2_output, l_min=2, l_max=2).view([-1, 5])
        aniso_stress, _ = reduce_node_to_system(node_l2, batch, num_systems)
        if self.reduce == "mean":
            aniso_stress = aniso_stress / data_dict["natoms"].astype(
                aniso_stress.dtype
            ).unsqueeze(1)
        stress = compose_tensor(iso_stress.unsqueeze(1), aniso_stress)
        return {"stress": stress}


class MLP_EFS_Direct_Head(paddle.nn.Module, HeadInterface):
    """Direct E/F/S prediction head without force/stress autograd-of-energy path.

    This head is intended for migration smoke tests where Paddle may not support
    higher-order derivatives for all ops used by the backbone.
    """

    def __init__(
        self,
        backbone: eSCNMDBackbone,
        reduce: str = "mean",
        wrap_property: bool = True,
    ) -> None:
        super().__init__()
        self.wrap_property = wrap_property
        self.energy_head = MLP_Energy_Head(
            backbone, reduce=reduce, prefix=None, wrap_property=False
        )
        self.force_head = Linear_Force_Head(backbone)
        self.stress_head = MLP_Stress_Head(backbone, reduce=reduce)

    def forward(
        self, data_dict: AtomicData, emb: dict[str, paddle.Tensor]
    ) -> dict[str, paddle.Tensor]:
        energy = self.energy_head(data_dict, emb)["energy"]
        forces = self.force_head(data_dict, emb)["forces"]
        stress = self.stress_head(data_dict, emb)["stress"]
        if self.wrap_property:
            return {
                "energy": {"energy": energy},
                "forces": {"forces": forces},
                "stress": {"stress": stress},
            }
        return {"energy": energy, "forces": forces, "stress": stress}


@registry.register_model("uma_single_task")
class UMASingleTaskModel(paddle.nn.Layer):
    """Single-dataset UMA adapter for ppmat BaseTrainer.

    This wrapper keeps the fairchem-style backbone/head internals while exposing
    the ppmat training contract:
      - forward(...) -> {"loss_dict": ..., "pred_dict": ...}
    """

    _HEADS = {
        "MLP_EFS_Head": MLP_EFS_Head,
        "MLP_EFS_Direct_Head": MLP_EFS_Direct_Head,
        "MLP_Energy_Head": MLP_Energy_Head,
        "Linear_Energy_Head": Linear_Energy_Head,
        "Linear_Force_Head": Linear_Force_Head,
        "MLP_Stress_Head": MLP_Stress_Head,
    }

    def __init__(
        self,
        backbone_params: dict | None = None,
        head_name: str = "MLP_EFS_Head",
        head_params: dict | None = None,
        loss_type: str = "mae",
        loss_weights: dict[str, float] | None = None,
        label_map: dict[str, str] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        backbone_params = backbone_params or {}
        head_params = head_params or {}
        self.backbone = eSCNMDBackbone(**backbone_params)
        if head_name not in self._HEADS:
            raise ValueError(
                f"Unknown head_name={head_name!r}. Available: {sorted(self._HEADS)}"
            )
        self.head = self._HEADS[head_name](self.backbone, **head_params)
        self.loss_type = loss_type.lower()
        self.loss_weights = loss_weights or {}
        self.label_map = label_map or {}
        self.freeze_backbone = freeze_backbone

    def _flatten_head_output(self, outputs: dict[str, Any]) -> dict[str, paddle.Tensor]:
        pred_dict: dict[str, paddle.Tensor] = {}
        for key, value in outputs.items():
            if isinstance(value, dict):
                if len(value) == 1:
                    inner_key, inner_value = next(iter(value.items()))
                    pred_dict[key] = inner_value
                    pred_dict.setdefault(inner_key, inner_value)
                else:
                    for inner_key, inner_value in value.items():
                        pred_dict[f"{key}.{inner_key}"] = inner_value
            else:
                pred_dict[key] = value
        return pred_dict

    def _compute_loss(
        self, pred_dict: dict[str, paddle.Tensor], batch_data: dict[str, Any]
    ) -> dict[str, paddle.Tensor]:
        loss_dict: dict[str, paddle.Tensor] = {}
        total_loss: paddle.Tensor | None = None
        for pred_key, pred in pred_dict.items():
            label_key = self.label_map.get(pred_key, pred_key)
            if label_key not in batch_data:
                continue
            label = batch_data[label_key]
            if not paddle.is_tensor(label):
                label = paddle.to_tensor(label)
            if not paddle.is_tensor(pred):
                pred = paddle.to_tensor(pred)
            # Match rank first, then mask invalid targets (NaN/Inf).
            if label.shape != pred.shape:
                try:
                    label = label.reshape(pred.shape)
                except Exception:
                    continue
            valid_mask = paddle.isfinite(label)
            if not bool(valid_mask.astype("int32").sum().item()):
                continue
            pred_valid = paddle.masked_select(pred, valid_mask)
            label_valid = paddle.masked_select(label, valid_mask)
            if self.loss_type == "mae":
                comp_loss = F.l1_loss(pred_valid, label_valid)
            elif self.loss_type == "mse":
                comp_loss = F.mse_loss(pred_valid, label_valid)
            elif self.loss_type == "huber":
                comp_loss = F.smooth_l1_loss(pred_valid, label_valid)
            else:
                raise ValueError(
                    f"Unsupported loss_type={self.loss_type!r}. Use mae/mse/huber."
                )
            loss_dict[pred_key] = comp_loss
            weight = float(self.loss_weights.get(pred_key, 1.0))
            total_loss = (
                comp_loss * weight
                if total_loss is None
                else total_loss + comp_loss * weight
            )

        if total_loss is None:
            # Keep graph-connected zero loss so backward always works.
            connected_zero = None
            for pred in pred_dict.values():
                z = pred.sum() * 0.0
                connected_zero = z if connected_zero is None else connected_zero + z
            if connected_zero is None:
                connected_zero = paddle.to_tensor(0.0, dtype="float32")
            total_loss = connected_zero
        loss_dict["loss"] = total_loss
        return loss_dict

    def _to_python(self, value: Any):
        if paddle.is_tensor(value):
            return value.numpy()
        if isinstance(value, dict):
            return {k: self._to_python(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        return value

    def forward(
        self,
        batch_data: dict[str, Any],
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict[str, dict[str, Any]]:
        if self.freeze_backbone:
            with paddle.no_grad():
                emb = self.backbone(batch_data)
            emb = {
                k: (v.detach() if paddle.is_tensor(v) else v) for k, v in emb.items()
            }
        else:
            emb = self.backbone(batch_data)
        raw_outputs = self.head(batch_data, emb)
        pred_dict = self._flatten_head_output(raw_outputs)
        out = {"loss_dict": {}, "pred_dict": {}}
        if return_loss:
            out["loss_dict"] = self._compute_loss(pred_dict, batch_data)
        if return_prediction:
            out["pred_dict"] = pred_dict
        return out

    def predict(self, data):
        from .single_dataset import UMASingleCollator

        if isinstance(data, list):
            batch_data = UMASingleCollator()(data)
        elif isinstance(data, dict) and "batch" in data:
            batch_data = data
        else:
            if hasattr(data, "to_dict"):
                data = data.to_dict()
            elif not isinstance(data, dict) and hasattr(data, "__dict__"):
                data = {
                    k: v for k, v in data.__dict__.items() if not k.startswith("__")
                }
            batch_data = UMASingleCollator()([data])
        out = self.forward(batch_data, return_loss=False, return_prediction=True)
        return self._to_python(out["pred_dict"])
