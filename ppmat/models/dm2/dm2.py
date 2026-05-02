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

# This implementation adapts the DM2/graphite NequIP denoising design:
# https://github.com/digital-synthesis-lab/DM2
# Original DM2/graphite code is MIT licensed,
# Copyright (c) 2022 Tim Hsu and (c) 2025 Digital Synthesis Lab @ UCLA.

from __future__ import annotations

import copy
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.nn import FullyConnectedNet
from ppmat.models.common.e3nn.nn import Gate
from ppmat.utils.scatter import scatter


def bessel(
    x: paddle.Tensor,
    start: float = 0.0,
    end: float = 1.0,
    num_basis: int = 8,
    eps: float = 1e-5,
) -> paddle.Tensor:
    """Expand scalar distances with the Bessel basis used by DM2/graphite."""

    x = x.unsqueeze(axis=-1) - start + eps
    width = end - start
    n = paddle.arange(1, num_basis + 1, dtype=x.dtype)
    return ((2.0 / width) ** 0.5) * paddle.sin(n * math.pi * x / width) / x


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Return whether a tensor-product path can produce ``ir_out``."""

    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Layer):
    """Compose two e3nn layers while preserving irreps metadata."""

    def __init__(self, first: nn.Layer, second: nn.Layer):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *inputs):
        return self.second(self.first(*inputs))


class GaussianBasisEmbedding(nn.Layer):
    """Embed scalar conditions with Gaussian basis functions.

    DM2 uses this for the cooling-rate conditional model. The same module is also
    useful for time or process-condition scalar conditioning.
    """

    def __init__(
        self,
        num_basis: int = 12,
        embedding_dim: int = 32,
        min_sigma: float = 0.1,
        learn_means: bool = False,
        learn_sigmas: bool = False,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        means = paddle.linspace(min_value, max_value, num_basis, dtype="float32")
        sigmas = paddle.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))

        if learn_means:
            self.means = self.create_parameter(
                shape=[num_basis],
                default_initializer=nn.initializer.Assign(means),
            )
        else:
            self.register_buffer("means", means)

        if learn_sigmas:
            self.sigmas = self.create_parameter(
                shape=[num_basis],
                default_initializer=nn.initializer.Assign(sigmas),
            )
        else:
            self.register_buffer("sigmas", sigmas)

        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def gaussian_basis(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(axis=1)
        x = x.astype(self.means.dtype)
        x_expanded = x.expand([-1, self.means.shape[0]])
        return paddle.exp(-0.5 * ((x_expanded - self.means) / self.sigmas) ** 2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        hidden = self.activation(self.layer1(self.gaussian_basis(x)))
        return self.layer2(hidden)


class DM2InitialEmbedding(nn.Layer):
    """Initial species and edge-distance embedding used by DM2."""

    def __init__(
        self,
        num_species: int,
        cutoff: float,
        node_embedding_dim: int = 8,
        edge_basis_size: int = 16,
    ):
        super().__init__()
        self.num_species = num_species
        self.cutoff = cutoff
        self.node_embedding_dim = node_embedding_dim
        self.edge_basis_size = edge_basis_size
        self.embed_node_x = nn.Embedding(num_species, node_embedding_dim)
        self.embed_node_z = nn.Embedding(num_species, node_embedding_dim)

    def forward(self, data: Data) -> Data:
        node_species = data.x.astype("int64").reshape([-1])
        data.h_node_x = self.embed_node_x(node_species)
        data.h_node_z = self.embed_node_z(node_species)
        edge_length = paddle.linalg.norm(data.edge_attr[:, :3], axis=-1)
        data.h_edge = bessel(
            edge_length,
            start=0.0,
            end=self.cutoff,
            num_basis=self.edge_basis_size,
        )
        return data


class DM2Interaction(nn.Layer):
    """NequIP interaction layer adapted from the official DM2 graphite code."""

    def __init__(
        self,
        irreps_in,
        irreps_node,
        irreps_edge,
        irreps_out,
        radial_neurons: Sequence[int] = (16, 64),
        num_neighbors: float = 1.0,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, permutation, _ = irreps_mid.sort()

        if irreps_mid.dim <= 0:
            raise ValueError(
                f"irreps_in={self.irreps_in} times irreps_edge={self.irreps_edge} "
                f"produces nothing in irreps_out={self.irreps_out}."
            )

        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.sc = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.lin1 = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_node,
            self.irreps_in,
            internal_weights=True,
            shared_weights=True,
        )
        self.conv = o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.lin2 = o3.FullyConnectedTensorProduct(
            irreps_mid,
            self.irreps_node,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.mlp = FullyConnectedNet(
            list(radial_neurons) + [self.conv.weight_numel],
            F.silu,
        )

        self.alpha = o3.FullyConnectedTensorProduct(
            irreps_mid,
            self.irreps_node,
            "0e",
            internal_weights=True,
            shared_weights=True,
        )
        with paddle.no_grad():
            self.alpha.weight.set_value(paddle.zeros_like(self.alpha.weight))
        if float(self.alpha.output_mask[0].item()) != 1.0:
            raise ValueError(
                f"irreps_mid={irreps_mid} and irreps_node={self.irreps_node} "
                "are not able to generate scalar skip weights."
            )

    def forward(
        self,
        x: paddle.Tensor,
        node_attr: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge_attr: paddle.Tensor,
        edge_len_emb: paddle.Tensor,
    ) -> paddle.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        node_self_connection = self.sc(x, node_attr)
        node_features = self.lin1(x, node_attr)
        edge_features = self.conv(
            node_features[src],
            edge_attr,
            weight=self.mlp(edge_len_emb),
        )
        node_features = scatter(
            edge_features,
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ) / (self.num_neighbors**0.5)
        node_conv_out = self.lin2(node_features, node_attr)

        alpha = self.alpha(node_features, node_attr)
        mask = self.sc.output_mask
        alpha = (1.0 - mask) + alpha * mask
        return node_self_connection + alpha * node_conv_out


class DM2NequIPDenoiser(nn.Layer):
    """Paddle implementation of the DM2 NequIP denoiser.

    The model follows ``digital-synthesis-lab/DM2``: species embeddings and
    Bessel edge-distance features are passed through gated NequIP interactions
    and a vector output head predicts Cartesian displacements/noise.
    """

    def __init__(
        self,
        num_species: int,
        cutoff: float = 5.0,
        node_embedding_dim: int = 8,
        edge_basis_size: int = 16,
        irreps_node_x: Optional[str] = None,
        irreps_node_z: Optional[str] = None,
        irreps_hidden: str = "64x0e + 32x1e",
        irreps_edge: str = "4x0e + 4x1e + 2x2e",
        irreps_out: str = "1x1e",
        num_convs: int = 3,
        radial_neurons: Sequence[int] = (16, 64),
        num_neighbors: float = 12.0,
        use_condition: bool = False,
        condition_key: str = "cooling_rate",
        condition_num_basis: int = 9,
        condition_min: float = -4.0,
        condition_max: float = 4.0,
        condition_min_sigma: float = 0.6,
    ):
        super().__init__()
        self.num_species = num_species
        self.cutoff = cutoff
        self.use_condition = use_condition
        self.condition_key = condition_key

        if irreps_node_x is None:
            irreps_node_x = f"{node_embedding_dim}x0e"
        if irreps_node_z is None:
            irreps_node_z = f"{node_embedding_dim}x0e"

        self.init_embed = DM2InitialEmbedding(
            num_species=num_species,
            cutoff=cutoff,
            node_embedding_dim=node_embedding_dim,
            edge_basis_size=edge_basis_size,
        )
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_convs = num_convs

        act_scalars = {1: F.silu, -1: paddle.tanh}
        act_gates = {1: F.sigmoid, -1: paddle.tanh}

        irreps = self.irreps_node_x
        self.interactions = nn.LayerList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)
                ]
            )

            if irreps_gated.dim > 0:
                if tp_path_exists(self.irreps_node_z, self.irreps_edge, "0e"):
                    gate_ir = "0e"
                elif tp_path_exists(self.irreps_node_z, self.irreps_edge, "0o"):
                    gate_ir = "0o"
                else:
                    raise ValueError(
                        f"irreps={irreps} times irreps_edge={self.irreps_edge} "
                        f"cannot produce gates for irreps_gated={irreps_gated}."
                    )
            else:
                gate_ir = None
            irreps_gates = o3.Irreps(
                [(mul, gate_ir) for mul, _ in irreps_gated]
            ).simplify()
            gate = Gate(
                irreps_scalars,
                [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = DM2Interaction(
                irreps_in=irreps,
                irreps_node=self.irreps_node_z,
                irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in,
                radial_neurons=radial_neurons,
                num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.hidden_irreps = irreps
        self.out = o3.FullyConnectedTensorProduct(
            irreps,
            self.irreps_node_z,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        if self.use_condition:
            condition_embed_dim = max(1, irreps.dim // 4)
            self.condition_embedding = GaussianBasisEmbedding(
                num_basis=condition_num_basis,
                embedding_dim=condition_embed_dim,
                min_value=condition_min,
                max_value=condition_max,
                min_sigma=condition_min_sigma,
            )
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_embed_dim, condition_embed_dim),
                nn.Silu(),
                nn.Linear(condition_embed_dim, irreps.dim),
            )
        else:
            self.condition_embedding = None
            self.condition_projection = None

    def _node_condition(
        self,
        data: Data,
        condition: Optional[paddle.Tensor],
        num_nodes: int,
    ) -> Optional[paddle.Tensor]:
        if not self.use_condition:
            return None
        if condition is None:
            condition = getattr(data, self.condition_key, None)
        if condition is None:
            raise ValueError(
                f"DM2 conditional denoiser expects condition '{self.condition_key}'."
            )
        condition = condition.reshape([-1, 1]).astype("float32")
        condition_embedding = self.condition_embedding(condition)

        batch = getattr(data, "batch", None)
        if batch is None:
            condition_embedding = condition_embedding[:1].expand([num_nodes, -1])
        else:
            condition_embedding = condition_embedding[batch]
        return self.condition_projection(condition_embedding)

    def forward(
        self,
        data: Data,
        condition: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr[:, :3]
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        condition_embedding = self._node_condition(data, condition, h_node_x.shape[0])
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge,
            edge_attr,
            normalize=True,
            normalization="component",
        )
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            if condition_embedding is not None:
                h_node_x = h_node_x + condition_embedding
        return self.out(h_node_x, h_node_z)


class RattleParticles(nn.Layer):
    """Apply Gaussian position noise and store the target displacement ``dx``."""

    def __init__(self, sigma_max: float, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _sample_sigma(self, shape, dtype):
        if self.sigma_min >= self.sigma_max:
            return paddle.full(shape, self.sigma_max, dtype=dtype)
        return paddle.empty(shape, dtype=dtype).uniform_(
            min=self.sigma_min,
            max=self.sigma_max,
        )

    def forward(self, data: Data) -> Data:
        batch = getattr(data, "batch", None)
        if batch is None:
            sigma = self._sample_sigma([1], data.pos.dtype)
            sigma = sigma.expand([data.pos.shape[0]]).unsqueeze(axis=-1)
        else:
            num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1
            sigma = self._sample_sigma([num_graphs], data.pos.dtype)
            sigma = sigma[batch].unsqueeze(axis=-1)

        eps = paddle.randn(data.pos.shape, dtype=data.pos.dtype)
        data.dx = sigma * eps
        data.pos = data.pos + data.dx

        if getattr(data, "edge_attr", None) is not None:
            src, dst = data.edge_index[0], data.edge_index[1]
            data.edge_attr = data.edge_attr + data.dx[dst] - data.dx[src]

        data.sigma = sigma
        data.eps = eps
        return data


class DownselectEdges(nn.Layer):
    """Keep only edges whose displacement length is within ``cutoff``."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, data: Data) -> Data:
        edge_length = paddle.linalg.norm(data.edge_attr[:, :3], axis=1)
        edge_ids = paddle.nonzero(edge_length <= self.cutoff).flatten()
        data.edge_index = paddle.index_select(data.edge_index, edge_ids, axis=1)
        data.edge_attr = paddle.index_select(data.edge_attr, edge_ids, axis=0)
        return data


def _get_graph_from_batch(batch) -> Data:
    if isinstance(batch, Data):
        return batch
    if isinstance(batch, dict):
        for key in ("dm2_graph", "graph", "data"):
            graph = batch.get(key)
            if isinstance(graph, Data):
                return graph
    raise TypeError(
        "DM2 expects a geometric Data/Batch object or a dict containing "
        "'dm2_graph', 'graph', or 'data'."
    )


def _get_condition_from_batch(
    batch,
    graph: Data,
    condition_key: str,
) -> Optional[paddle.Tensor]:
    if hasattr(graph, condition_key):
        return getattr(graph, condition_key)
    if isinstance(batch, dict) and condition_key in batch:
        return batch[condition_key]
    return None


def _clone_graph(data: Data) -> Data:
    return data.clone() if hasattr(data, "clone") else copy.deepcopy(data)


def _apply_position_update(data: Data, new_pos: paddle.Tensor) -> Data:
    delta = new_pos - data.pos
    data.pos = new_pos
    if getattr(data, "edge_attr", None) is not None:
        src, dst = data.edge_index[0], data.edge_index[1]
        data.edge_attr = data.edge_attr + delta[dst] - delta[src]
    return data


def _get_lattice(data: Data, graph_id: int = 0) -> paddle.Tensor:
    lattice = getattr(data, "lattice", None)
    if lattice is None:
        lattice = getattr(data, "cell", None)
    if lattice is None:
        return paddle.eye(3, dtype=data.pos.dtype)
    if lattice.ndim == 2:
        return lattice
    return lattice[graph_id]


def _graph_to_structure_arrays(data: Data) -> List[Dict[str, np.ndarray]]:
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = paddle.zeros([data.pos.shape[0]], dtype="int64")
    num_graphs = int(batch.max()) + 1 if batch.numel() > 0 else 1

    atom_types = getattr(data, "atomic_numbers", None)
    if atom_types is None:
        atom_types = data.x.astype("int64") + 1

    results = []
    for graph_id in range(num_graphs):
        node_ids = paddle.nonzero(batch == graph_id).flatten()
        pos = paddle.index_select(data.pos, node_ids, axis=0)
        lattice = _get_lattice(data, graph_id).astype(pos.dtype)
        frac_coords = pos @ paddle.linalg.inv(lattice)
        frac_coords = frac_coords - paddle.floor(frac_coords)
        graph_atom_types = paddle.index_select(atom_types, node_ids, axis=0)
        results.append(
            {
                "frac_coords": frac_coords.numpy(),
                "atom_types": graph_atom_types.numpy().astype("int64"),
                "lattice": lattice.numpy(),
            }
        )
    return results


class DM2(nn.Layer):
    """DM2 diffusion-style denoising model for disordered materials.

    Training follows the official DM2 demos: random Gaussian ``rattle`` noise is
    added to atom positions, short edges are selected, and the denoiser learns to
    predict the displacement target. Sampling iteratively denoises a supplied
    noisy/random periodic structure.
    """

    def __init__(
        self,
        denoiser_cfg: Dict,
        cutoff: float = 5.0,
        sigma_min: float = 0.001,
        sigma_max: float = 0.75,
        loss_weight: float = 1.0,
        condition_key: str = "cooling_rate",
    ):
        super().__init__()
        denoiser_cfg = dict(denoiser_cfg)
        denoiser_cfg.setdefault("cutoff", cutoff)
        denoiser_cfg.setdefault("condition_key", condition_key)
        self.denoiser = DM2NequIPDenoiser(**denoiser_cfg)
        self.cutoff = cutoff
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.loss_weight = loss_weight
        self.condition_key = condition_key
        self.rattle_particles = RattleParticles(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.downselect_edges = DownselectEdges(cutoff)

    def forward(self, batch, **kwargs):
        graph = _clone_graph(_get_graph_from_batch(batch))
        condition = _get_condition_from_batch(batch, graph, self.condition_key)
        graph = self.rattle_particles(graph)
        graph = self.downselect_edges(graph)
        pred_dx = self.denoiser(graph, condition=condition)

        loss_dx = F.mse_loss(pred_dx, graph.dx)
        loss = self.loss_weight * loss_dx
        return {
            "loss_dict": {
                "loss": loss,
                "loss_dx": loss_dx,
            },
            "pred_dict": {
                "dx": pred_dx,
            },
        }

    @paddle.no_grad()
    def sample(
        self,
        batch_data,
        num_inference_steps: int = 100,
        final_relax_steps: int = 0,
        max_sigma_for_denoising: Optional[float] = None,
        return_trajectory: bool = False,
        **kwargs,
    ):
        graph = _clone_graph(_get_graph_from_batch(batch_data))
        condition = _get_condition_from_batch(batch_data, graph, self.condition_key)
        max_sigma = (
            self.sigma_max
            if max_sigma_for_denoising is None
            else max_sigma_for_denoising
        )

        trajectory = []
        sigmas = paddle.linspace(max_sigma, self.sigma_min, num_inference_steps)
        for sigma in sigmas:
            sigma_value = float(sigma.item())
            noisy_graph = _clone_graph(graph)
            noisy_graph = RattleParticles(
                sigma_min=sigma_value,
                sigma_max=sigma_value,
            )(noisy_graph)
            noisy_graph = self.downselect_edges(noisy_graph)
            pred_dx = self.denoiser(noisy_graph, condition=condition)
            graph = noisy_graph
            graph = _apply_position_update(graph, noisy_graph.pos - pred_dx)
            if return_trajectory:
                trajectory.append(_graph_to_structure_arrays(graph))

        for _ in range(final_relax_steps):
            work_graph = self.downselect_edges(_clone_graph(graph))
            pred_dx = self.denoiser(work_graph, condition=condition)
            graph = work_graph
            graph = _apply_position_update(graph, graph.pos - pred_dx)
            if return_trajectory:
                trajectory.append(_graph_to_structure_arrays(graph))

        result = {"result": _graph_to_structure_arrays(graph)}
        if return_trajectory:
            result["trajectory"] = trajectory
        return result
