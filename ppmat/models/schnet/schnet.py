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

import math
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.utils.scatter import scatter


class ShiftedSoftplus(nn.Layer):
    """Shifted softplus used by SchNet."""

    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.softplus(x) - self.shift


class Dense(nn.Layer):
    """Linear layer with optional activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=bias)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.XavierUniform()(self.linear.weight)
        if self.linear.bias is not None:
            nn.initializer.Constant(0.0)(self.linear.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class GaussianRBF(nn.Layer):
    """Gaussian radial basis expansion."""

    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        start: float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.n_rbf = int(n_rbf)
        self.cutoff = float(cutoff)

        centers = paddle.linspace(start, cutoff, n_rbf)
        if n_rbf > 1:
            spacing = float(abs(centers[1] - centers[0]))
        else:
            spacing = max(float(cutoff - start), 1e-6)
        widths = paddle.full([n_rbf], spacing, dtype=centers.dtype)

        if trainable:
            self.centers = self.create_parameter(
                shape=centers.shape,
                dtype=centers.dtype,
                default_initializer=nn.initializer.Assign(centers),
            )
            self.widths = self.create_parameter(
                shape=widths.shape,
                dtype=widths.dtype,
                default_initializer=nn.initializer.Assign(widths),
            )
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, distances: paddle.Tensor) -> paddle.Tensor:
        distances = distances.unsqueeze(-1)
        coeff = -0.5 / (self.widths * self.widths + 1e-12)
        diff = distances - self.centers
        return paddle.exp(coeff * diff * diff)


class CosineCutoff(nn.Layer):
    """Behler-style cosine cutoff."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer("cutoff", paddle.to_tensor([float(cutoff)], dtype="float32"))

    def forward(self, distances: paddle.Tensor) -> paddle.Tensor:
        cut = 0.5 * (paddle.cos(math.pi * distances / self.cutoff) + 1.0)
        cut = cut * (distances < self.cutoff).astype(cut.dtype)
        return cut


class SchNetInteraction(nn.Layer):
    """SchNet interaction block aligned with schnetpack implementation."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        cutoff_fn: nn.Layer,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        if activation is None:
            activation = ShiftedSoftplus()
        self.cutoff_fn = cutoff_fn
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, n_filters, activation=None),
        )

    def forward(
        self,
        x: paddle.Tensor,
        f_ij: paddle.Tensor,
        idx_i: paddle.Tensor,
        idx_j: paddle.Tensor,
        distances: paddle.Tensor,
    ) -> paddle.Tensor:
        x = self.in2f(x)
        w_ij = self.filter_network(f_ij)
        w_ij = w_ij * self.cutoff_fn(distances).unsqueeze(-1)

        x_j = x[idx_j]
        x_ij = x_j * w_ij
        x = scatter(x_ij, idx_i, dim=0, dim_size=x.shape[0], reduce="sum")
        x = self.f2out(x)
        return x


class SchNet(nn.Layer):
    """SchNet model for small-molecule regression tasks in PaddleMaterials."""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 6,
        n_filters: Optional[int] = None,
        cutoff: float = 5.0,
        n_rbf: int = 50,
        max_z: int = 100,
        property_name: Union[str, list[str]] = "energy_per_atom",
        force_name: str = "force",
        enable_force_loss: bool = False,
        energy_weight: float = 0.005,
        force_weight: float = 0.995,
        readout: str = "sum",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "l1_loss",
        shared_interactions: bool = False,
        trainable_rbf: bool = False,
        atomref_path: Optional[str] = None,
        atomref_key: str = "atom_ref",
        atomref_index: Optional[int] = None,
    ):
        super().__init__()
        self.n_atom_basis = int(n_atom_basis)
        self.n_filters = int(n_filters or n_atom_basis)
        self.n_interactions = int(n_interactions)
        self.max_z = int(max_z)
        self.readout = str(readout).lower()

        if isinstance(property_name, list):
            self.property_name = property_name[0]
        else:
            self.property_name = str(property_name)
        self.force_name = str(force_name)
        self.enable_force_loss = bool(enable_force_loss)
        self.energy_weight = float(energy_weight)
        self.force_weight = float(force_weight)

        self.embedding = nn.Embedding(self.max_z + 1, self.n_atom_basis, padding_idx=0)
        nn.initializer.XavierUniform()(self.embedding.weight)

        self.radial_basis = GaussianRBF(
            n_rbf=n_rbf,
            cutoff=cutoff,
            trainable=trainable_rbf,
        )
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)

        activation = ShiftedSoftplus()
        if shared_interactions:
            shared_block = SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=n_rbf,
                n_filters=self.n_filters,
                cutoff_fn=self.cutoff_fn,
                activation=activation,
            )
            self.interactions = nn.LayerList([shared_block] * self.n_interactions)
        else:
            self.interactions = nn.LayerList(
                [
                    SchNetInteraction(
                        n_atom_basis=self.n_atom_basis,
                        n_rbf=n_rbf,
                        n_filters=self.n_filters,
                        cutoff_fn=self.cutoff_fn,
                        activation=activation,
                    )
                    for _ in range(self.n_interactions)
                ]
            )

        self.output_network = nn.Sequential(
            Dense(self.n_atom_basis, self.n_atom_basis // 2, activation=activation),
            Dense(self.n_atom_basis // 2, 1, activation=None),
        )

        self.register_buffer("data_mean", paddle.to_tensor(data_mean, dtype="float32"))
        self.register_buffer("data_std", paddle.to_tensor(data_std, dtype="float32"))
        if atomref_path is None:
            self.atomref = None
        else:
            atomref_np = np.load(atomref_path)[atomref_key]
            if atomref_index is not None and atomref_np.ndim >= 2:
                atomref_np = atomref_np[:, int(atomref_index)]
            if atomref_np.ndim == 1:
                atomref_np = atomref_np[:, None]
            if atomref_np.ndim != 2 or atomref_np.shape[1] != 1:
                raise ValueError(
                    "Loaded atomref must have shape [num_elements, 1] after indexing."
                )
            self.register_buffer(
                "atomref",
                paddle.to_tensor(atomref_np.astype("float32")),
            )

        if loss_type == "mse_loss":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def normalize(
        self,
        x: paddle.Tensor,
        atomref_bias: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if atomref_bias is not None:
            x = x - atomref_bias
        return (x - self.data_mean) / (self.data_std + 1e-12)

    def unnormalize(
        self,
        x: paddle.Tensor,
        atomref_bias: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        x = x * self.data_std + self.data_mean
        if atomref_bias is not None:
            x = x + atomref_bias
        return x

    def _as_tensor(self, x, dtype: str):
        if isinstance(x, paddle.Tensor):
            return x.astype(dtype)
        return paddle.to_tensor(x, dtype=dtype)

    def _prepare_graph_inputs(self, graph, force_from_pos: bool = False):
        if hasattr(graph, "tensor"):
            graph = graph.tensor()

        edges = self._as_tensor(graph.edges, "int64")
        idx_i = edges[:, 1]
        idx_j = edges[:, 0]

        atom_types = graph.node_feat.get("atom_types", None)
        if atom_types is None:
            feat = graph.node_feat.get("feat", None)
            if feat is None:
                raise KeyError(
                    "Graph must provide node_feat['atom_types'] or node_feat['feat']."
                )
            feat = self._as_tensor(feat, "float32")
            atom_types = paddle.argmax(feat, axis=-1)
        atom_types = self._as_tensor(atom_types, "int64")
        atom_types = paddle.clip(atom_types, min=0, max=self.max_z)

        if hasattr(graph, "graph_node_id"):
            graph_node_id = self._as_tensor(graph.graph_node_id, "int64")
            num_graphs = getattr(graph, "num_graph", int(graph_node_id.max()) + 1)
            num_graphs = int(num_graphs)
        else:
            graph_node_id = paddle.zeros([atom_types.shape[0]], dtype="int64")
            num_graphs = 1

        pos = graph.node_feat.get("cart_coords", None)
        if pos is not None:
            pos = self._as_tensor(pos, "float32")

        distances = None
        if not force_from_pos:
            distances = graph.edge_feat.get("bond_dist", None)
            if distances is not None:
                distances = self._as_tensor(distances, "float32")

        if distances is None:
            if pos is None:
                raise KeyError(
                    "Graph must provide edge_feat['bond_dist'] or node_feat['cart_coords']."
                )
            if force_from_pos:
                pos.stop_gradient = False
            rij = pos[idx_i] - pos[idx_j]
            # Use sqrt(sum(r^2)) instead of linalg.norm for 2nd-order grad stability
            # in force-supervised training.
            distances = paddle.sqrt(paddle.sum(rij * rij, axis=-1) + 1e-12)

        return atom_types, idx_i, idx_j, distances, graph_node_id, num_graphs, pos

    def _compute_atomref_bias(
        self,
        atom_types: paddle.Tensor,
        graph_node_id: paddle.Tensor,
        num_graphs: int,
    ) -> Optional[paddle.Tensor]:
        if self.atomref is None:
            return None
        max_index = int(self.atomref.shape[0]) - 1
        atom_types = paddle.clip(atom_types, min=0, max=max_index)
        atomref_per_atom = self.atomref[atom_types]
        return scatter(
            atomref_per_atom,
            graph_node_id,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )

    def _forward(self, data, force_from_pos: bool = False):
        graph = data["graph"]
        atom_types, idx_i, idx_j, distances, graph_node_id, num_graphs, pos = (
            self._prepare_graph_inputs(graph, force_from_pos=force_from_pos)
        )
        atomref_bias = self._compute_atomref_bias(atom_types, graph_node_id, num_graphs)

        x = self.embedding(atom_types)
        f_ij = self.radial_basis(distances)

        for interaction in self.interactions:
            x = x + interaction(x, f_ij, idx_i, idx_j, distances)

        atom_pred = self.output_network(x)

        if self.readout == "mean":
            pred = scatter(
                atom_pred,
                graph_node_id,
                dim=0,
                dim_size=num_graphs,
                reduce="mean",
            )
        else:
            pred = scatter(
                atom_pred,
                graph_node_id,
                dim=0,
                dim_size=num_graphs,
                reduce="sum",
            )

        return pred, pos, atomref_bias

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        use_force_loss = (
            self.enable_force_loss
            and return_loss
            and (self.force_name in data)
        )
        pred, pos, atomref_bias = self._forward(data, force_from_pos=use_force_loss)

        loss_dict = {}
        if return_loss:
            label = data[self.property_name]
            label = self._as_tensor(label, "float32")
            if label.ndim == 1:
                label = label.unsqueeze(-1)
            label = self.normalize(label, atomref_bias=atomref_bias)
            energy_loss = self.loss_fn(pred, label)
            loss = energy_loss
            loss_dict["energy_loss"] = energy_loss

            force_pred = None
            if use_force_loss:
                force_target = self._as_tensor(data[self.force_name], "float32")
                if force_target.ndim == 3:
                    force_target = force_target.reshape([-1, force_target.shape[-1]])
                force_pred = -paddle.grad(
                    outputs=[paddle.sum(self.unnormalize(pred, atomref_bias=atomref_bias))],
                    inputs=[pos],
                    create_graph=True,
                    retain_graph=True,
                )[0]
                force_loss = F.mse_loss(force_pred, force_target)
                loss = self.energy_weight * energy_loss + self.force_weight * force_loss
                loss_dict["force_loss"] = force_loss

            loss_dict["loss"] = loss

        pred_dict = {}
        if return_prediction:
            pred_dict[self.property_name] = self.unnormalize(
                pred,
                atomref_bias=atomref_bias,
            )
            if use_force_loss:
                pred_dict[self.force_name] = force_pred

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @paddle.no_grad()
    def predict(self, graphs):
        if isinstance(graphs, list):
            results = []
            for g in graphs:
                pred, _, atomref_bias = self._forward({"graph": g}, force_from_pos=False)
                pred = self.unnormalize(pred, atomref_bias=atomref_bias)
                val = float(pred.numpy().reshape([-1])[0])
                results.append({self.property_name: val})
            return results

        pred, _, atomref_bias = self._forward({"graph": graphs}, force_from_pos=False)
        pred = self.unnormalize(pred, atomref_bias=atomref_bias)
        val = float(pred.numpy().reshape([-1])[0])
        return {self.property_name: val}

    def predict_forces(self, data):
        pred, pos, atomref_bias = self._forward(data, force_from_pos=True)
        force_pred = -paddle.grad(
            outputs=[paddle.sum(self.unnormalize(pred, atomref_bias=atomref_bias))],
            inputs=[pos],
            create_graph=False,
            retain_graph=False,
        )[0]
        return force_pred
