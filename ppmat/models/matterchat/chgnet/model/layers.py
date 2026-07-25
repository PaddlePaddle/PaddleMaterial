# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# Reference: raw-code/Model/chgnet_lib/model/layers.py
from __future__ import annotations

import paddle
from paddle import Tensor, nn

from ppmat.models.matterchat.chgnet.model.functions import (
    MLP,
    GatedMLP,
    aggregate,
    find_activation,
    find_normalization,
)


class AtomConv(nn.Layer):
    """Convolution layer to update atom features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0,
        activation: str = "silu",
        norm: str | None = None,
        use_mlp_out: bool = True,
        resnet: bool = True,
        gMLP_norm: str = "batch",
    ) -> None:
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_atom = GatedMLP(
            input_dim=2 * atom_fea_dim + bond_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=gMLP_norm,
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=atom_fea_dim, output_dim=atom_fea_dim, hidden_dim=0
            )
        self.atom_norm = find_normalization(name=norm, dim=atom_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        bond_weights: Tensor,
        atom_graph: Tensor,
        directed2undirected: Tensor,
    ) -> Tensor:
        center_atoms = paddle.index_select(atom_feas, 0, atom_graph[:, 0])
        nbr_atoms = paddle.index_select(atom_feas, 0, atom_graph[:, 1])
        bonds = paddle.index_select(bond_feas, 0, directed2undirected)
        messages = paddle.concat([center_atoms, bonds, nbr_atoms], axis=1)
        messages = self.twoBody_atom(messages)

        bond_weight = paddle.index_select(bond_weights, 0, directed2undirected)
        messages = messages * bond_weight

        new_atom_feas = aggregate(messages, atom_graph[:, 0], average=False)

        if self.use_mlp_out:
            new_atom_feas = self.mlp_out(new_atom_feas)
        if self.resnet:
            new_atom_feas += atom_feas
        if self.atom_norm is not None:
            new_atom_feas = self.atom_norm(new_atom_feas)
        return new_atom_feas


class BondConv(nn.Layer):
    """Convolution layer to update bond features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0,
        activation: str = "silu",
        norm: str | None = None,
        use_mlp_out: bool = True,
        resnet=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_mlp_out = use_mlp_out
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=bond_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=kwargs.pop("gMLP_norm", "batch"),
            activation=activation,
        )
        if self.use_mlp_out:
            self.mlp_out = MLP(
                input_dim=bond_fea_dim, output_dim=bond_fea_dim, hidden_dim=0
            )
        self.bond_norm = find_normalization(name=norm, dim=bond_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        bond_weights: Tensor,
        angle_feas: Tensor,
        bond_graph: Tensor,
    ) -> Tensor:
        center_atoms = paddle.index_select(atom_feas, 0, bond_graph[:, 0])
        bond_feas_i = paddle.index_select(bond_feas, 0, bond_graph[:, 1])
        bond_feas_j = paddle.index_select(bond_feas, 0, bond_graph[:, 2])
        total_fea = paddle.concat(
            [bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )
        bond_update = self.twoBody_bond(total_fea)

        bond_weights_i = paddle.index_select(bond_weights, 0, bond_graph[:, 1])
        bond_weights_j = paddle.index_select(bond_weights, 0, bond_graph[:, 2])
        bond_update = bond_update * bond_weights_i * bond_weights_j

        new_bond_feas = aggregate(
            bond_update, bond_graph[:, 1], average=False, num_owner=len(bond_feas)
        )

        if self.use_mlp_out:
            new_bond_feas = self.mlp_out(new_bond_feas)
        if self.resnet:
            new_bond_feas += bond_feas
        if self.bond_norm is not None:
            new_bond_feas = self.bond_norm(new_bond_feas)
        return new_bond_feas


class AngleUpdate(nn.Layer):
    """Update angle features."""

    def __init__(
        self,
        atom_fea_dim: int,
        bond_fea_dim: int,
        angle_fea_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0,
        activation: str = "silu",
        norm: str | None = None,
        resnet: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.resnet = resnet
        self.activation = find_activation(activation)
        self.twoBody_bond = GatedMLP(
            input_dim=atom_fea_dim + 2 * bond_fea_dim + angle_fea_dim,
            output_dim=angle_fea_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=kwargs.pop("gMLP_norm", "batch"),
            activation=activation,
        )
        self.angle_norm = find_normalization(norm, dim=angle_fea_dim)

    def forward(
        self,
        atom_feas: Tensor,
        bond_feas: Tensor,
        angle_feas: Tensor,
        bond_graph: Tensor,
    ) -> Tensor:
        center_atoms = paddle.index_select(atom_feas, 0, bond_graph[:, 0])
        bond_feas_i = paddle.index_select(bond_feas, 0, bond_graph[:, 1])
        bond_feas_j = paddle.index_select(bond_feas, 0, bond_graph[:, 2])
        total_fea = paddle.concat(
            [bond_feas_i, bond_feas_j, angle_feas, center_atoms], axis=1
        )

        new_angle_feas = self.twoBody_bond(total_fea)

        if self.resnet:
            new_angle_feas += angle_feas
        if self.angle_norm is not None:
            new_angle_feas = self.angle_norm(new_angle_feas)
        return new_angle_feas


class GraphPooling(nn.Layer):
    """Pool sub-graphs in a batched graph."""

    def __init__(self, average: bool = False) -> None:
        super().__init__()
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        output = aggregate(atom_feas, atom_owner, average=False)
        if self.average:
            bc = paddle.bincount(atom_owner.cast("int32"))
            bc = paddle.where(bc != 0, bc, paddle.ones([1], dtype=bc.dtype))
            output = (output.T / bc.cast(output.dtype)).T
        return output


class GraphAttentionReadOut(nn.Layer):
    """Multi-head attention read-out: atom_feas -> crystal_fea."""

    def __init__(
        self, atom_fea_dim: int, num_head: int = 3, hidden_dim: int = 32, average=False
    ) -> None:
        super().__init__()
        self.key = MLP(
            input_dim=atom_fea_dim, output_dim=num_head, hidden_dim=hidden_dim
        )
        self.softmax = nn.Softmax(axis=0)
        self.average = average

    def forward(self, atom_feas: Tensor, atom_owner: Tensor) -> Tensor:
        crystal_feas = []
        weights = self.key(atom_feas)
        bin_count = paddle.bincount(atom_owner)
        start_index = 0
        for n_atom in bin_count:
            atom_fea = atom_feas[start_index : start_index + n_atom, :]
            weight = self.softmax(weights[start_index : start_index + n_atom, :])
            crystal_fea = (atom_fea.T @ weight).reshape([-1])
            if self.average:
                crystal_fea /= n_atom.cast(crystal_fea.dtype)
            crystal_feas.append(crystal_fea)
            start_index += n_atom
        return paddle.stack(crystal_feas, axis=0)
