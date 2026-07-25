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

from __future__ import annotations

from typing import Sequence

from paddle import Tensor, nn

from ppmat.models.chgnet.chgnet import aggregate
from ppmat.models.matterchat.utils.activations import ScaledSiLU
from ppmat.models.matterchat.utils.activations import get_activation as find_activation


class MLP(nn.Layer):
    """Multi-Layer Perceptron with configurable activation.

    Kept local because the framework CHGNet MLP hardcodes paddle.nn.Silu().
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int | Sequence[int] | None = (64, 64),
        dropout: float = 0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_dim in (None, 0):
            layers = [nn.Dropout(dropout), nn.Linear(input_dim, output_dim)]
        elif type(hidden_dim) == int:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                find_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            ]
        elif isinstance(hidden_dim, Sequence):
            layers = [nn.Linear(input_dim, hidden_dim[0]), find_activation(activation)]
            if len(hidden_dim) != 1:
                for h_in, h_out in zip(hidden_dim[0:-1], hidden_dim[1:]):
                    layers.append(nn.Linear(h_in, h_out))
                    layers.append(find_activation(activation))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim[-1], output_dim))
        else:
            raise TypeError(
                f"{hidden_dim=} must be an integer, a list of integers, or None."
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        return self.layers(X)


class GatedMLP(nn.Layer):
    """Gated MLP (core * sigmoid(gate)). Used in CGCNN and M3GNet."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | list[int] | None = None,
        dropout=0,
        activation="silu",
        norm="batch",
    ) -> None:
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.activation = find_activation(activation)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)

    def forward(self, X: Tensor) -> Tensor:
        if self.norm is None:
            core = self.activation(self.mlp_core(X))
            gate = self.sigmoid(self.mlp_gate(X))
        else:
            core = self.activation(self.bn1(self.mlp_core(X)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(X)))
        return core * gate


def find_normalization(name: str, dim: int = None) -> nn.Layer | None:
    if name is None:
        return None
    return {
        "batch": nn.BatchNorm1D(dim),
        "layer": nn.LayerNorm(dim),
    }.get(name.lower(), None)


__all__ = [
    "MLP",
    "GatedMLP",
    "aggregate",
    "find_normalization",
    "find_activation",
    "ScaledSiLU",
]
