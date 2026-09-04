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
from __future__ import annotations

import math

import paddle
from paddle import nn


class LiFlow(nn.Layer):
    """Runnable Paddle flow-matching migration of LiFlow.

    This model preserves the reference LiFlow inputs, periodic dual-endpoint
    message passing, temperature/time conditioning, and flow-matching objective.
    It is a compact equivariant message-passing migration, not an exact Paddle
    reproduction of the reference implementation's full DualPaiNN architecture.
    """

    def __init__(
        self,
        num_features: int = 64,
        num_radial_basis: int = 20,
        num_layers: int = 3,
        num_elements: int = 77,
        r_max: float = 5.0,
        r_offset: float = 0.5,
        ref_temp: float = 1000.0,
        prediction_mode: str = "velocity",
    ):
        super().__init__()
        if num_features % 2:
            raise ValueError("num_features must be even")
        if prediction_mode not in {"velocity", "data"}:
            raise ValueError("prediction_mode must be 'velocity' or 'data'")
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = float(r_max)
        self.r_offset = float(r_offset)
        self.ref_temp = float(ref_temp)
        self.prediction_mode = prediction_mode

        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, num_features), nn.Silu(), nn.Linear(num_features, num_features)
        )
        self.temp_embedding = nn.Sequential(
            nn.Linear(1, num_features), nn.Silu(), nn.Linear(num_features, num_features)
        )
        self.message_mlps = nn.LayerList(
            [
                nn.Sequential(
                    nn.Linear(num_features + num_radial_basis * 2, num_features),
                    nn.Silu(),
                    nn.Linear(num_features, num_features),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_mlps = nn.LayerList(
            [
                nn.Sequential(
                    nn.Linear(num_features * 2, num_features),
                    nn.Silu(),
                    nn.Linear(num_features, num_features),
                )
                for _ in range(num_layers)
            ]
        )
        self.vector_gates = nn.LayerList(
            [nn.Linear(num_features, 2) for _ in range(num_layers)]
        )
        self.output_gate = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Silu(),
            nn.Linear(num_features, 1),
        )

    def _radial_basis(self, distance):
        distance = paddle.clip(distance + self.r_offset, min=1e-6, max=self.r_max)
        frequencies = paddle.arange(1, self.num_radial_basis + 1, dtype=distance.dtype)
        frequencies = frequencies * math.pi / self.r_max
        basis = paddle.sin(distance.unsqueeze(-1) * frequencies) / distance.unsqueeze(-1)
        cutoff = 0.5 * (1.0 + paddle.cos(distance * math.pi / self.r_max))
        return basis * cutoff.unsqueeze(-1)

    def _periodic_edges(self, positions, lattice, batch_index):
        sources = []
        targets = []
        shifts = []
        for graph_id in range(int(lattice.shape[0])):
            atom_ids = paddle.nonzero(batch_index == graph_id).flatten()
            count = int(atom_ids.shape[0])
            if count < 2:
                continue
            local_src = paddle.arange(count).tile([count])
            local_dst = paddle.arange(count).unsqueeze(1).tile([1, count]).flatten()
            keep = local_src != local_dst
            local_src = local_src[keep]
            local_dst = local_dst[keep]
            delta = positions[atom_ids[local_dst]] - positions[atom_ids[local_src]]
            inv_lattice = paddle.linalg.inv(lattice[graph_id])
            fractional = paddle.matmul(delta, inv_lattice)
            image = -paddle.round(fractional)
            delta = delta + paddle.matmul(image, lattice[graph_id])
            distance = paddle.linalg.norm(delta, axis=-1)
            keep = distance < self.r_max
            sources.append(atom_ids[local_src[keep]])
            targets.append(atom_ids[local_dst[keep]])
            shifts.append(paddle.matmul(image[keep], lattice[graph_id]))
        if not sources:
            empty_index = paddle.empty([0], dtype="int64")
            return empty_index, empty_index, paddle.empty([0, 3], dtype=positions.dtype)
        return (
            paddle.concat(sources),
            paddle.concat(targets),
            paddle.concat(shifts),
        )

    def _velocity(self, batch_data):
        positions_1 = batch_data["positions_1"]
        source = positions_1 + batch_data.get("prior", paddle.zeros_like(positions_1))
        positions_2 = batch_data["positions_2"]
        num_atoms = batch_data["num_atoms"].reshape([-1]).astype("int64")
        batch_index = paddle.repeat_interleave(
            paddle.arange(num_atoms.shape[0], dtype="int64"), num_atoms
        )
        graph_time = batch_data["time"].reshape([-1, 1])
        atom_time = paddle.repeat_interleave(graph_time, num_atoms, axis=0)
        x_t = (1.0 - atom_time) * source + atom_time * positions_2
        lattice = batch_data["lattice"]
        src, dst, shifts = self._periodic_edges(x_t, lattice, batch_index)

        atom = self.atom_embedding(batch_data["elements"])
        graph_temp = batch_data["temp"].reshape([-1, 1]) / self.ref_temp
        scalar = atom + self.time_embedding(graph_time)[batch_index]
        scalar = scalar + self.temp_embedding(graph_temp)[batch_index]
        vector = (x_t - positions_1).unsqueeze(-1).tile([1, 1, self.num_features])

        if int(src.shape[0]) == 0:
            return vector.mean(axis=-1)
        delta_1 = positions_1[dst] + shifts - positions_1[src]
        delta_2 = x_t[dst] + shifts - x_t[src]
        dist_1 = paddle.linalg.norm(delta_1, axis=-1).clip(min=1e-6)
        dist_2 = paddle.linalg.norm(delta_2, axis=-1).clip(min=1e-6)
        unit_1 = delta_1 / dist_1.unsqueeze(-1)
        unit_2 = delta_2 / dist_2.unsqueeze(-1)
        radial = paddle.concat(
            [self._radial_basis(dist_1), self._radial_basis(dist_2)], axis=-1
        )

        for message_mlp, update_mlp, vector_gate in zip(
            self.message_mlps, self.update_mlps, self.vector_gates
        ):
            message = message_mlp(paddle.concat([scalar[dst], radial], axis=-1))
            aggregated = paddle.zeros_like(scalar)
            aggregated = paddle.scatter_nd_add(aggregated, src.unsqueeze(-1), message)
            scalar = scalar + update_mlp(paddle.concat([scalar, aggregated], axis=-1))
            gate = vector_gate(message)
            direction = gate[:, :1] * unit_1 + gate[:, 1:] * unit_2
            vector_message = direction.unsqueeze(-1) * message.unsqueeze(1)
            vector_update = paddle.zeros_like(vector)
            vector_update = paddle.scatter_nd_add(
                vector_update, src.unsqueeze(-1), vector_message
            )
            vector = vector + vector_update
        return (vector * self.output_gate(scalar).unsqueeze(1)).sum(axis=-1)

    def _forward(self, batch_data):
        return self._velocity(batch_data)

    def forward(self, batch_data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        prediction = self._forward(batch_data)

        loss_dict = {}
        if return_loss:
            target = batch_data["target"]
            if self.prediction_mode == "data":
                target = batch_data["positions_2"] - batch_data["positions_1"]
            atom_loss = paddle.sum((prediction - target) ** 2, axis=-1)
            loss_dict["loss"] = paddle.mean(atom_loss)

        pred_dict = {"target": prediction} if return_prediction else {}
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    @paddle.no_grad()
    def predict(self, batch_data):
        return {"target": self._forward(batch_data)}
