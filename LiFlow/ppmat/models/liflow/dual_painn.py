# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Faithful Paddle port of liflow.model.models.DualPaiNN (commit e6fc475).

The reference network consumes already-prepared per-graph fields:
  positions_1 / positions_2 / edge_index / shifts / elements / time / temp
and never constructs graphs by itself. Single-graph convenience fields
(time/temp scalars, missing edge_index) are accepted for the PaddleMaterials
pipeline but are always resolved to the reference semantics internally.
"""

import paddle
from paddle import nn

from ppmat.models.liflow.layers import (
    BesselBasis,
    CosineCutoff,
    DualMessageBlock,
    GatedEquivariantBlock,
    GaussianFourierBasis,
    UpdateBlock,
)


def get_unit_vectors_and_lengths(positions, edge_index, shifts, eps=1e-8):
    idx_i, idx_j = edge_index[0], edge_index[1]
    vectors = positions[idx_j] - positions[idx_i] + shifts
    lengths = paddle.linalg.norm(vectors, axis=-1, keepdim=True)
    return vectors / (lengths + eps), lengths


class DualPaiNN(nn.Layer):
    def __init__(
        self,
        num_features=64,
        num_radial_basis=20,
        num_layers=3,
        num_elements=77,
        r_max=5.0,
        r_offset=0.0,
        ref_temp=1000.0,
    ):
        super().__init__()
        assert num_features % 2 == 0, "Number of features must be even"
        self.num_features = num_features
        self.num_radial_basis = num_radial_basis
        self.num_layers = num_layers
        self.r_max = float(r_max)
        self.r_offset = float(r_offset)
        self.ref_temp = float(ref_temp)

        self.atom_embedding = nn.Embedding(num_elements, num_features)
        self.time_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.temp_embedding = GaussianFourierBasis(num_basis=num_features // 2)
        self.radial_embedding = BesselBasis(num_basis=num_radial_basis, r_max=self.r_max)
        self.cutoff_fn = CosineCutoff(r_max=self.r_max)
        self.linear_v = nn.Linear(1, num_features, bias_attr=False)

        self.messages = nn.LayerList(
            [DualMessageBlock(num_features, num_radial_basis) for _ in range(num_layers)]
        )
        self.updates = nn.LayerList([UpdateBlock(num_features) for _ in range(num_layers)])
        self.output_block = GatedEquivariantBlock(
            num_scalar_inputs=num_features, num_vector_inputs=num_features
        )

    @staticmethod
    def _num_atoms(positions):
        return positions.shape[0]

    @staticmethod
    def _per_node(value, num_atoms, batch):
        """Expand a per-graph value to per-node, or pass a per-node value through."""
        value = paddle.reshape(value, [-1])
        if value.shape[0] == num_atoms:
            return value
        if value.shape[0] == 1:
            return paddle.full([num_atoms], value[0], dtype=value.dtype)
        # multi-graph: value[g] -> value[batch]
        return value[batch]

    def forward(self, batch_data):
        positions_1 = batch_data["positions_1"]
        positions_2 = batch_data["positions_2"]
        num_atoms = self._num_atoms(positions_1)
        batch = batch_data.get("batch")
        if batch is None:
            num_nodes_per_graph = batch_data.get("num_atoms")
            if num_nodes_per_graph is not None:
                n = num_nodes_per_graph.reshape([-1]).astype("int64")
                batch = paddle.repeat_interleave(paddle.arange(n.shape[0]), n)
            else:
                batch = paddle.zeros([num_atoms], dtype="int64")

        edge_index = batch_data.get("edge_index")
        if edge_index is None:
            sources, targets = [], []
            offset = 0
            counts = paddle.unique_consecutive(batch).shape[0]
            for g in range(counts):
                local = paddle.nonzero(batch == g).reshape([-1]).astype("int64")
                m = local.shape[0]
                src = local.tile([m])
                dst = local.unsqueeze(1).tile([1, m]).flatten()
                keep = src != dst
                sources.append(src[keep] + offset)
                targets.append(dst[keep] + offset)
                offset += m
            edge_index = paddle.stack([paddle.concat(sources), paddle.concat(targets)], axis=0)
        shifts = batch_data.get("shifts", paddle.zeros([edge_index.shape[1], 3], dtype=positions_1.dtype))

        unit_vectors_1, lengths_1 = get_unit_vectors_and_lengths(positions_1, edge_index, shifts)
        unit_vectors_2, lengths_2 = get_unit_vectors_and_lengths(positions_2, edge_index, shifts)

        r_max = paddle.to_tensor(self.r_max, dtype=positions_1.dtype)
        lengths_1 = paddle.clip(lengths_1 + self.r_offset, max=self.r_max)
        lengths_2 = paddle.clip(lengths_2 + self.r_offset, max=self.r_max)
        radial_embeddings_1 = self.radial_embedding(lengths_1)
        radial_embeddings_2 = self.radial_embedding(lengths_2)
        f_cut_1 = self.cutoff_fn(lengths_1)
        f_cut_2 = self.cutoff_fn(lengths_2)

        s_atom = self.atom_embedding(batch_data["elements"])
        time = self._per_node(batch_data["time"], num_atoms, batch)
        temp = self._per_node(batch_data["temp"], num_atoms, batch)
        s_time = self.time_embedding(time)
        s_temp = self.temp_embedding(temp / self.ref_temp)
        s = s_atom + paddle.concat([s_time, s_temp], axis=-1)
        s = s.unsqueeze(1)  # [n, 1, F]

        positions_diff = positions_2 - positions_1
        v = self.linear_v(positions_diff.unsqueeze(-1))  # [n, 3, F]

        for message, update in zip(self.messages, self.updates):
            s, v = message(
                s,
                v,
                radial_embeddings_1,
                radial_embeddings_2,
                f_cut_1,
                f_cut_2,
                unit_vectors_1,
                unit_vectors_2,
                edge_index,
            )
            s, v = update(s, v)
        v_out = self.output_block(s, v).squeeze(-1)
        return v_out
