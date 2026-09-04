# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Faithful Paddle ports of liflow.model.layers (frozen commit e6fc475).

Attribute/parameter/buffer names intentionally match the reference so converted
checkpoints load without key rewriting:
  time_embedding.freqs / temp_embedding.freqs / radial_embedding.freqs,
  radial_embedding.prefactor / cutoff_fn.r_max / linear_* / mlp_* etc.
"""

import math

import paddle
from paddle import nn

from ppmat.utils.scatter import scatter_sum


class GaussianFourierBasis(nn.Layer):
    """Fixed random Gaussian Fourier embeddings (buffer, not learnable)."""

    def __init__(self, num_basis):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = paddle.randn([num_basis // 2]) * (2 * math.pi)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        args = self.freqs * x.unsqueeze(-1)
        return paddle.concat((paddle.sin(args), paddle.cos(args)), axis=-1)


class BesselBasis(nn.Layer):
    def __init__(self, num_basis, r_max):
        super().__init__()
        self.num_basis = num_basis
        freqs = paddle.arange(1, num_basis + 1, dtype="float32") * (math.pi / float(r_max))
        prefactor = paddle.to_tensor(math.sqrt(2.0 / float(r_max)), dtype="float32")
        self.register_buffer("freqs", freqs)
        self.register_buffer("prefactor", prefactor)

    def forward(self, x):
        args = self.freqs * x.unsqueeze(-1)
        return self.prefactor * paddle.sin(args) / x.unsqueeze(-1)


class CosineCutoff(nn.Layer):
    def __init__(self, r_max):
        super().__init__()
        self.register_buffer("r_max", paddle.to_tensor(float(r_max), dtype="float32"))

    def forward(self, x):
        x_cut = 0.5 * (1.0 + paddle.cos(x * math.pi / self.r_max))
        x_cut = x_cut * (x < self.r_max).astype(x.dtype)
        return x_cut


class DualMessageBlock(nn.Layer):
    def __init__(self, num_features, num_radial_basis):
        super().__init__()
        self.num_features = num_features
        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Silu(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s,  # [n_nodes, 1, F]
        v,  # [n_nodes, 3, F]
        radial_embeddings_1,  # [n_edges, 1, R]
        radial_embeddings_2,  # [n_edges, 1, R]
        f_cut_1,  # [n_edges, 1]
        f_cut_2,  # [n_edges, 1]
        unit_vectors_1,  # [n_edges, 3]
        unit_vectors_2,  # [n_edges, 3]
        edge_index,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1.unsqueeze(-1)
            + self.linear_W(radial_embeddings_2) * f_cut_2.unsqueeze(-1)
        )
        x = phi[idx_j] * W
        f = self.num_features
        x_s, x_vv, x_vs_1, x_vs_2 = paddle.split(x, [f, f, f, f], axis=-1)
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1.unsqueeze(-1)
            + x_vs_2 * unit_vectors_2.unsqueeze(-1)
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class UpdateBlock(nn.Layer):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.Silu(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias_attr=False)

    def forward(self, s, v):
        f = self.num_features
        U_v, V_v = paddle.split(self.linear_UV(v), 2, axis=-1)
        a = self.mlp_a(paddle.concat((s, V_v.norm(p=2, axis=-2, keepdim=True)), axis=-1))
        a_vv, a_sv, a_ss = paddle.split(a, [f, f, f], axis=-1)
        dv = a_vv * U_v
        ds = a_ss + a_sv * paddle.sum(U_v * V_v, axis=-2, keepdim=True)
        return s + ds, v + dv


class GatedEquivariantBlock(nn.Layer):
    """Modified gated equivariant block outputting a single vector."""

    def __init__(self, num_scalar_inputs, num_vector_inputs):
        super().__init__()
        self.num_scalar_inputs = num_scalar_inputs
        self.num_vector_inputs = num_vector_inputs
        self.linear_v = nn.Linear(num_vector_inputs, 2, bias_attr=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_scalar_inputs + 1, num_scalar_inputs + 1),
            nn.Silu(),
            nn.Linear(num_scalar_inputs + 1, 1),
        )

    def forward(self, s, v):
        W_v1, W_v2 = paddle.split(self.linear_v(v), 2, axis=-1)
        s_out = self.mlp_s(paddle.concat((s, W_v2.norm(axis=-2, keepdim=True)), axis=-1))
        v_out = W_v1 * s_out
        return v_out
