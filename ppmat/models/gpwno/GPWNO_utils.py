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

import paddle

from ppmat.utils.scatter import scatter

from ..common.e3nn import o3
from ..common.e3nn.nn import Activation
from ..common.e3nn.nn import Extract
from ..common.e3nn.nn import FullyConnectedNet


class ScalarActivation(paddle.nn.Module):
    """
    Use the invariant scalar features to gate higher order equivariant features.
    Adapted from `e3nn.nn.Gate`.
    """

    def __init__(self, irreps_in, act_scalars, act_gates):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_gates: gate activation function (for higher order features)
        """
        super(ScalarActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_spherical = len(self.irreps_in)
        irreps_scalars = self.irreps_in[0:1]
        irreps_gates = irreps_scalars * (self.num_spherical - 1)
        irreps_gated = self.irreps_in[1:]
        self.act_scalars = Activation(irreps_scalars, [act_scalars])
        self.act_gates = Activation(
            irreps_gates, [act_gates] * (self.num_spherical - 1)
        )
        self.extract = Extract(
            self.irreps_in,
            [irreps_scalars, irreps_gated],
            instructions=[(0,), tuple(range(1, self.irreps_in.lmax + 1))],
        )
        self.mul = o3.ElementwiseTensorProduct(irreps_gates, irreps_gated)

    def forward(self, features):
        scalars, gated = self.extract(features)
        scalars_out = self.act_scalars(scalars)
        if gated.shape[-1]:
            gates = self.act_gates(scalars.repeat(1, self.num_spherical - 1))
            gated_out = self.mul(gates, gated)
            features = paddle.cat([scalars_out, gated_out], dim=-1)
        else:
            features = scalars_out
        return features


class NormActivation(paddle.nn.Module):
    """
    Use the norm of the higher order equivariant features to gate themselves.
    Idea from the TFN paper.
    """

    def __init__(
        self,
        irreps_in,
        act_scalars=paddle.nn.functional.silu,
        act_vectors=paddle.sigmoid,
    ):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_vectors: vector activation function (for the norm of higher order features)
        """
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul
        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])
        self.inner_prod = o3.TensorProduct(
            self.vector_irreps,
            self.vector_irreps,
            inner_out,
            [(i, i, i, "uuu", False) for i in range(len(self.vector_irreps))],
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features):
        scalars = self.act_scalars(features[..., : self.scalar_idx])
        vectors = features[..., self.scalar_idx :]
        norm = paddle.sqrt(self.inner_prod(vectors, vectors) + 1e-08)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return paddle.cat([scalars, vectors_out], dim=-1)


class GCNLayer(paddle.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_edge,
        radial_embed_size,
        num_radial_layer,
        radial_hidden_size,
        is_fc=True,
        use_sc=True,
        irrep_normalization="component",
        path_normalization="element",
        *args,
        **kwargs
    ):
        """
        A single InfGCN layer for Tensor Product-based message passing.
        If the tensor product is fully connected, we have (for every path)

        .. math::
            z_w=\\sum_{uv}w_{uvw}x_u\\otimes y_v=\\sum_{u}w_{uw}x_u \\otimes y

        Else, we have

        .. math::
            z_u=x_u\\otimes \\sum_v w_{uv}y_v=w_u (x_u\\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input, and output, respectively.
        Notice that in our model, the second input is always the spherical harmonics of the edge vector,
        so the index v can be safely ignored.

        :param irreps_in: irreducible representations of input node features
        :param irreps_out: irreducible representations of output node features
        :param irreps_edge: irreducible representations of edge features
        :param radial_embed_size: embedding size of the edge length
        :param num_radial_layer: number of hidden layers in the radial network
        :param radial_hidden_size: hidden size of the radial network
        :param is_fc: whether to use fully connected tensor product
        :param use_sc: whether to use self-connection
        :param irrep_normalization: representation normalization passed to the `o3.FullyConnectedTensorProduct`
        :param path_normalization: path normalization passed to the `o3.FullyConnectedTensorProduct`
        """
        super(GCNLayer, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        self.is_fc = is_fc
        self.use_sc = use_sc
        if self.is_fc:
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            instr = [
                (i_1, i_2, i_out, "uvu", True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            self.tp = o3.TensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                instr,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        self.fc = FullyConnectedNet(
            [radial_embed_size]
            + num_radial_layer * [radial_hidden_size]
            + [self.tp.weight_numel],
            paddle.nn.functional.silu,
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce="sum")
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out

    def forward_mean(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce="mean")
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out
