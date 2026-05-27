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

"""
CSPNet for MiAD model. Inherits core structure from ppmat.models.diffcsp.diffcsp.CSPNet.
Overrides gen_edges for block_diag edge construction and forward for pre-computed
time embeddings and flexible atom type (one-hot / discrete) handling.
"""

import paddle

from paddle_scatter import scatter
from ppmat.models.diffcsp.diffcsp import CSPNet as _DiffCSPCSPNet
from ppmat.models.miad.graph_utils import dense_to_sparse


class CSPNet(_DiffCSPCSPNet):
    """MiAD-specific CSPNet inheriting from DiffCSP CSPNet.

    Key differences from the parent:
    - gen_edges: uses paddle.block_diag + dense_to_sparse for fully-connected edges
    - forward: accepts pre-computed t_emb (time embedding) and handles both
      one-hot and discrete atom types
    - Removes prop_mlp from each CSPLayer (MiAD does not use property embeddings)
    """

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        cutoff=7.0,
        max_neighbors=20,
        model_name=None,
        **kwargs,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            num_classes=max_atoms,
        )
        for i in range(self.num_layers):
            layer = self._modules.get(f"csp_layer_{i}")
            if layer and hasattr(layer, 'prop_mlp'):
                del layer.prop_mlp

    def gen_edges(self, num_atoms, frac_coords):
        """Fully-connected graph via block diagonal adjacency matrix."""
        if self.edge_style == "fc":
            lis = [paddle.ones([n, n], dtype=num_atoms.dtype) for n in num_atoms]
            fc_graph = paddle.block_diag(lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])

    def forward(
        self, t, t_emb, atom_types, frac_coords, lattices, num_atoms, node2graph
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]
        # Handle both discrete atom types and one-hot encoding
        if atom_types.ndim > 1:
            if self.smooth:
                node_features = self.node_embedding(atom_types.cast("float32"))
            else:
                atom_indices = atom_types.argmax(axis=-1)
                node_features = self.node_embedding(atom_indices.cast("int64"))
        else:
            if self.smooth:
                node_features = self.node_embedding(atom_types.cast("float32"))
            else:
                node_features = self.node_embedding(atom_types - 1)

        t_per_atom = paddle.repeat_interleave(t_emb, num_atoms, axis=0)
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])

        if self.ip:
            lattice_out = paddle.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out
