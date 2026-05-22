# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMATG-specific CSPNet with knn graph, internal time embedding, dual outputs.

Reuses SinusoidsEmbedding and CSPLayer from diffcsp (ppmat.models.diffcsp.diffcsp).
"""

import paddle
import paddle.nn as nn

from ppmat.models.diffcsp.diffcsp import CSPLayer, SinusoidsEmbedding
from .utils import radius_graph_pbc, repeat_blocks


class OMATGCSPNet(nn.Layer):
    """OMATG-specific CSPNet extending diffcsp backbone with:
    - knn graph construction via radius_graph_pbc
    - internal time embedding (SinusoidalTimeEmbeddings)
    - dual output heads (coord_out_2, lattice_out_2, type_out_2)
    - species_shift and enable_masked_species() for DNG mode
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        time_embed_dim=None,
    ):
        super().__init__()
        self.ip = ip
        self.smooth = smooth
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.species_shift = 1

        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)

        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings

            self.time_embedder = SinusoidalTimeEmbeddings(time_embed_dim)
            self.atom_latent_emb = nn.Linear(
                hidden_dim + time_embed_dim, hidden_dim
            )
        else:
            self.time_embedder = None
            self.atom_latent_emb = nn.Linear(hidden_dim + 1, hidden_dim)

        self.act_fn = nn.Silu()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)
            dis_dim = num_freqs * 2 * 3
        else:
            self.dis_emb = None
            dis_dim = 0
        self.dis_dim = dis_dim

        for i in range(num_layers):
            self.add_sublayer(
                "csp_layer_%d" % i,
                CSPLayer(
                    hidden_dim,
                    prop_dim=hidden_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                ),
            )
        self.num_layers = num_layers

        self.coord_out = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias_attr=False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias_attr=False)

        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style

        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, max_atoms)
            self.type_out_2 = nn.Linear(hidden_dim, max_atoms)

        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def enable_masked_species(self):
        self.node_embedding = nn.Embedding(self.max_atoms + 1, self.hidden_dim)
        self.species_shift = 0

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat([tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms = mask_same_atoms & cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand([2, -1])].reshape([2, -1])
        edge_index_cat = paddle.concat(
            [
                edge_index_new,
                paddle.stack([edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            paddle.arange(neighbors.shape[0]), neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(batch_edge, minlength=neighbors.shape[0])
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )
        return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            cumsum = paddle.concat(
                [paddle.to_tensor([0]), paddle.cumsum(num_atoms, axis=0)]
            )
            rows_list, cols_list = [], []
            for i in range(len(num_atoms)):
                n = int(num_atoms[i].item())
                offset = int(cumsum[i].item())
                r = (
                    paddle.arange(n).unsqueeze(1).expand([n, n]).reshape([-1]) + offset
                )
                c = (
                    paddle.arange(n).unsqueeze(0).expand([n, n]).reshape([-1]) + offset
                )
                rows_list.append(r)
                cols_list.append(c)
            rows = paddle.concat(rows_list)
            cols = paddle.concat(cols_list)
            fc_edges = paddle.stack([rows, cols])
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

        elif self.edge_style == "knn":
            lattice_nodes = lattices[node2graph]
            cart_coords = paddle.einsum("bi,bij->bj", frac_coords, lattice_nodes)
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                None,
                None,
                num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=num_atoms.place,
                lattices=lattices,
            )
            j_index, i_index = edge_index[0], edge_index[1]
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors = distance_vectors + to_jimages
            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )
            return edge_index_new, -edge_vector_new

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types.cast("float32"))
        else:
            node_features = self.node_embedding(atom_types - self.species_shift)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        if self.time_embed_dim is not None:
            t_embed = self.time_embedder(t)
        else:
            t_embed = t

        if t_embed.ndim == 1:
            t_embed = t_embed.unsqueeze(0)
        t_per_atom = paddle.repeat_interleave(t_embed, num_atoms, axis=0)
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            layer = getattr(self, "csp_layer_%d" % i)
            node_features = layer(
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
        coord_out_2 = self.coord_out_2(node_features)

        graph_features = paddle.geometric.segment_mean(node_features, node2graph)

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.einsum("bij,bjk->bik", lattice_out, lattices)

        lattice_out_2 = self.lattice_out_2(graph_features)
        lattice_out_2 = lattice_out_2.reshape([-1, 3, 3])
        if self.ip:
            lattice_out_2 = paddle.einsum("bij,bjk->bik", lattice_out_2, lattices)

        if self.pred_type:
            type_out = self.type_out(node_features)
            type_out_2 = self.type_out_2(node_features)
            return (
                lattice_out,
                coord_out,
                type_out,
                lattice_out_2,
                coord_out_2,
                type_out_2,
            )

        return lattice_out, coord_out, lattice_out_2, coord_out_2

    def forward_dict(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        """Forward returning dict of velocity fields b and denoisers eta.

        Aligned with original omg Model.forward output keys:
        pos_b, pos_eta, cell_b, cell_eta, and species_b/species_eta for DNG.
        """
        preds = OMATGCSPNet.forward(
            self, t, atom_types, frac_coords, lattices, num_atoms, node2graph
        )
        if self.pred_scalar:
            return {"scalar": preds}
        if self.pred_type:
            cell_b, pos_b, species_b, cell_eta, pos_eta, species_eta = preds
            return {
                "pos_b": pos_b,
                "pos_eta": pos_eta,
                "cell_b": cell_b,
                "cell_eta": cell_eta,
                "species_b": species_b,
                "species_eta": species_eta,
            }
        cell_b, pos_b, cell_eta, pos_eta = preds
        return {
            "pos_b": pos_b,
            "pos_eta": pos_eta,
            "cell_b": cell_b,
            "cell_eta": cell_eta,
        }
