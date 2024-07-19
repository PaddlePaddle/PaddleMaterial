import math
import sys

import paddle
import paddle.nn as nn
from einops import repeat
from utils import paddle_aux

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(paddle.nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]  # .to(x.devices)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class CSPLayer(paddle.nn.Layer):
    """Message passing layer for cspnet."""

    def __init__(
        self, hidden_dim=128, act_fn=paddle.nn.Silu(), dis_emb=None, ln=False, ip=True
    ):
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=hidden_dim * 2 + 9 + self.dis_dim, out_features=hidden_dim
            ),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.prop_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=512, out_features=hidden_dim),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            x = lattices
            perm_0 = list(range(x.ndim))
            perm_0[-1] = -2
            perm_0[-2] = -1
            lattice_ips = lattices @ x.transpose(perm=perm_0)
        else:
            lattice_ips = lattices

        lattice_ips_flatten = lattice_ips.reshape([-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            x=[hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
        )
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = paddle.geometric.segment_mean(edge_features, edge_index[0])
        agg = paddle.concat(x=[node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
        num_atoms=None,
        property_emb=None,
        property_mask=None,
    ):
        # import pdb;pdb.set_trace()
        if property_emb is not None:
            property_features = self.prop_mlp(property_emb)
            if property_mask is not None:
                property_features = property_features * property_mask
            property_features = paddle.repeat_interleave(
                property_features, num_atoms, axis=0
            )
            node_features = node_features + property_features

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(paddle.nn.Layer):
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
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
    ):
        super(CSPNet, self).__init__()
        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = paddle.nn.Linear(
                in_features=max_atoms, out_features=hidden_dim
            )
        else:
            self.node_embedding = paddle.nn.Embedding(
                num_embeddings=max_atoms, embedding_dim=hidden_dim
            )
        self.atom_latent_emb = paddle.nn.Linear(
            in_features=hidden_dim + latent_dim, out_features=hidden_dim
        )
        if act_fn == "silu":
            self.act_fn = paddle.nn.Silu()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_sublayer(
                name="csp_layer_%d" % i,
                sublayer=CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        self.num_layers = num_layers
        self.coord_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=3, bias_attr=False
        )
        self.lattice_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=9, bias_attr=False
        )
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        if self.pred_type:
            self.type_out = paddle.nn.Linear(
                in_features=hidden_dim, out_features=MAX_ATOMIC_NUM
            )
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = paddle.nn.Linear(in_features=hidden_dim, out_features=1)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat(x=[tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            lis = [paddle.ones(shape=[n, n]) for n in num_atoms]
            cum_num_atoms = paddle.cumsum(x=num_atoms)
            indices_pp = []
            rows = paddle.arange(num_atoms.max())
            ind1, ind2 = paddle.meshgrid(rows, rows)
            index = paddle.stack(x=[ind1, ind2], axis=0)
            for n, cum_n in zip(num_atoms, cum_num_atoms):
                offset = cum_n - n
                indices_pp.append(index[:, :n, :n].reshape((2, -1)) + offset)
            indices_pp = paddle.concat(x=indices_pp, axis=1)
            fc_edges = indices_pp
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        else:
            raise NotImplementedError("Edge style '%s'" % self.edge_style)

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        property_emb=None,
        property_mask=None,
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(repeats=num_atoms, axis=0)
        node_features = paddle.concat(x=[node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = eval("self.csp_layer_%d" % i)(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
                num_atoms=num_atoms,
                property_emb=property_emb,
                property_mask=property_mask,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = paddle.geometric.segment_mean(node_features, node2graph)
        if self.pred_scalar:
            return self.scalar_out(graph_features)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out
        return lattice_out, coord_out
