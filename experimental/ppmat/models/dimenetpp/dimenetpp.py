from functools import partial
from typing import Callable
from typing import Optional

import paddle
import sympy as sym
from paddle.nn.functional import swish

from ppmat.models.common.basis_utils import bessel_basis
from ppmat.models.common.basis_utils import real_sph_harm
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.scatter import scatter

"""This module is adapted from https://github.com/Open-Catalyst-Project/ocp/tree/master/ocpmodels/models
"""


class Envelope(paddle.nn.Layer):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(y=p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(
            x.dtype
        )


class BesselBasisLayer(paddle.nn.Layer):
    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = paddle.create_parameter(
            shape=paddle.empty(
                shape=[
                    num_radial,
                ]
            ).shape,
            dtype=paddle.empty(
                shape=[
                    num_radial,
                ]
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(
                    shape=[
                        num_radial,
                    ]
                )
            ),
        )
        self.freq.stop_gradient = False

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        dist = dist.unsqueeze(axis=-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, True, True, True)
        self.sph_funcs = []
        self.bessel_funcs = []
        x, theta = sym.symbols("x theta")
        modules = {"sin": paddle.sin, "cos": paddle.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.zeros_like(x=x) + sph

    def forward(
        self, dist: paddle.Tensor, angle: paddle.Tensor, idx_kj: paddle.Tensor
    ) -> paddle.Tensor:
        dist = dist / self.cutoff
        rbf = paddle.stack(x=[f(dist) for f in self.bessel_funcs], axis=1)
        rbf = self.envelope(dist).unsqueeze(axis=-1) * rbf
        cbf = paddle.stack(x=[f(angle) for f in self.sph_funcs], axis=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].reshape([-1, n, k]) * cbf.reshape([-1, n, 1])).reshape(
            [-1, n * k]
        )
        return out


class EmbeddingBlock(paddle.nn.Layer):
    def __init__(
        self, num_embeddings, num_radial: int, hidden_channels: int, act: Callable
    ):
        super().__init__()
        self.act = act
        self.emb = paddle.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=hidden_channels
        )
        self.lin_rbf = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels
        )
        self.lin = paddle.nn.Linear(
            in_features=3 * hidden_channels, out_features=hidden_channels
        )

    def forward(
        self, x: paddle.Tensor, rbf: paddle.Tensor, i: paddle.Tensor, j: paddle.Tensor
    ) -> paddle.Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(paddle.concat(x=[x[i], x[j], rbf], axis=-1)))


class ResidualLayer(paddle.nn.Layer):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin2 = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act
        self.lin_rbf1 = paddle.nn.Linear(
            in_features=num_radial, out_features=basis_emb_size, bias_attr=False
        )
        self.lin_rbf2 = paddle.nn.Linear(
            in_features=basis_emb_size, out_features=hidden_channels, bias_attr=False
        )
        self.lin_sbf1 = paddle.nn.Linear(
            in_features=num_spherical * num_radial,
            out_features=basis_emb_size,
            bias_attr=False,
        )
        self.lin_sbf2 = paddle.nn.Linear(
            in_features=basis_emb_size, out_features=int_emb_size, bias_attr=False
        )
        self.lin_kj = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin_ji = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.lin_down = paddle.nn.Linear(
            in_features=hidden_channels, out_features=int_emb_size, bias_attr=False
        )
        self.lin_up = paddle.nn.Linear(
            in_features=int_emb_size, out_features=hidden_channels, bias_attr=False
        )
        self.layers_before_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
            ]
        )
        self.lin = paddle.nn.Linear(
            in_features=hidden_channels, out_features=hidden_channels
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
            ]
        )

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf
        x_kj = self.act(self.lin_down(x_kj))
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.shape[0])
        x_kj = self.act(self.lin_up(x_kj))
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return h


class OutputPPBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act
        self.lin_rbf = paddle.nn.Linear(
            in_features=num_radial, out_features=hidden_channels, bias_attr=False
        )
        self.lin_up = paddle.nn.Linear(
            in_features=hidden_channels, out_features=out_emb_channels, bias_attr=True
        )
        self.lins = paddle.nn.LayerList()
        for _ in range(num_layers):
            self.lins.append(
                paddle.nn.Linear(
                    in_features=out_emb_channels, out_features=out_emb_channels
                )
            )
        self.lin = paddle.nn.Linear(
            in_features=out_emb_channels, out_features=out_channels, bias_attr=False
        )

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNetPlusPlus(paddle.nn.Layer):
    """
    Unified DimeNet++: integrates base architecture and data wrapping.
    Supports both direct tensor inputs or data dict with on-the-fly graph.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_embeddings: int = 95,
        num_radial: int = 6,
        otf_graph: bool = False,
        cutoff: float = 10.0,
        max_num_neighbors: int = 20,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        readout: str = "mean",
        property_names: Optional[str] = "formation_energy_per_atom",
        data_norm_mean: float = 0.0,
        data_norm_std: float = 1.0,
        data_clip_min: float = -6.0,
        loss_type: str = "l1_loss",
        act=swish,
    ):
        super().__init__()
        # store hyperparams
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph
        self.readout = readout
        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            assert isinstance(property_names, str)
            self.property_names = property_names
        self.register_buffer(
            tensor=paddle.to_tensor(data_norm_mean), name="data_norm_mean"
        )
        self.register_buffer(
            tensor=paddle.to_tensor(data_norm_std), name="data_norm_std"
        )
        self.register_buffer(
            tensor=paddle.to_tensor(data_clip_min), name="data_clip_min"
        )

        # basis layers
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        # embedding and blocks
        self.emb = EmbeddingBlock(num_embeddings, num_radial, hidden_channels, act)
        self.output_blocks = paddle.nn.LayerList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.interaction_blocks = paddle.nn.LayerList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index
        value = paddle.arange(1, row.shape[0] + 1, dtype="int64")
        # build matrix of edge ids per target
        n = col.shape[0]
        rows = paddle.arange(n).unsqueeze(1)
        cols = paddle.arange(n).unsqueeze(0)
        mask = (col.unsqueeze(1) == col.unsqueeze(0)) & (cols <= rows)
        col_ = mask.astype("int64").sum(axis=1) - 1
        mat = paddle.scatter_nd(
            paddle.stack([col, col_], axis=1),
            value,
            shape=[num_nodes, col_.max().item() + 1],
        )
        idx_kj = mat[row][mat[row] > 0] - 1
        tmp = paddle.nonzero(mat[row], as_tuple=False)
        idx_ji = tmp[:, 0]
        idx_k = row[idx_kj]
        idx_j = row[idx_ji]
        idx_i = col[idx_ji]
        mask2 = idx_i != idx_k
        return (
            col,
            row,
            idx_i[mask2],
            idx_j[mask2],
            idx_k[mask2],
            idx_kj[mask2],
            idx_ji[mask2],
        )

    def normalize(self, tensor):
        return (tensor - self.data_norm_mean) / self.data_norm_std

    def unnormalize(self, tensor):
        return tensor * self.data_norm_std + self.data_norm_mean

    def _forward(self, data):
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        data["graph"] = data["graph"].tensor()

        # unpack graph dict
        graph = data["graph"]
        batch = graph.graph_node_id
        lattices = graph.node_feat["lattice"]
        pos = graph.node_feat["cart_coords"]
        frac = graph.node_feat["frac_coords"]
        edge_index = graph.edges
        to_jimages = graph.edge_feat["pbc_offset"]
        num_atoms = graph.node_feat["num_atoms"]
        num_bonds = graph.edge_feat["num_edges"]
        atom_types = graph.node_feat["atom_types"]

        out = get_pbc_distances(
            frac,
            edge_index.T,
            lattices,
            to_jimages,
            num_atoms,
            num_bonds,
            return_offsets=True,
        )
        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]
        j, i, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=atom_types.shape[0]
        )
        # compute angles
        pos_i = pos[idx_i]
        pos_j = pos[idx_j]
        pos_ji = pos_j - pos_i + offsets[idx_ji]
        pos_kj = pos[idx_k] - pos_j + offsets[idx_kj]
        a = (pos_ji * pos_kj).sum(axis=-1)
        b = paddle.cross(pos_ji, pos_kj).norm(axis=-1)
        angle = paddle.atan2(b, a)

        # basis expansions
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(atom_types, rbf, i, j)

        # output and interactions
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.shape[0])
        for interact, out_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interact(x, rbf, sbf, idx_kj, idx_ji)
            P += out_block(x, rbf, i, num_nodes=pos.shape[0])

        # readout
        energy = scatter(P, batch, dim=0, reduce=self.readout)
        return energy

    def forward(self, data, return_loss=True, return_prediction=True):
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(
                input=pred,
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_names] = pred
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, graphs):
        if isinstance(graphs, list):
            results = []
            for graph in graphs:
                result = self._forward(
                    {
                        "graph": graph,
                    }
                )
                result = self.unnormalize(result).numpy()[0, 0]
                result = {self.property_name: result}
                results.append(result)
            return results

        else:
            data = {
                "graph": graphs,
            }
            result = self._forward(data)
            result = self.unnormalize(result).numpy()[0, 0]
            result = {self.property_name: result}
            return result
