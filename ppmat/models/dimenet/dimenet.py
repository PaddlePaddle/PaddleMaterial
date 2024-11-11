from functools import partial
from typing import Callable

import paddle
import sympy as sym
from paddle.nn.functional import swish
from paddle.sparse import sparse_coo_tensor

from ppmat.models.common.scatter import scatter
from ppmat.models.dimenet.dimenet_utils import bessel_basis
from ppmat.models.dimenet.dimenet_utils import real_sph_harm
from ppmat.utils.crystal import get_pbc_distances

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
        sph_harm_forms = real_sph_harm(num_spherical)
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
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
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
    """DimeNet++ implementation based on https://github.com/klicperajo/dimenet.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_blocks,
        int_emb_size,
        basis_emb_size,
        out_emb_channels,
        num_spherical,
        num_embeddings,
        num_radial,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ):
        super(DimeNetPlusPlus, self).__init__()
        self.cutoff = cutoff
        if sym is None:
            raise ImportError("Package `sympy` could not be found.")
        self.num_blocks = num_blocks
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.emb = EmbeddingBlock(num_embeddings, num_radial, hidden_channels, act)
        self.output_blocks = paddle.nn.LayerList(
            sublayers=[
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
            sublayers=[
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

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index
        # sort_index = col.argsort()
        # row, col = row[sort_index], col[sort_index]

        value = paddle.arange(start=1, end=row.shape[0] + 1)

        indices = paddle.to_tensor([col, row])
        adj_t = sparse_coo_tensor(indices, value, (num_nodes, num_nodes))
        num_triplets = (adj_t.to_dense()[row] > 0).sum(axis=1)

        adj_t_row = adj_t.to_dense()[row].to_sparse_coo(2)
        adj_t_row_values = adj_t_row.values() - 1
        adj_t_row_rows = adj_t_row.indices()[0]
        adj_t_row_cols = adj_t_row.indices()[1]

        idx_i = col.repeat_interleave(repeats=num_triplets)
        idx_j = row.repeat_interleave(repeats=num_triplets)
        idx_k = adj_t_row_cols
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj = adj_t_row_values[mask]
        idx_ji = adj_t_row_rows[mask]
        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """"""
        raise NotImplementedError


class DimeNetPlusPlusWrap(DimeNetPlusPlus):
    def __init__(
        self,
        num_targets,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_embeddings=95,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        max_num_neighbors=20,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        readout="mean",
    ):
        self.num_targets = num_targets
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph
        self.readout = readout
        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_embeddings=num_embeddings,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, data):
        # batch = data["batch"]
        graph = data["graph"]
        batch = graph.graph_node_id

        lattices = graph.node_feat["lattice"]
        pos = graph.node_feat["cart_coords"]
        frac_coords = graph.node_feat["frac_coords"]
        edge_index = graph.edges
        to_jimages = graph.edge_feat["pbc_offset"]
        num_atoms = graph.node_feat["num_atoms"]
        num_bonds = graph.edge_feat["num_edges"]
        atom_types = graph.node_feat["atom_types"]

        out = get_pbc_distances(
            frac_coords,
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
        j, i = edge_index
        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=atom_types.shape[0]
        )
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i + offsets[idx_ji],
            pos[idx_k].detach() - pos_j + offsets[idx_kj],
        )
        a = (pos_ji * pos_kj).sum(axis=-1)
        b = paddle.cross(x=pos_ji, y=pos_kj).norm(axis=-1)
        angle = paddle.atan2(x=b, y=a)
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(atom_types, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.shape[0])
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.shape[0])
        if batch is None:
            if self.readout == "mean":
                energy = P.mean(axis=0)
            elif self.readout == "sum":
                energy = P.sum(axis=0)
            elif self.readout == "cat":
                energy = paddle.concat(x=[P.sum(axis=0), P.mean(axis=0)])
            else:
                raise NotImplementedError
        else:
            energy = scatter(P, batch, dim=0, reduce=self.readout)
        results = {}
        results["formation_energy_per_atom"] = energy
        return results

    @property
    def num_params(self):
        return sum(p.size for p in self.parameters())
