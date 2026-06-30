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
from paddle import nn
from paddle.nn import Embedding
from paddle.nn import Linear

from ppmat.models.common.initializer import glorot_orthogonal_
from ppmat.models.common.spherical_fourier_bessel import AngleEmbedding
from ppmat.models.common.spherical_fourier_bessel import DistEmbedding
from ppmat.models.common.spherical_fourier_bessel import TorsionEmbedding
from ppmat.utils.scatter import scatter_sum
from ppmat.models.common.xyz_utils import xyz_to_dat


def swish(x):
    return x * paddle.nn.functional.sigmoid(x)


class SphereNetEmbedding(paddle.nn.Layer):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super().__init__()
        self.dist_emb = DistEmbedding(num_radial, cutoff, envelope_exponent)
        self.angle_emb = AngleEmbedding(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.torsion_emb = TorsionEmbedding(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb


class ResidualLayer(paddle.nn.Layer):
    def __init__(self, hidden_channels, act=swish):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal_(self.lin1.weight, scale=1.0)
        self.lin1.bias.set_value(paddle.zeros_like(self.lin1.bias))
        glorot_orthogonal_(self.lin2.weight, scale=1.0)
        self.lin2.bias.set_value(paddle.zeros_like(self.lin2.bias))

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class EdgeInitializer(paddle.nn.Layer):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        act=swish,
        use_node_features=True,
        use_extra_node_feature=False,
    ):
        super().__init__()
        self.act = act
        self.use_node_features = use_node_features
        self.use_extra_node_feature = use_extra_node_feature
        if self.use_node_features:
            self.emb = Embedding(95, hidden_channels)
        else:
            self.node_embedding = paddle.create_parameter(
                shape=[hidden_channels],
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Normal(),
            )
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        if self.use_extra_node_feature:
            self.lin = Linear(5 * hidden_channels, hidden_channels)
        else:
            self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = Linear(num_radial, hidden_channels, bias_attr=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_node_features:
            self.emb.weight.data = paddle.uniform(
                shape=self.emb.weight.shape,
                dtype=self.emb.weight.dtype,
                min=-(3.0**0.5),
                max=3.0**0.5,
            )
        glorot_orthogonal_(self.lin_rbf_0.weight, scale=1.0)
        self.lin_rbf_0.bias.set_value(paddle.zeros_like(self.lin_rbf_0.bias))
        glorot_orthogonal_(self.lin.weight, scale=1.0)
        self.lin.bias.set_value(paddle.zeros_like(self.lin.bias))
        glorot_orthogonal_(self.lin_rbf_1.weight, scale=1.0)

    def forward(self, x, node_feature, emb_in, i, j):
        rbf, _, _ = emb_in
        if self.use_node_features:
            x = self.emb(x)
        else:
            x = self.node_embedding.unsqueeze(0).expand([x.shape[0], -1])
        if node_feature is not None and self.use_extra_node_feature:
            x = paddle.concat([x, node_feature], axis=1)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(paddle.concat([x[i], x[j], rbf0], axis=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        return e1, e2


class EdgeUpdate(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf1 = Linear(num_radial, basis_emb_size_dist, bias_attr=False)
        self.lin_rbf2 = Linear(basis_emb_size_dist, hidden_channels, bias_attr=False)
        self.lin_sbf1 = Linear(
            num_spherical * num_radial, basis_emb_size_angle, bias_attr=False
        )
        self.lin_sbf2 = Linear(basis_emb_size_angle, int_emb_size, bias_attr=False)
        self.lin_t1 = Linear(
            num_spherical * num_spherical * num_radial,
            basis_emb_size_torsion,
            bias_attr=False,
        )
        self.lin_t2 = Linear(basis_emb_size_torsion, int_emb_size, bias_attr=False)
        self.lin_rbf = Linear(num_radial, hidden_channels, bias_attr=False)

        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.lin_down = Linear(hidden_channels, int_emb_size, bias_attr=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias_attr=False)

        self.layers_before_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = nn.LayerList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal_(self.lin_rbf1.weight, scale=1.0)
        glorot_orthogonal_(self.lin_rbf2.weight, scale=1.0)
        glorot_orthogonal_(self.lin_sbf1.weight, scale=1.0)
        glorot_orthogonal_(self.lin_sbf2.weight, scale=1.0)
        glorot_orthogonal_(self.lin_t1.weight, scale=1.0)
        glorot_orthogonal_(self.lin_t2.weight, scale=1.0)

        glorot_orthogonal_(self.lin_kj.weight, scale=1.0)
        self.lin_kj.bias.set_value(paddle.zeros_like(self.lin_kj.bias))
        glorot_orthogonal_(self.lin_ji.weight, scale=1.0)
        self.lin_ji.bias.set_value(paddle.zeros_like(self.lin_ji.bias))

        glorot_orthogonal_(self.lin_down.weight, scale=1.0)
        glorot_orthogonal_(self.lin_up.weight, scale=1.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal_(self.lin.weight, scale=1.0)
        self.lin.bias.set_value(paddle.zeros_like(self.lin.bias))
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal_(self.lin_rbf.weight, scale=1.0)

    def forward(self, x, emb_in, idx_kj, idx_ji):
        rbf0, sbf, t = emb_in
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter_sum(x_kj, idx_ji, dim=0, dim_size=x1.shape[0])
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        return e1, e2


class NodeUpdate(paddle.nn.Layer):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ):
        super().__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = Linear(hidden_channels, out_emb_channels, bias_attr=True)
        self.lins = nn.LayerList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias_attr=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal_(self.lin_up.weight, scale=1.0)
        for lin in self.lins:
            glorot_orthogonal_(lin.weight, scale=1.0)
            lin.bias.set_value(paddle.zeros_like(lin.bias))
        if self.output_init == "zeros":
            self.lin.weight.set_value(paddle.zeros_like(self.lin.weight))
        if self.output_init == "GlorotOrthogonal":
            glorot_orthogonal_(self.lin.weight, scale=1.0)

    def forward(self, e, i, dim_size=None):
        _, e2 = e
        v = scatter_sum(e2, i, dim=0, dim_size=dim_size)
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class GraphUpdate(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, u, v, batch):
        u = u + scatter_sum(v, batch, dim=0)
        return u


class SphereNet(paddle.nn.Layer):
    """Spherical Message Passing for 3D Molecular Graphs.

    Ported from DIG SphereNet (https://github.com/divelab/DIG) to PaddlePaddle.

    Args:
        energy_and_force: If True, predict energy and compute forces via
            autograd.
        cutoff: Cutoff distance for interatomic interactions.
        num_layers: Number of building blocks.
        hidden_channels: Hidden embedding size.
        out_channels: Size of each output sample.
        int_emb_size: Embedding size for interaction triplets.
        basis_emb_size_dist: Basis transformation size for distance.
        basis_emb_size_angle: Basis transformation size for angle.
        basis_emb_size_torsion: Basis transformation size for torsion.
        out_emb_channels: Output embedding size for atoms.
        num_spherical: Number of spherical harmonics.
        num_radial: Number of radial basis functions.
        envelope_exponent: Shape of the smooth cutoff.
        num_before_skip: Residual layers before skip connection.
        num_after_skip: Residual layers after skip connection.
        num_output_layers: Linear layers for output blocks.
        act: Activation function.
        output_init: Output initialisation ('GlorotOrthogonal' or 'zeros').
        use_node_features: Use atomic number embedding.
        use_extra_node_feature: Use extra node features.
        extra_node_feature_dim: Dimension of extra node features.
    """

    def __init__(
        self,
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        output_init="GlorotOrthogonal",
        use_node_features=True,
        use_extra_node_feature=False,
        extra_node_feature_dim=1,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force
        self.use_extra_node_feature = use_extra_node_feature

        if use_extra_node_feature:
            self.extra_emb = Linear(extra_node_feature_dim, hidden_channels)

        self.init_e = EdgeInitializer(
            num_radial,
            hidden_channels,
            act,
            use_node_features=use_node_features,
            use_extra_node_feature=use_extra_node_feature,
        )
        self.init_v = NodeUpdate(
            hidden_channels,
            out_emb_channels,
            out_channels,
            num_output_layers,
            act,
            output_init,
        )
        self.init_u = GraphUpdate()
        self.emb_layer = SphereNetEmbedding(
            num_spherical, num_radial, self.cutoff, envelope_exponent
        )

        self.update_vs = nn.LayerList(
            [
                NodeUpdate(
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                    output_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_es = nn.LayerList(
            [
                EdgeUpdate(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size_dist,
                    basis_emb_size_angle,
                    basis_emb_size_torsion,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_us = nn.LayerList([GraphUpdate() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_extra_node_feature:
            glorot_orthogonal_(self.extra_emb.weight, scale=1.0)
            self.extra_emb.bias.set_value(paddle.zeros_like(self.extra_emb.bias))
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb_layer.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    def forward(self, z, pos, batch, node_feature=None, edge_index=None):
        """Pure tensor forward.

        Args:
            z: [num_nodes] atomic numbers (int64)
            pos: [num_nodes, 3] 3D positions
            batch: [num_nodes] batch assignment
            node_feature: optional [num_nodes, extra_dim] extra features
            edge_index: optional [2, num_edges] pre-computed edge indices.
                When provided, the radius graph is not built internally
                (recommended for production use).  When ``None``, the graph
                is built on-the-fly via :func:`radius_graph`.

        Returns:
            u: [num_graphs, out_channels] predicted properties
        """
        if self.use_extra_node_feature and node_feature is not None:
            extra_node_feature = self.extra_emb(node_feature)
        else:
            extra_node_feature = None

        if edge_index is None:
            from ppmat.datasets.graph_utils.spherenet_graph_utils import radius_graph as _build_edges

            if self.energy_and_force:
                edge_index = _build_edges(pos, batch, self.cutoff)
            else:
                with paddle.no_grad():
                    edge_index = _build_edges(pos, batch, self.cutoff)
        else:
            # edge_index was pre-built by dataset preprocessing
            pass

        num_nodes = z.shape[0]
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(
            pos, edge_index, num_nodes, use_torsion=True,
        )

        emb_out = self.emb_layer(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, extra_node_feature, emb_out, i, j)
        v = self.init_v(e, i, dim_size=num_nodes)
        u = self.init_u(
            paddle.zeros_like(scatter_sum(v, batch, dim=0)),
            v,
            batch,
        )

        for update_e, update_v, update_u in zip(
            self.update_es, self.update_vs, self.update_us
        ):
            e = update_e(e, emb_out, idx_kj, idx_ji)
            v = update_v(e, i, dim_size=num_nodes)
            u = update_u(u, v, batch)

        return u


class SphereNetPP(paddle.nn.Layer):
    """SphereNet wrapper for property prediction training pipeline.

    Adapts the pure-tensor SphereNet forward to the PaddleMaterials
    training interface: accepts a batch dict and returns loss/pred dicts.
    Label normalisation is delegated to the dataset/transform layer —
    this wrapper only handles unnormalisation for output predictions.
    """

    def __init__(
        self,
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act="swish",
        output_init="GlorotOrthogonal",
        use_node_features=True,
        use_extra_node_feature=False,
        extra_node_feature_dim=1,
        data_mean=0.0,
        data_std=1.0,
        property_name="mu",
        force_key="force",
    ):
        super().__init__()

        act_fn = swish if act == "swish" else swish

        self.energy_and_force = energy_and_force
        self.property_name = property_name
        self.force_key = force_key
        self.register_buffer(
            "data_mean", paddle.to_tensor(data_mean, dtype=paddle.get_default_dtype())
        )
        self.register_buffer(
            "data_std", paddle.to_tensor(data_std, dtype=paddle.get_default_dtype())
        )

        self.spherenet = SphereNet(
            energy_and_force=energy_and_force,
            cutoff=cutoff,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            int_emb_size=int_emb_size,
            basis_emb_size_dist=basis_emb_size_dist,
            basis_emb_size_angle=basis_emb_size_angle,
            basis_emb_size_torsion=basis_emb_size_torsion,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act_fn,
            output_init=output_init,
            use_node_features=use_node_features,
            use_extra_node_feature=use_extra_node_feature,
            extra_node_feature_dim=extra_node_feature_dim,
        )

    def _unnormalize(self, x):
        return x * self.data_std + self.data_mean

    def forward(self, data, return_loss=True, return_prediction=True):
        """Forward with PaddleMaterials dict interface.

        Args:
            data: Dict with 'z', 'pos', 'batch' keys and property key.
            return_loss: Whether to compute loss.
            return_prediction: Whether to return predictions.

        Returns:
            Dict with 'loss_dict' and 'pred_dict'.
        """
        z = data["z"]
        pos = data["pos"]
        batch = data["batch"]
        edge_index = data.get("edge_index", None)

        if self.energy_and_force:
            pos = pos.detach()
            pos.stop_gradient = False

        pred = self.spherenet(z, pos, batch, edge_index=edge_index)

        # Compute forces if needed (used in both loss and prediction paths)
        forces_pred = None
        if self.energy_and_force:
            # F = -dE/d(pos)
            # FIXME: Paddle's atan2/put_along_axis lack 2nd-order grad ops;
            # use create_graph=False to avoid NaN during force computation.
            grad = paddle.grad(pred.sum(), pos, create_graph=False, allow_unused=True)
            if grad is None or grad[0] is None:
                forces_pred = None
            else:
                forces_pred = -grad[0]

        loss_dict = {}
        if return_loss:
            label = data[self.property_name]
            loss = paddle.nn.functional.l1_loss(pred, label)
            loss_dict["loss"] = loss

            if self.energy_and_force and forces_pred is not None:
                forces_target = data[self.force_key]
                force_loss = paddle.nn.functional.l1_loss(forces_pred, forces_target)
                loss_dict["loss"] = loss + force_loss

        prediction = {}
        if return_prediction:
            pred_out = self._unnormalize(pred)
            prediction[self.property_name] = pred_out
            if self.energy_and_force:
                if forces_pred is not None:
                    prediction[self.force_key] = forces_pred.detach()
                else:
                    # eval with paddle.no_grad(): force graph unavailable
                    prediction[self.force_key] = paddle.zeros_like(pos)

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, graphs):
        """Inference interface.

        Accepts a single data dict, a list of data dicts (each with
        ``z``, ``pos``, ``batch`` and optionally ``edge_index``), or
        a ``pymatgen.Structure`` (automatically converted).
        """
        from pymatgen.core import Structure
        if isinstance(graphs, Structure):
            # Convert pymatgen Structure → molecular data dict
            atomic_nums = paddle.to_tensor(
                [el.Z for el in graphs.species], dtype=paddle.int64
            )
            pos = paddle.to_tensor(
                graphs.cart_coords, dtype=paddle.get_default_dtype()
            )
            batch = paddle.zeros([len(atomic_nums)], dtype=paddle.int64)
            graphs = {"z": atomic_nums, "pos": pos, "batch": batch}
        if isinstance(graphs, list):
            if any(isinstance(g, Structure) for g in graphs):
                converted = []
                for g in graphs:
                    atomic_nums = paddle.to_tensor(
                        [el.Z for el in g.species], dtype=paddle.int64
                    )
                    pos = paddle.to_tensor(
                        g.cart_coords, dtype=paddle.get_default_dtype()
                    )
                    batch = paddle.zeros([len(atomic_nums)], dtype=paddle.int64)
                    converted.append({"z": atomic_nums, "pos": pos, "batch": batch})
                graphs = converted
            results = []
            for g in graphs:
                pred = self.spherenet(
                    g["z"], g["pos"], g["batch"], edge_index=g.get("edge_index", None)
                )
                val = self._unnormalize(pred).numpy()[0, 0]
                results.append({self.property_name: val})
            return results
        else:
            pred = self.spherenet(
                graphs["z"], graphs["pos"], graphs["batch"],
                edge_index=graphs.get("edge_index", None),
            )
            val = self._unnormalize(pred)
            return {self.property_name: val}
