from typing import Literal

import paddle

from ppmat.models.comformer.transformer import ComformerConv
from ppmat.models.comformer.transformer import ComformerConv_edge
from ppmat.models.comformer.utils import RBFExpansion
from ppmat.models.common.scatter import scatter


def bond_cosine(r1, r2):
    bond_cosine = paddle.sum(x=r1 * r2, axis=-1) / (
        paddle.linalg.norm(x=r1, axis=-1) * paddle.linalg.norm(x=r2, axis=-1)
    )
    bond_cosine = paddle.clip(x=bond_cosine, min=-1, max=1)
    return bond_cosine


class iComformer(paddle.nn.Layer):
    def __init__(
        self,
        conv_layers: int = 4,
        edge_layers: int = 1,
        atom_input_features: int = 92,
        edge_features: int = 256,
        triplet_input_features: int = 256,
        node_features: int = 256,
        fc_layers: int = 1,
        fc_features: int = 256,
        output_features: int = 1,
        node_layer_head: int = 1,
        edge_layer_head: int = 1,
        nn_based: bool = False,
        zero_inflated: bool = False,
        use_angle: bool = False,
        angle_lattice: bool = False,
        property_names: Literal[
            "band_gap", "formation_energy_per_atom"
        ] = "formation_energy_per_atom",
    ):
        super().__init__()
        self.conv_layers = conv_layers
        self.edge_layers = edge_layers
        self.atom_input_features = atom_input_features
        self.edge_features = edge_features
        self.triplet_input_features = triplet_input_features
        self.node_features = node_features
        self.fc_layers = fc_layers
        self.fc_features = fc_features
        self.output_features = output_features
        self.node_layer_head = node_layer_head
        self.edge_layer_head = edge_layer_head
        self.nn_based = nn_based
        self.zero_inflated = zero_inflated
        self.use_angle = use_angle
        self.angle_lattice = angle_lattice
        self.property_names = property_names

        self.atom_embedding = paddle.nn.Linear(
            in_features=self.atom_input_features, out_features=self.node_features
        )
        self.rbf = paddle.nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=self.edge_features),
            paddle.nn.Linear(
                in_features=self.edge_features, out_features=self.node_features
            ),
            paddle.nn.Softplus(),
        )
        self.rbf_angle = paddle.nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=self.triplet_input_features),
            paddle.nn.Linear(
                in_features=self.triplet_input_features, out_features=self.node_features
            ),
            paddle.nn.Softplus(),
        )
        self.att_layers = paddle.nn.LayerList(
            sublayers=[
                ComformerConv(
                    in_channels=self.node_features,
                    out_channels=self.node_features,
                    heads=self.node_layer_head,
                    edge_dim=self.node_features,
                )
                for _ in range(self.conv_layers)
            ]
        )
        self.edge_update_layer = ComformerConv_edge(
            in_channels=self.node_features,
            out_channels=self.node_features,
            heads=self.node_layer_head,
            edge_dim=self.node_features,
        )
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=self.node_features, out_features=self.fc_features
            ),
            paddle.nn.Silu(),
        )
        self.sigmoid = paddle.nn.Sigmoid()

        self.fc_out = paddle.nn.Linear(
            in_features=self.fc_features, out_features=self.output_features
        )

    def forward(self, data) -> paddle.Tensor:
        batch_idx = data["graph"].graph_node_id
        edges = data["graph"].edges.T.contiguous()

        node_features = self.atom_embedding(
            data["graph"].node_feat["node_feat"].cast("float32")
        )
        edge_feat = -0.75 / paddle.linalg.norm(x=data["graph"].edge_feat["r"], axis=1)
        edge_nei_len = -0.75 / paddle.linalg.norm(
            x=data["graph"].edge_feat["nei"], axis=-1
        )
        edge_nei_angle = bond_cosine(
            data["graph"].edge_feat["nei"],
            data["graph"].edge_feat["r"].unsqueeze(1).tile(repeat_times=[1, 3, 1]),
        )
        num_edge = tuple(edge_feat.shape)[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape([-1])).reshape([num_edge, 3, -1])
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape([-1])).reshape(
            [num_edge, 3, -1]
        )
        node_features = self.att_layers[0](
            node_features, edges, edge_features  # data['graph'].edges.T.contiguous(),
        )

        edge_features = self.edge_update_layer(
            edge_features, edge_nei_len, edge_nei_angle
        )
        node_features = self.att_layers[1](
            node_features, edges, edge_features  # data['graph'].edges.T.contiguous(),
        )
        node_features = self.att_layers[2](
            node_features, edges, edge_features  # data['graph'].edges.T.contiguous(),
        )
        node_features = self.att_layers[3](
            node_features, edges, edge_features  # data['graph'].edges.T.contiguous(),
        )
        features = scatter(node_features, batch_idx, dim=0, reduce="mean")
        features = self.fc(features)
        out = self.fc_out(features)
        result = {}
        result[self.property_names] = out
        return result
