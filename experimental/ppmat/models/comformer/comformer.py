# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from typing import Optional

import paddle
import paddle.nn as nn

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


class iComformer(nn.Layer):
    """Complete and Efficient Graph Transformers for Crystal Material Property
    Prediction,  https://arxiv.org/pdf/2403.11857

    Args:
        conv_layers (int, optional): The number of ComformerConv layers.
            Defaults to 4.
        edge_layers (int, optional): The number of ComformerConv_edge layers.
            Defaults to 1.
        atom_input_features (int, optional): The dimension of input features.
            Defaults to 92.
        edge_features (int, optional): The dimension of edge feature. Defaults to 256.
        triplet_input_features (int, optional): The dimension of triplet feature.
            Defaults to 256.
        node_features (int, optional): The dimension of node feature. Defaults to 256.
        fc_features (int, optional): The input dimension of the fully connected layer.
            Defaults to 256.
        output_features (int, optional): The output dimension. Defaults to 1.
        node_layer_head (int, optional): Heads of the node layer. Defaults to 1.
        edge_layer_head (int, optional): Heads of the edge layer. Defaults to 1.
        property_names (Optional[List[str]|str], optional):  Property name of the
            prediction data. Defaults to "formation_energy_per_atom".
    """

    def __init__(
        self,
        conv_layers: int = 4,
        edge_layers: int = 1,
        atom_input_features: int = 92,
        edge_features: int = 256,
        triplet_input_features: int = 256,
        node_features: int = 256,
        fc_features: int = 256,
        output_features: int = 1,
        node_layer_head: int = 1,
        edge_layer_head: int = 1,
        property_names: Optional[List[str] | str] = "formation_energy_per_atom",
    ):
        super().__init__()
        self.conv_layers = conv_layers
        self.edge_layers = edge_layers
        self.atom_input_features = atom_input_features
        self.edge_features = edge_features
        self.triplet_input_features = triplet_input_features
        self.node_features = node_features
        self.fc_features = fc_features
        self.output_features = output_features
        self.node_layer_head = node_layer_head
        self.edge_layer_head = edge_layer_head
        if isinstance(property_names, str):
            self.property_names = property_names
        elif isinstance(property_names, list):
            assert len(property_names) == 1, "property_names must be a list of length 1"
            self.property_names = property_names[0]

        self.atom_embedding = nn.Linear(
            in_features=self.atom_input_features, out_features=self.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=self.edge_features),
            nn.Linear(in_features=self.edge_features, out_features=self.node_features),
            nn.Softplus(),
        )
        self.rbf_angle = nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=self.triplet_input_features),
            nn.Linear(
                in_features=self.triplet_input_features, out_features=self.node_features
            ),
            nn.Softplus(),
        )
        self.att_layers = nn.LayerList(
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
            in_channels=self.edge_features,
            out_channels=self.node_features,
            heads=self.edge_layer_head,
            edge_dim=self.node_features,
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.node_features, out_features=self.fc_features),
            nn.Silu(),
        )
        self.fc_out = nn.Linear(
            in_features=self.fc_features, out_features=self.output_features
        )

    def forward(self, data) -> paddle.Tensor:
        #  The data in data['graph'] is numpy.ndarray, convert it to paddle.Tensor
        data["graph"] = data["graph"].tensor()

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
        node_features = self.att_layers[0](node_features, edges, edge_features)

        edge_features = self.edge_update_layer(
            edge_features, edge_nei_len, edge_nei_angle
        )
        node_features = self.att_layers[1](node_features, edges, edge_features)
        node_features = self.att_layers[2](node_features, edges, edge_features)
        node_features = self.att_layers[3](node_features, edges, edge_features)
        features = scatter(node_features, batch_idx, dim=0, reduce="mean")
        features = self.fc(features)

        result = {}
        result[self.property_names] = self.fc_out(features)
        return result
