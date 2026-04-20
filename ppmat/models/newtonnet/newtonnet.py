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

from typing import List
from typing import Optional
from typing import Union

import paddle

from ...losses.newtonnet_loss import get_loss_by_string
from .activations import get_activation_by_string
from .output import CustomOutputSet
from .output import DerivativeProperty
from .output import get_aggregator_by_string
from .output import get_output_by_string
from .representations import EdgeEmbedding
from .scalers import get_scaler_by_string
from .scatter import scatter


def repeat_interleave(
    repeats: List[int],
    device: Optional[Union[paddle.CPUPlace, paddle.CUDAPlace]] = None,
) -> paddle.Tensor:
    outs = [paddle.full([n], i, device=device) for i, n in enumerate(repeats)]
    return paddle.concat(outs, axis=0)


class NewtonNet(paddle.nn.Module):
    """
    Molecular Newtonian Message Passing

    Parameters:
        cutoff (float): Cutoff radius for the edge embedding. Default: 5.0.
        n_features (int): Number of features in the latent layer. Default: 128.
        n_basis (int): Number of radial basis functions. Default: 20.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function. Default: 'swish'.
        layer_norm (bool): Whether to use layer normalization. Default: False.
        output_properties (list): The properties to predict. Default: [].
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        n_features: int = 128,
        n_basis: int = 20,
        n_interactions: int = 3,
        activation: str = "swish",
        layer_norm: bool = False,
        output_properties: list = [],
    ) -> None:
        super().__init__()
        config = {
            "energy": {"weight": 1.0, "mode": "mse"},
            "gradient_force": {"weight": 50.0, "mode": "mse"},
        }
        self.main_loss, self.eval_loss = get_loss_by_string(config)
        activation = get_activation_by_string(activation)
        self.embedding_layers = EmbeddingNet(
            cutoff=cutoff, n_features=n_features, n_basis=n_basis
        )
        self.interaction_layers = paddle.nn.ModuleList(
            [
                InteractionNet(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=activation,
                    layer_norm=layer_norm,
                )
                for _ in range(n_interactions)
            ]
        )
        self.output_properties = output_properties
        self.output_layers = paddle.nn.ModuleList()
        self.scalers = paddle.nn.ModuleList()
        self.aggregators = paddle.nn.ModuleList()
        for key in self.output_properties:
            output_layer = get_output_by_string(key, n_features, activation)
            self.output_layers.append(output_layer)
            if isinstance(output_layer, DerivativeProperty):
                self.embedding_layers.requires_dr = True
            scaler = get_scaler_by_string(key)
            self.scalers.append(scaler)
            aggregator = get_aggregator_by_string(key)
            self.aggregators.append(aggregator)

    def forward(self, data):
        """
        Network forward pass

        Parameters:
            data: The input data.

        Returns:
            outputs (dict): The outputs of the network.
        """
        z_shape = data["z"].shape
        data["z"] = data["z"].reshape(-1)
        data["pos"] = data["pos"].reshape(-1, 3)
        data["cell"] = data["cell"].reshape(-1, 3, 3)
        data["energy"] = data["energy"].reshape(-1)
        data["force"] = data["force"].reshape(-1, 3)
        z, pos, cell = data["z"], data["pos"], data["cell"]
        batch = repeat_interleave(
            paddle.full(z_shape[0], z_shape[1], dtype=paddle.int32)
        )
        batch = batch.to(dtype=paddle.int64)
        (
            atom_node,
            force_node,
            dir_edge,
            dist_edge,
            edge_index,
            displacement,
        ) = self.embedding_layers(z, pos, cell, batch)
        for interaction_layer in self.interaction_layers:
            atom_node, force_node = interaction_layer(
                atom_node, force_node, dir_edge, dist_edge, edge_index
            )
        outputs = CustomOutputSet(
            z=z,
            pos=pos,
            atom_node=atom_node,
            force_node=force_node,
            edge_index=edge_index,
            cell=cell,
            displacement=displacement,
            batch=batch,
        )
        for key, output_layer, scaler, aggregator in zip(
            self.output_properties, self.output_layers, self.scalers, self.aggregators
        ):
            output = output_layer(outputs)
            output = scaler(output, outputs)
            output = aggregator(output, outputs)
            setattr(outputs, key, output)

        # convert dict to object attribute
        inputs = CustomOutputSet(**data)
        outputs_data = outputs.__dict__
        outputs_data["loss_dict"] = {"loss": self.main_loss(outputs, inputs)}
        return outputs_data

    def train(self, mode=True):
        """
        Set the network to training mode
        """
        super().train(mode)
        for output_layer in self.output_layers:
            if isinstance(output_layer, DerivativeProperty):
                output_layer.create_graph = mode

    def update_by_train_loader(self, train_loader):
        from .scalers import set_scaler_by_string

        fit_scalers = {"fit_scale": True, "fit_shift": True}
        stats_calc = MolecularStatistics()
        batch_size = len(train_loader.dataset)
        data_loader = paddle.io.DataLoader(
            train_loader.dataset,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            batch_size=batch_size,
        )
        for batch in data_loader:
            stats = stats_calc(batch)
            break

        for key, scaler in zip(self.output_properties, self.scalers):
            set_scaler_by_string(key, scaler, stats, **fit_scalers.pop(key, {}))


class MolecularStatistics(paddle.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        stats = {}
        z = data["z"]
        z_shape = z.shape
        z = z.reshape(-1)
        z_unique = z.unique()

        from ppmat.models.newtonnet.newtonnet import repeat_interleave

        batch = repeat_interleave(
            paddle.full(shape=[z_shape[0]], fill_value=z_shape[1], dtype=paddle.int64)
        )

        try:
            energy_org = data["energy"].cpu().reshape(-1)
            energy = energy_org
            z_hot = paddle.nn.functional.one_hot(z)
            z_hot = z_hot.to(dtype=paddle.int32)
            batch = batch.to(dtype=paddle.int32)
            formula = scatter(z_hot, batch, dim=0).to(energy.dtype)
            energy_reshape = False
            if energy.ndim == 1:
                energy_reshape = True
                energy = energy.reshape(-1, 1)
            # gelsd is supported in cpu
            if paddle.device.get_device() == "cpu":
                solution = paddle.linalg.lstsq(x=formula, y=energy, driver="gelsd")[0]
            else:
                current_device = paddle.device.get_device()
                paddle.set_device("cpu")
                solution = paddle.linalg.lstsq(
                    x=formula.to(device="cpu"),
                    y=energy.to(device="cpu"),
                    driver="gelsd",
                )[0]
                paddle.set_device(current_device)
                solution = solution.to(device=current_device)
            if energy_reshape:
                solution = solution.reshape(-1)
                energy = energy_org

            energy_shifts = paddle.zeros(
                118 + 1, dtype=energy.dtype, device=energy.device
            )

            energy_shifts[z_unique] = solution[z_unique]
            stds = (
                (energy - paddle.matmul(formula, solution)).square().sum()
                / formula.sum()
            ).sqrt()
            energy_scale = paddle.ones(
                118 + 1, dtype=energy.dtype, device=energy.device
            )
            energy_scale[z_unique] = stds
            stats["energy"] = {"shift": energy_shifts, "scale": energy_scale}
        except AttributeError as e:
            print(e)
            pass
        try:
            force = data["force"].norm(dim=-1).cpu()
            force = force.reshape(-1)
            means = scatter(force, z, reduce="mean")
            force_scale = paddle.ones(118 + 1, dtype=force.dtype, device=force.device)
            force_scale[z_unique] = means[z_unique]
            stats["force"] = {"scale": force_scale}
        except AttributeError:
            pass
        return stats


class EmbeddingNet(paddle.nn.Module):
    """
    Embedding layer of the network

    Parameters:
        cutoff (float): Cutoff radius for the edge embedding.
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
    """

    def __init__(self, cutoff, n_features, n_basis):
        super().__init__()
        self.n_features = n_features
        # weight = paddle.create_parameter(shape=[118 + 1, n_features],
        # dtype=paddle.float64, default_initializer=paddle.nn.initializer.Normal())
        self.node_embedding = paddle.nn.Embedding(118 + 1, n_features, padding_idx=0)
        self.edge_embedding = EdgeEmbedding(cutoff=cutoff, n_basis=n_basis)
        self.requires_dr = False

    def forward(self, z, pos, cell, batch):
        atom_node = self.node_embedding(z)
        force_node = paddle.zeros(
            z.size(0), 3, self.n_features, dtype=pos.dtype, device=pos.device
        )
        displacement = paddle.zeros_like(cell)
        displacement[:, 0, 0] = 1.0
        displacement[:, 1, 1] = 1.0
        displacement[:, 2, 2] = 1.0
        if self.requires_dr:
            pos.stop_gradient = not True
            displacement.stop_gradient = not True
        symmetric_displacement = (displacement + displacement.transpose(-1, -2)) / 2
        pos_displaced = paddle.bmm(
            pos.unsqueeze(1), symmetric_displacement[batch]
        ).squeeze(1)
        cell_displaced = paddle.bmm(cell, symmetric_displacement).squeeze(1)
        dist_edge, dir_edge, edge_index = self.edge_embedding(
            pos_displaced, cell_displaced, batch
        )
        return (atom_node, force_node, dir_edge, dist_edge, edge_index, displacement)


class InteractionNet(paddle.nn.Module):
    """
    Message passing layer of the network

    Parameters:
        n_features (int): Number of features in the hidden layer.
        n_basis (int): Number of radial basis functions.
        activation (nn.Module): Activation function.
        layer_norm (bool): Whether to use layer normalization.
    """

    def __init__(self, n_features, n_basis, activation, layer_norm):
        super().__init__()
        self.n_features = n_features
        self.message_nodepart = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, n_features),
        )
        self.message_edgepart = paddle.compat.nn.Linear(n_basis, n_features, bias=False)
        self.equiv_message1 = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features, bias=False),
            activation,
            paddle.compat.nn.Linear(n_features, n_features, bias=False),
        )
        self.equiv_message2 = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features, bias=False),
            activation,
            paddle.compat.nn.Linear(n_features, n_features, bias=False),
        )
        self.equiv_update = paddle.compat.nn.Linear(n_features, n_features, bias=False)
        if layer_norm:
            self.layer_norm = paddle.nn.LayerNorm(n_features)
        else:
            self.layer_norm = None

    def forward(self, atom_node, force_node, dir_edge, dist_edge, edge_index):
        message_nodepart = self.message_nodepart(atom_node)
        message_edgepart = self.message_edgepart(dist_edge)
        message = (
            message_edgepart
            * message_nodepart[edge_index[0]]
            * message_nodepart[edge_index[1]]
        )
        inv_message1 = message
        inv_update1 = scatter(
            inv_message1, edge_index[0], dim=0, dim_size=atom_node.size(0)
        )
        atom_node = atom_node + inv_update1
        equiv_message1_invpart = self.equiv_message1(message).unsqueeze(1)
        equiv_message1_equivpart = dir_edge.unsqueeze(2)
        equiv_message1 = equiv_message1_invpart * equiv_message1_equivpart
        equiv_message2_invpart = self.equiv_message2(message).unsqueeze(1)
        equiv_message2_equivpart = force_node[edge_index[1]]
        equiv_message2 = equiv_message2_invpart * equiv_message2_equivpart
        force_update = scatter(
            equiv_message1 + equiv_message2,
            edge_index[0],
            dim=0,
            dim_size=force_node.size(0),
        )
        force_node = force_node + force_update
        inv_update2 = paddle.sum(force_node * self.equiv_update(force_node), dim=1)
        atom_node = atom_node + inv_update2
        if self.layer_norm is not None:
            atom_node = self.layer_norm(atom_node)
        return atom_node, force_node
