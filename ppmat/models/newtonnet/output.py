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

import paddle
from les import Les

from .scatter import scatter


def get_output_by_string(key, n_features=None, activation=None):
    if key == "energy":
        output_layer = EnergyOutput(n_features, activation)
    elif key == "gradient_force":
        output_layer = GradientForceOutput()
    elif key == "direct_force":
        output_layer = DirectForceOutput(n_features, activation)
    elif key == "virial":
        output_layer = VirialOutput()
    elif key == "stress":
        output_layer = StressOutput()
    elif key == "charge":
        output_layer = ChargeOutput(n_features, activation)
    elif key == "bec":
        output_layer = BornEffectiveChargeOutput()
    else:
        raise NotImplementedError(f"Output type {key} is not implemented yet")
    return output_layer


def get_aggregator_by_string(key):
    if key == "energy":
        aggregator = EnergyAggregator()
    elif key == "gradient_force":
        aggregator = NullAggregator()
    elif key == "direct_force":
        aggregator = NullAggregator()
    elif key == "hessian":
        aggregator = NullAggregator()
    elif key == "virial":
        aggregator = NullAggregator()
    elif key == "stress":
        aggregator = NullAggregator()
    elif key == "charge":
        aggregator = NullAggregator()
    elif key == "bec":
        aggregator = NullAggregator()
    else:
        raise NotImplementedError(f"Aggregate type {key} is not implemented yet")
    return aggregator


class CustomOutputSet:
    def __init__(self, **outputs):
        for key, value in outputs.items():
            setattr(self, key, value)


class DirectProperty(paddle.nn.Module):
    def __init__(self):
        super().__init__()


class DerivativeProperty(paddle.nn.Module):
    def __init__(self):
        super().__init__()
        self.create_graph = False

    def _save_grad(self, outputs):
        outputs.pos_grad, outputs.displacement_grad = paddle.grad(
            outputs=outputs.energy,
            inputs=(outputs.pos, outputs.displacement),
            grad_outputs=paddle.ones_like(outputs.energy),
            create_graph=self.create_graph,
            retain_graph=self.create_graph,
        )


class SecondDerivativeProperty(DerivativeProperty):
    def __init__(self):
        super().__init__()


class EnergyOutput(DirectProperty):
    """
    Energy prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
    """

    def __init__(self, n_features, activation):
        super().__init__()
        self.layers = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, 1),
        )

    def forward(self, outputs):
        energy = self.layers(outputs.atom_node)
        return energy


class GradientForceOutput(DerivativeProperty):
    """
    Gradient force prediction
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        if not hasattr(outputs, "pos_grad"):
            super()._save_grad(outputs)
        force = -outputs.pos_grad
        return force


class DirectForceOutput(DirectProperty):
    """
    Direct force prediction
    """

    def __init__(self, n_features, activation):
        super().__init__()
        self.layers = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, n_features),
        )

    def forward(self, outputs):
        force = self.layers(outputs.atom_node).unsqueeze(1) * outputs.force_node
        force = force.sum(dim=-1)
        return force


class VirialOutput(DerivativeProperty):
    """
    Virial prediction
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        if not hasattr(outputs, "displacement_grad"):
            super()._save_grad(outputs)
        virial = -outputs.displacement_grad
        return virial


class StressOutput(DerivativeProperty):
    """
    Stress prediction
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        if not hasattr(outputs, "displacement_grad"):
            super()._save_grad(outputs)
        virial = outputs.displacement_grad
        volume = outputs.cell.det().view(-1, 1, 1)
        stress = virial / volume
        return stress


class ChargeOutput(DirectProperty):
    """
    Charge prediction

    Parameters:
        n_features (int): Number of features in the hidden layer.
        activation (nn.Module): Activation function.
    """

    def __init__(self, n_features, activation):
        super().__init__()
        self.layers = paddle.nn.Sequential(
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, n_features),
            activation,
            paddle.compat.nn.Linear(n_features, 1),
        )

    def forward(self, outputs):
        charge = self.layers(outputs.atom_node)
        return charge


class BornEffectiveChargeOutput(SecondDerivativeProperty):
    """
    Born effective charge prediction
    """

    def __init__(self):
        super().__init__()
        self.les = Les()
        del self.les.atomwise
        del self.les.ewald

    def forward(self, outputs):
        bec = self.les(
            positions=outputs.pos,
            cell=outputs.cell,
            latent_charges=outputs.charge,
            batch=outputs.batch,
            compute_energy=False,
            compute_bec=True,
        )["BEC"]
        return bec


class EnergyAggregator(paddle.nn.Module):
    def __init__(self):
        super().__init__()
        self.les = Les()
        del self.les.atomwise
        del self.les.bec

    def forward(self, energy, outputs):
        if hasattr(outputs, "charge"):
            energy_sr = scatter(energy, outputs.batch, dim=0, reduce="sum").reshape(-1)
            energy_lr = self.les(
                positions=outputs.pos,
                cell=outputs.cell,
                latent_charges=outputs.charge,
                batch=outputs.batch,
                compute_energy=True,
                compute_bec=False,
            )["E_lr"]
            return energy_sr + energy_lr
        else:
            energy = scatter(energy, outputs.batch, dim=0, reduce="sum").reshape(-1)
            return energy


class NullAggregator(paddle.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, outputs):
        return output
