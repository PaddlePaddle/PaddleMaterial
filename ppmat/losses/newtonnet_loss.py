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

from ..models.newtonnet.scatter import scatter


def get_loss_by_string(losses):
    """
    Get loss function by string

    Parameters:
        losses (dict): The loss function settings.
            key (str): The loss function type.
            value (dict): The loss function settings.
                weight (float): The weight for the loss function. Default: None.
                mode (str): The loss function to use. Default: 'mse'.
                    'mse': Mean squared error.
                    'mae': Mean absolute error.
                    'huber': Huber loss.
                transform (str): The transformation to apply to the data. Default: None.
                    'cos': 1 - cosine similarity.
                    'norm': Norm.

    Returns:
        main_loss (nn.Module): The main loss function for model
        training (back propagation) and validation (learning rate scheduling).
        eval_loss (nn.Module): The evaluation loss function
        for task-specific model evaluation.
    """
    main_losses = []
    eval_losses = []
    assert losses is not None, "losses is not defined."
    for key, kwargs in losses.items():
        if key == "energy":
            main_losses.append(EnergyLoss(**kwargs))
            eval_losses.append(EnergyLoss(mode="mae"))
            eval_losses.append(EnergyLoss(mode="mse"))
            eval_losses.append(EnergyPerAtomLoss(mode="mae"))
            eval_losses.append(EnergyPerAtomLoss(mode="mse"))
        elif key == "gradient_force":
            main_losses.append(GradientForceLoss(**kwargs))
            eval_losses.append(GradientForceLoss(mode="mae"))
            eval_losses.append(GradientForceLoss(mode="mse"))
        elif key == "direct_force":
            main_losses.append(DirectForceLoss(**kwargs))
            eval_losses.append(DirectForceLoss(mode="mae"))
            eval_losses.append(DirectForceLoss(mode="mse"))
            eval_losses.append(DirectForceLoss(mode="mae", transform="cos"))
            eval_losses.append(DirectForceLoss(mode="mse", transform="cos"))
            eval_losses.append(DirectForceLoss(mode="mae", transform="norm"))
            eval_losses.append(DirectForceLoss(mode="mse", transform="norm"))

        def main_loss(pred, data):
            return sum([loss_fn(pred, data) for loss_fn in main_losses])

        def eval_loss(pred, data):
            return {loss_fn.name: loss_fn(pred, data) for loss_fn in eval_losses}

    return main_loss, eval_loss


class BaseLoss(paddle.nn.Module):
    """
    Base loss class

    Parameters:
        mode (str): The loss function to use.
            'mse': Mean squared error.
            'mae': Mean absolute error.
            'huber': Huber loss.
        weight (float): The weight for the loss function. Default: 1.
        transform (str): The transformation to apply to the data. Default: None.
            'cos': 1 - cosine similarity.
            'norm': Norm.
    """

    def __init__(
        self, mode: str = "mse", weight: float = 1, transform: str = None, **kwargs
    ):
        super().__init__()
        self.weight = weight
        self.mode = mode
        if self.mode == "mse":
            self.loss_fn = paddle.nn.MSELoss()
        elif self.mode == "mae":
            self.loss_fn = paddle.nn.L1Loss()
        elif self.mode == "huber":
            self.loss_fn = paddle.nn.HuberLoss(**kwargs)
        else:
            raise ValueError(f"loss mode {mode} not implemented")
        self.transform = transform
        if self.transform is None:
            pass
        elif self.transform == "cos":
            self.cos = paddle.nn.CosineSimilarity(dim=-1)
            self.transform_fn = lambda x, y: (
                self.cos(x, y),
                paddle.ones(x.shape[:-1], device=x.device),
            )
        elif self.transform == "norm":
            self.transform_fn = lambda x, y: (x.norm(dim=-1), y.norm(dim=-1))
        else:
            raise ValueError(f"transform {transform} not implemented")

    def forward(self, pred, data):
        if self.weight == 0:
            return 0
        value_pred, value_data = self.from_outputs(pred, data)
        if self.transform is not None:
            value_pred, value_data = self.transform_fn(value_pred, value_data)
        loss = self.loss_fn(value_pred, value_data)
        if self.weight == 1:
            return loss
        else:
            return self.weight * loss

    def from_outputs(self, pred, data):
        raise NotImplementedError


class EnergyLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f"energy_{self.mode}"
        else:
            raise ValueError(f"transform {self.transform} for energy not implemented")

    def from_outputs(self, pred, data):
        return pred.energy, data.energy


class EnergyPerAtomLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f"energy_per_atom_{self.mode}"
        else:
            raise ValueError(
                f"transform {self.transform} for energy per atom not implemented"
            )

    def from_outputs(self, pred, data):
        n_atoms = scatter(paddle.ones_like(data.z), data.batch.to(dtype=paddle.int32))
        n_atoms = n_atoms.to(dtype=data.energy.dtype)
        return pred.energy / n_atoms, data.energy / n_atoms


class GradientForceLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f"gradient_force_{self.mode}"
        else:
            self.name = f"gradient_force_{self.transform}_{self.mode}"

    def from_outputs(self, pred, data):
        return pred.gradient_force, data.force


class DirectForceLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.transform is None:
            self.name = f"direct_force_{self.mode}"
        else:
            self.name = f"direct_force_{self.transform}_{self.mode}"

    def from_outputs(self, pred, data):
        return pred.direct_force, data.force
