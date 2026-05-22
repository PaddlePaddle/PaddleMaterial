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

"""Single Stochastic Interpolant implementation.
"""

from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import paddle
from paddle_scatter import scatter_mean

from .abstracts import (
    Corrector,
    Epsilon,
    Interpolant,
    LatentGamma,
    StochasticInterpolant,
)


class DifferentialEquationType(Enum):
    """Enum for differential equation types."""

    ODE = "ode"
    SDE = "sde"


class SingleStochasticInterpolant(StochasticInterpolant):
    """Stochastic interpolant x_t = I(t, x_0, x_1) + gamma(t) * z.
    Supports ODE or SDE during inference.
    """

    def __init__(
        self,
        interpolant: Interpolant,
        gamma: Optional[LatentGamma] = None,
        epsilon: Optional[Epsilon] = None,
        differential_equation_type: str = "ODE",
        integrator_kwargs: Optional[dict[str, Any]] = None,
        correct_center_of_mass_motion: bool = False,
        velocity_annealing_factor: float = 0.0,
    ) -> None:
        """Construct stochastic interpolant."""
        super().__init__()
        self._interpolant = interpolant
        self._gamma = gamma
        if self._gamma is not None:
            self._use_antithetic = self._gamma.requires_antithetic()
        else:
            self._use_antithetic = False
        self._epsilon = epsilon
        self._differential_equation_type = differential_equation_type.upper()
        self._corrector = self._interpolant.get_corrector()

        if self._differential_equation_type == "ODE":
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
            if self._epsilon is not None:
                raise ValueError("Epsilon function should not be provided for ODEs.")
        elif self._differential_equation_type == "SDE":
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate
            if self._epsilon is None:
                raise ValueError("Epsilon function should be provided for SDEs.")
            if self._gamma is None:
                raise ValueError("Gamma function should be provided for SDEs.")
        else:
            raise ValueError(
                f"Unknown differential equation type: {differential_equation_type}"
            )

        self._integrator_kwargs = (
            integrator_kwargs if integrator_kwargs is not None else {}
        )
        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._velocity_annealing_factor = velocity_annealing_factor

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Stochastically interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: paddle.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: paddle.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: paddle.Tensor

        :return:
            Stochastically interpolated points x_t, random variables z used for interpolation.
        :rtype: tuple[paddle.Tensor, paddle.Tensor]
        """
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        if self._gamma is not None:
            z = paddle.randn(x_0.shape)
            gamma_t = self._gamma.gamma(t)
            interpolate = self._corrector.correct(interpolate + gamma_t * z)
        else:
            z = paddle.zeros_like(x_0)
        return interpolate, z

    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: Iterable[str]
        """
        if self._differential_equation_type == "ODE":
            yield "loss_b"
        else:
            yield "loss_b"
            yield "loss_z"

    def loss(
        self,
        model_function: Callable[[paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]],
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        """
        Compute the losses for the stochastic interpolant.

        This method is only defined here to define all methods of the abstract base class.
        The actual loss method is either _ode_loss or _sde_loss.

        :return:
            Losses.
        :rtype: Dict[str, paddle.Tensor]
        """
        raise NotImplementedError

    def _ode_loss(
        self,
        model_function: Callable[[paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]],
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        """
        Compute the losses for the ODE stochastic interpolant.

        :return:
            Losses.
        :rtype: Dict[str, paddle.Tensor]
        """
        assert x_0.shape == x_1.shape

        if self._use_antithetic:
            assert self._gamma is not None
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)

            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(
                t, x_0, x_1
            )
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z

            if self._correct_center_of_mass_motion:
                mean_velocity_p = self._compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = self._compute_mean_velocity(
                    expected_velocity_m, batch_indices
                )
                expected_velocity_m = expected_velocity_m - mean_velocity_m

            pred_b_p = model_function(x_t_p)[0]
            pred_b_m = model_function(x_t_m)[0]

            loss = (
                0.5 * paddle.mean(pred_b_p**2)
                + 0.5 * paddle.mean(pred_b_m**2)
                - paddle.mean(pred_b_p * expected_velocity_p)
                - paddle.mean(pred_b_m * expected_velocity_m)
            )
        else:
            expected_velocity = self._interpolant.interpolate_derivative(t, x_0, x_1)
            if self._gamma is not None:
                expected_velocity += self._gamma.gamma_derivative(t) * z

            pred_b = model_function(x_t)[0]

            if self._correct_center_of_mass_motion:
                mean_velocity = self._compute_mean_velocity(
                    expected_velocity, batch_indices
                )
                expected_velocity = expected_velocity - mean_velocity

            loss = paddle.mean(pred_b**2) - 2.0 * paddle.mean(
                pred_b * expected_velocity
            )

        return {"loss_b": loss}

    def _sde_loss(
        self,
        model_function: Callable[[paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]],
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        x_t: paddle.Tensor,
        z: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        """
        Compute the losses for the SDE stochastic interpolant.

        :return:
            Losses.
        :rtype: Dict[str, paddle.Tensor]
        """
        assert x_0.shape == x_1.shape
        assert self._gamma is not None

        if self._use_antithetic:
            x_t_without_gamma = self._interpolant.interpolate(t, x_0, x_1)
            gamma = self._gamma.gamma(t)
            x_t_p = self._corrector.correct(x_t_without_gamma + gamma * z)
            x_t_m = self._corrector.correct(x_t_without_gamma - gamma * z)

            expected_velocity_without_gamma = self._interpolant.interpolate_derivative(
                t, x_0, x_1
            )
            gamma_derivative = self._gamma.gamma_derivative(t)
            expected_velocity_p = expected_velocity_without_gamma + gamma_derivative * z
            expected_velocity_m = expected_velocity_without_gamma - gamma_derivative * z

            if self._correct_center_of_mass_motion:
                mean_velocity_p = self._compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = self._compute_mean_velocity(
                    expected_velocity_m, batch_indices
                )
                expected_velocity_m = expected_velocity_m - mean_velocity_m

            pred_b_p, pred_z = model_function(x_t_p)
            pred_b_m, _ = model_function(x_t_m)

            loss_b = (
                0.5 * paddle.mean(pred_b_p**2)
                + 0.5 * paddle.mean(pred_b_m**2)
                - paddle.mean(pred_b_p * expected_velocity_p)
                - paddle.mean(pred_b_m * expected_velocity_m)
            )
        else:
            expected_velocity = (
                self._interpolant.interpolate_derivative(t, x_0, x_1)
                + self._gamma.gamma_derivative(t) * z
            )
            pred_b, pred_z = model_function(x_t)

            if self._correct_center_of_mass_motion:
                mean_velocity = self._compute_mean_velocity(
                    expected_velocity, batch_indices
                )
                expected_velocity = expected_velocity - mean_velocity

            loss_b = paddle.mean(pred_b**2) - paddle.mean(pred_b * expected_velocity)

        loss_z = paddle.mean(pred_z**2) - 2.0 * paddle.mean(pred_z * z)

        return {"loss_b": loss_b, "loss_z": loss_z}

    def _compute_mean_velocity(
        self, velocity: paddle.Tensor, batch_indices: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Compute the mean velocity for every configuration and replicate for every atom.

        :param velocity:
            Velocity tensor.
        :type velocity: paddle.Tensor
        :param batch_indices:
            Batch indices for each atom.
        :type batch_indices: paddle.Tensor

        :return:
            Mean velocity replicated for each atom.
        :rtype: paddle.Tensor
        """
        mean_vel = scatter_mean(velocity, batch_indices, dim=0)
        return mean_vel[batch_indices]

    def _ode_integrate(
        self,
        model_function: Callable[
            [paddle.Tensor, paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]
        ],
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Integrate the ODE for the current positions.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param x_t:
            Current positions.
        :param time:
            Initial time.
        :param time_step:
            Time step.
        :param batch_indices:
            Batch indices.

        :return:
            Integrated position.
        """
        # Simple Euler integration using paddle operations
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if hasattr(time, "item") else float(time)

        t_tensor = paddle.to_tensor([t_val] * x_t.shape[0], dtype=x_t.dtype)
        model_result = model_function(t_tensor, self._corrector.correct(x_t))
        velocity = model_result[0]
        annealing_factor = 1.0 + self._velocity_annealing_factor * t_val
        x_new = x_t + dt * annealing_factor * velocity

        return self._corrector.correct(x_new)

    def _sde_integrate(
        self,
        model_function: Callable[
            [paddle.Tensor, paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]
        ],
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Integrate the SDE for the current positions.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param x_t:
            Current positions.
        :param time:
            Initial time.
        :param time_step:
            Time step.
        :param batch_indices:
            Batch indices.

        :return:
            Integrated position.
        """
        # Euler-Maruyama integration using paddle operations
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if hasattr(time, "item") else float(time)

        t_tensor = paddle.to_tensor([t_val] * x_t.shape[0], dtype=x_t.dtype)
        model_result = model_function(t_tensor, self._corrector.correct(x_t))
        drift = model_result[0]
        eta = model_result[1] if len(model_result) > 1 else paddle.zeros_like(drift)

        epsilon_t = (
            self._epsilon.epsilon(paddle.to_tensor([t_val])).item()
            if self._epsilon
            else 0.0
        )
        gamma_t = (
            self._gamma.gamma(paddle.to_tensor([t_val])).item() if self._gamma else 1.0
        )

        # Euler-Maruyama update: x_{t+dt} = x_t + drift * dt + sqrt(2 * epsilon) * dW
        diffusion = (
            paddle.sqrt(paddle.to_tensor(2.0 * epsilon_t * dt))
            * paddle.randn(x_t.shape)
        )
        x_new = x_t + drift * dt - (epsilon_t / gamma_t) * eta * dt + diffusion

        return self._corrector.correct(x_new)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant.

        :return:
            Corrector.
        :rtype: Corrector
        """
        return self._corrector

    def integrate(self, *args, **kwargs):
        return self._ode_integrate(*args, **kwargs)
