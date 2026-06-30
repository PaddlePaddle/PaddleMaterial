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

import importlib
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import paddle

from ppmat.datasets.omatg_dataset import OMATGData
from ppmat.utils.scatter import scatter_mean

from .interpolants import (
    Corrector,
    Epsilon,
    IdentityCorrector,
    Interpolant,
    LatentGamma,
    ScoreBasedDiffusionModelInterpolantVE,
    ScoreBasedDiffusionModelInterpolantVP,
    StochasticInterpolant,
    StochasticInterpolantSpecies,
)


class DifferentialEquationType(Enum):
    """Enum for differential equation types."""

    ODE = "ode"
    SDE = "sde"


def _compute_mean_velocity(
    velocity: paddle.Tensor, batch_indices: paddle.Tensor
) -> paddle.Tensor:
    mean_vel = scatter_mean(velocity, batch_indices, dim=0)
    return mean_vel[batch_indices]


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

    def loss(self, *args, **kwargs):
        raise NotImplementedError  # Overridden in __init__ via self.loss = _ode_loss / _sde_loss

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
                mean_velocity_p = _compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = _compute_mean_velocity(
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
                mean_velocity = _compute_mean_velocity(
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
                mean_velocity_p = _compute_mean_velocity(
                    expected_velocity_p, batch_indices
                )
                expected_velocity_p = expected_velocity_p - mean_velocity_p
                mean_velocity_m = _compute_mean_velocity(
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
                mean_velocity = _compute_mean_velocity(
                    expected_velocity, batch_indices
                )
                expected_velocity = expected_velocity - mean_velocity

            loss_b = paddle.mean(pred_b**2) - paddle.mean(pred_b * expected_velocity)

        loss_z = paddle.mean(pred_z**2) - 2.0 * paddle.mean(pred_z * z)

        return {"loss_b": loss_b, "loss_z": loss_z}

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
        raise NotImplementedError  # Overridden in __init__

class SingleStochasticInterpolantOS(StochasticInterpolant):
    """One-sided stochastic interpolant where x_0 is Gaussian.

    The latent variable z in SingleStochasticInterpolant is merged with x_0.
    Supports ODE or SDE during inference via fixed-step Euler integration.

    :param interpolant: Interpolant I(t, x_0, x_1).
    :param epsilon: Optional epsilon function for SDE.
    :param differential_equation_type: "ODE" or "SDE".
    :param integrator_kwargs: Optional kwargs (unused by fixed-step Euler).
    :param correct_center_of_mass_motion: Whether to zero COM velocity in loss.
    :param predict_velocity: Whether to compute loss for velocity field b.
    :param velocity_annealing_factor: Annealing factor for b during inference.
    """

    def __init__(
        self,
        interpolant: Interpolant,
        epsilon: Optional[Epsilon],
        differential_equation_type: str,
        integrator_kwargs: Optional[dict[str, Any]] = None,
        correct_center_of_mass_motion: bool = False,
        predict_velocity: bool = True,
        velocity_annealing_factor: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._interpolant = interpolant
        self._epsilon = epsilon
        self._corrector = self._interpolant.get_corrector()
        try:
            self._differential_equation_type = DifferentialEquationType[
                differential_equation_type
            ]
        except KeyError:
            raise ValueError(
                f"Unknown differential equation type: {differential_equation_type}."
            )
        if self._differential_equation_type == DifferentialEquationType.ODE:
            self.loss = self._ode_loss
            self.integrate = self._ode_integrate
            if self._epsilon is not None:
                raise ValueError("Epsilon function should not be provided for ODEs.")
        else:
            assert self._differential_equation_type == DifferentialEquationType.SDE
            self.loss = self._sde_loss
            self.integrate = self._sde_integrate
            if self._epsilon is None:
                raise ValueError("Epsilon function should be provided for SDEs.")
        self._correct_center_of_mass_motion = correct_center_of_mass_motion
        self._predict_velocity = predict_velocity
        self._use_antithetic = isinstance(
            self._interpolant,
            (
                ScoreBasedDiffusionModelInterpolantVP,
                ScoreBasedDiffusionModelInterpolantVE,
            ),
        )
        self._velocity_annealing_factor = velocity_annealing_factor
        if not self._predict_velocity and self._velocity_annealing_factor is not None:
            raise ValueError(
                "Velocity annealing factor should only be set if predict_velocity is True."
            )
        if self._predict_velocity and self._velocity_annealing_factor is None:
            self._velocity_annealing_factor = 0.0

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        interpolate = self._interpolant.interpolate(t, x_0, x_1)
        return interpolate, x_0.clone()

    def loss_keys(self) -> Iterable[str]:
        if self._predict_velocity:
            yield "loss_b"
            if self._differential_equation_type == DifferentialEquationType.SDE:
                yield "loss_z"
        else:
            yield "loss_z"

    def loss(self, *args, **kwargs):
        raise NotImplementedError  # Overridden in __init__ via self.loss = _ode_loss / _sde_loss

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
        assert x_0.shape == x_1.shape
        if self._predict_velocity:
            if self._use_antithetic:
                x_t_p = self._interpolant.interpolate(t, x_0, x_1)
                x_t_m = self._interpolant.interpolate(t, -x_0, x_1)
                expected_velocity_p = self._interpolant.interpolate_derivative(
                    t, x_0, x_1
                )
                expected_velocity_m = self._interpolant.interpolate_derivative(
                    t, -x_0, x_1
                )
                if self._correct_center_of_mass_motion:
                    expected_velocity_p = expected_velocity_p - _compute_mean_velocity(
                        expected_velocity_p, batch_indices
                    )
                    expected_velocity_m = expected_velocity_m - _compute_mean_velocity(
                        expected_velocity_m, batch_indices
                    )
                pred_b_p = model_function(x_t_p)[0]
                pred_b_m = model_function(x_t_m)[0]
                loss = (
                    0.5 * paddle.mean(pred_b_p**2)
                    + 0.5 * paddle.mean(pred_b_m**2)
                    - paddle.mean(pred_b_p * expected_velocity_p)
                    - paddle.mean(pred_b_m * expected_velocity_m)
                )
            else:
                expected_velocity = self._interpolant.interpolate_derivative(
                    t, x_0, x_1
                )
                pred_b = model_function(x_t)[0]
                if self._correct_center_of_mass_motion:
                    expected_velocity = expected_velocity - _compute_mean_velocity(
                        expected_velocity, batch_indices
                    )
                loss = paddle.mean(pred_b**2) - 2.0 * paddle.mean(
                    pred_b * expected_velocity
                )
            return {"loss_b": loss}
        else:
            pred_z = model_function(x_t)[1]
            return {"loss_z": paddle.mean(pred_z**2) - 2.0 * paddle.mean(pred_z * z)}

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
        assert x_0.shape == x_1.shape
        pred_b, pred_z = model_function(x_t)
        loss_z = paddle.mean(pred_z**2) - 2.0 * paddle.mean(pred_z * z)
        if self._predict_velocity:
            if self._use_antithetic:
                x_t_m = self._interpolant.interpolate(t, -x_0, x_1)
                expected_velocity_p = self._interpolant.interpolate_derivative(
                    t, x_0, x_1
                )
                expected_velocity_m = self._interpolant.interpolate_derivative(
                    t, -x_0, x_1
                )
                if self._correct_center_of_mass_motion:
                    expected_velocity_p = expected_velocity_p - _compute_mean_velocity(
                        expected_velocity_p, batch_indices
                    )
                    expected_velocity_m = expected_velocity_m - _compute_mean_velocity(
                        expected_velocity_m, batch_indices
                    )
                pred_b_m = model_function(x_t_m)[0]
                loss_b = (
                    0.5 * paddle.mean(pred_b**2)
                    + 0.5 * paddle.mean(pred_b_m**2)
                    - paddle.mean(pred_b * expected_velocity_p)
                    - paddle.mean(pred_b_m * expected_velocity_m)
                )
            else:
                expected_velocity = self._interpolant.interpolate_derivative(
                    t, x_0, x_1
                )
                if self._correct_center_of_mass_motion:
                    expected_velocity = expected_velocity - _compute_mean_velocity(
                        expected_velocity, batch_indices
                    )
                loss_b = paddle.mean(pred_b**2) - 2.0 * paddle.mean(
                    pred_b * expected_velocity
                )
            return {"loss_b": loss_b, "loss_z": loss_z}
        else:
            return {"loss_z": loss_z}

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
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if hasattr(time, "item") else float(time)
        t_tensor = paddle.to_tensor([t_val] * x_t.shape[0], dtype=x_t.dtype)
        if self._predict_velocity:
            assert self._velocity_annealing_factor is not None
            model_result = model_function(t_tensor, self._corrector.correct(x_t))
            velocity = model_result[0]
            annealing_factor = 1.0 + self._velocity_annealing_factor * t_val
            x_new = x_t + dt * annealing_factor * velocity
        else:
            model_result = model_function(t_tensor, self._corrector.correct(x_t))
            z = model_result[1]
            t1 = self._interpolant.alpha_dot(t_tensor) * z
            corr = self._corrector.correct(x_t)
            t2 = (
                self._interpolant.beta_dot(t_tensor)
                / self._interpolant.beta(t_tensor)
                * (corr - self._interpolant.alpha(t_tensor) * z)
            )
            x_new = x_t + dt * (t1 + t2)
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
        dt = time_step.item() if hasattr(time_step, "item") else float(time_step)
        t_val = time.item() if hasattr(time, "item") else float(time)
        t_tensor = paddle.to_tensor([t_val] * x_t.shape[0], dtype=x_t.dtype)
        model_result = model_function(t_tensor, self._corrector.correct(x_t))
        if self._predict_velocity:
            assert self._velocity_annealing_factor is not None
            drift = (1.0 + self._velocity_annealing_factor * t_val) * model_result[0]
            eta = model_result[1]
        else:
            z = model_result[1]
            t1 = self._interpolant.alpha_dot(t_tensor) * z
            corr = self._corrector.correct(x_t)
            t2 = (
                self._interpolant.beta_dot(t_tensor)
                / self._interpolant.beta(t_tensor)
                * (corr - self._interpolant.alpha(t_tensor) * z)
            )
            drift = t1 + t2
            eta = z
        epsilon_t = (
            self._epsilon.epsilon(paddle.to_tensor([t_val])).item()
            if self._epsilon
            else 0.0
        )
        alpha_t = self._interpolant.alpha(t_tensor)
        alpha_val = alpha_t[0].item() if alpha_t.numel() > 0 else 1.0
        diffusion = (
            paddle.sqrt(paddle.to_tensor(2.0 * epsilon_t * dt))
            * paddle.randn(x_t.shape)
        )
        x_new = x_t + drift * dt - (epsilon_t / max(alpha_val, 1e-8)) * eta * dt + diffusion
        return self._corrector.correct(x_new)

    def get_corrector(self) -> Corrector:
        return self._corrector

    def integrate(self, *args, **kwargs):
        raise NotImplementedError  # Overridden in __init__

class SingleStochasticInterpolantIdentity(StochasticInterpolantSpecies):
    """Stochastic interpolant x_t = x_0 = x_1 for atom species which must remain constant."""

    def __init__(self) -> None:
        super().__init__()

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Interpolate between x_0 and x_1 (must be equal)."""
        assert bool(paddle.equal_all(x_0, x_1))
        return x_0.clone(), paddle.zeros_like(x_0)

    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: Iterable[str]
        """
        yield "loss"

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

        This class always returns a zero loss with the key 'loss'.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param t:
            Times in [0,1].
        :param x_0:
            Points from p_0.
        :param x_1:
            Points from p_1.
        :param x_t:
            Stochastically interpolated points x_t.
        :param z:
            Random variable z.
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.

        :return:
            Losses.
        :rtype: Dict[str, paddle.Tensor]
        """
        assert bool(paddle.equal_all(x_0, x_1))
        return {"loss": paddle.to_tensor(0.0, place=x_0.place)}

    def integrate(
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
        Integrate the current positions x_t at the given time for the given time step.

        :param model_function:
            Model function (not used for identity).
        :param x_t:
            Current positions.
        :param time:
            Initial time.
        :param time_step:
            Time step.
        :param batch_indices:
            Batch indices.

        :return:
            Integrated position (unchanged).
        :rtype: paddle.Tensor
        """
        # Always return new object (unchanged for identity interpolant).
        return x_t.clone()

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant.

        :return:
            Identity corrector.
        :rtype: Corrector
        """
        return IdentityCorrector()

    def uses_masked_species(self) -> bool:
        """
        Return whether the stochastic interpolant uses masked species.

        :return:
            Whether the stochastic interpolant uses masked species.
        :rtype: bool
        """
        # Dataset does not contain masked species.
        return False

# Global constants
SMALL_TIME: float = 1.0e-3
BIG_TIME: float = 1.0 - SMALL_TIME


class DataField(Enum):
    """Enum for data fields in OMATGData relevant for stochastic interpolants."""

    pos = "pos"
    cell = "cell"
    species = "species"


def reshape_t(
    t: paddle.Tensor, n_atoms: paddle.Tensor, data_field: DataField
) -> paddle.Tensor:
    """Reshape times tensor for batch configurations for the given data field."""
    assert len(t.shape) == 1
    assert len(n_atoms.shape) == 1

    # Repeat t for each atom
    t_per_atom = paddle.repeat_interleave(t, n_atoms.cast("int64"))
    sum_n_atoms = int(n_atoms.sum().item())
    batch_size = len(t)

    if data_field == DataField.pos:
        # Reshape to (sum_n_atoms, 3) by repeating 3 times
        return paddle.reshape(
            paddle.repeat_interleave(t_per_atom, 3),
            [sum_n_atoms, 3],
        )
    elif data_field == DataField.cell:
        # Reshape to (batch_size, 3, 3)
        return paddle.reshape(
            paddle.repeat_interleave(t, 9), [batch_size, 3, 3]
        )
    else:
        # species case
        return t_per_atom


class StochasticInterpolants:
    """
    Collection of several stochastic interpolants between points x_0 and x_1 from two distributions
    p_0 and p_1 at times t for different coordinate types x (like atom species, fractional coordinates,
    and lattice vectors).

    Every stochastic interpolant is associated with a data field and a cost factor.

    :param stochastic_interpolants:
        Sequence of stochastic interpolants for the different coordinate types.
    :param data_fields:
        Sequence of data fields for the different stochastic interpolants.
    :param integration_time_steps:
        Number of integration time steps for the integration.

    :raises ValueError:
        If the number of stochastic interpolants and costs are not equal.
        If the number of stochastic interpolants and data fields are not equal.
        If the number of integration time steps is not positive.
    """

    def __init__(
        self,
        stochastic_interpolants: Sequence,
        data_fields: Sequence[str],
        integration_time_steps: int,
    ) -> None:
        """Constructor of the StochasticInterpolants class."""
        super().__init__()

        if not len(stochastic_interpolants) == len(data_fields):
            raise ValueError(
                "The number of stochastic interpolants and data fields must be equal."
            )

        try:
            self._data_fields = [DataField[df.lower()] for df in data_fields]
        except KeyError:
            raise ValueError(
                f"All data fields must be in {[d.value for d in DataField]}."
            )

        if not integration_time_steps > 0:
            raise ValueError("The number of integration time steps must be positive.")

        self._stochastic_interpolants = stochastic_interpolants
        self._integration_time_steps = integration_time_steps

    def __len__(self) -> int:
        """
        Return the number of stochastic interpolants handled by this class.

        :return:
            Number of stochastic interpolants.
        """
        return len(self._stochastic_interpolants)

    def loss_keys(self) -> List[str]:
        """
        Return the keys of the losses returned by this class.

        :return:
            Keys of the losses.
        """
        loss_keys = []
        for df, si in zip(self._data_fields, self._stochastic_interpolants):
            for key in si.loss_keys():
                full_key = f"{df.value}_{key}"
                if full_key in loss_keys:
                    raise ValueError(f"Key {full_key} is already used as a loss key.")
                loss_keys.append(full_key)
        return loss_keys

    def _interpolate(
        self,
        t: paddle.Tensor,
        x_0: OMATGData,
        x_1: OMATGData,
    ) -> Tuple[OMATGData, OMATGData]:
        """
        Stochastically interpolate between the collection of points x_0 and x_1 from the
        collection of two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :param x_0:
            Collection of points from the collection of distributions p_0.
        :param x_1:
            Collection of points from the collection of distributions p_1.

        :return:
            Collection of stochastically interpolated points x_t, and the collection of z values.
        :rtype: tuple[OMATGData, OMATGData]
        """
        assert bool(paddle.equal_all(x_0.batch, x_1.batch))
        assert bool(paddle.equal_all(x_0.n_atoms, x_1.n_atoms))

        n_atoms = x_0.n_atoms
        x_t = x_0.clone()
        z_data = {}

        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            reshaped_t = reshape_t(t, n_atoms, data_field)

            # Cell data requires different batch indices.
            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0.n_atoms))
            else:
                batch_indices = x_0.batch

            interpolated_x_t, z = stochastic_interpolant.interpolate(
                reshaped_t,
                getattr(x_0, field_name),
                getattr(x_1, field_name),
                batch_indices,
            )

            # Update x_t (using internal method)
            x_t.set_field(field_name, interpolated_x_t)
            z_data[field_name] = z

        return x_t, OMATGData._from_dict(z_data)

    def losses(
        self,
        model_function: Callable[[OMATGData, paddle.Tensor], OMATGData],
        t: paddle.Tensor,
        x_0: OMATGData,
        x_1: OMATGData,
    ) -> dict[str, paddle.Tensor]:
        """
        Compute the losses for the collection of stochastic interpolants.

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param t:
            Times in [0,1].
        :param x_0:
            Collection of points from the distribution p_0.
        :param x_1:
            Collection of points from the distribution p_1.

        :return:
            The losses for the collection of stochastic interpolants.
        :rtype: dict[str, paddle.Tensor]
        """
        # Interpolate everything first (asserts on batch/n_atoms are done inside _interpolate)
        x_t, z = self._interpolate(t, x_0, x_1)

        n_atoms = x_0.n_atoms

        losses = {}
        for stochastic_interpolant, data_field in zip(
            self._stochastic_interpolants, self._data_fields
        ):
            field_name = data_field.value
            b_data_field = field_name + "_b"
            eta_data_field = field_name + "_eta"

            reshaped_t = reshape_t(t, n_atoms, data_field)

            # Cell data requires different batch indices.
            if data_field == DataField.cell:
                batch_indices = paddle.arange(len(x_0.n_atoms))
            else:
                batch_indices = x_0.batch

            def model_prediction_fn(x):
                # Create a copy of x_t and update the field
                x_t_clone = x_t.clone()
                x_t_clone.set_field(field_name, x)
                model_result = model_function(x_t_clone, t)
                return model_result[b_data_field], model_result[eta_data_field]

            field_losses = stochastic_interpolant.loss(
                model_prediction_fn,
                reshaped_t,
                getattr(x_0, field_name),
                getattr(x_1, field_name),
                getattr(x_t, field_name),
                z.get_field(field_name),
                batch_indices,
            )

            for loss_key, loss_value in field_losses.items():
                assert loss_key not in losses
                losses[f"{field_name}_{loss_key}"] = loss_value

        return losses

    def integrate(
        self,
        x_0: OMATGData,
        model_function: Callable[[OMATGData, paddle.Tensor], OMATGData],
        save_intermediate: bool = False,
    ) -> Union[OMATGData, Tuple[OMATGData, List[OMATGData]]]:
        """
        Integrate the collection of points x_0 from time 0 to 1 based on the model
        that provides the velocity fields b and denoisers eta.

        :param x_0:
            Collection of points from the distribution p_0.
        :param model_function:
            Model function returning the velocity fields b and the denoisers eta.
        :param save_intermediate:
            If True, the intermediate points of the integration are saved and returned.

        :return:
            Collection of integrated points x_1.
            If save_intermediate is True, also returns a list of the intermediate points.
        """
        times = paddle.linspace(SMALL_TIME, BIG_TIME, self._integration_time_steps)
        dt = (BIG_TIME - SMALL_TIME) / (self._integration_time_steps - 1)

        x_t = x_0.clone()
        new_x_t = x_0.clone()

        if save_intermediate:
            inter_list = [x_t]
        else:
            inter_list = None

        with paddle.no_grad():
            for t_index in range(1, len(times)):
                t = times[t_index - 1]

                for stochastic_interpolant, data_field in zip(
                    self._stochastic_interpolants, self._data_fields
                ):
                    field_name = data_field.value
                    b_data_field = field_name + "_b"
                    eta_data_field = field_name + "_eta"

                    def model_prediction_fn(time, x):
                        # Repeat time for each element in the batch
                        t_val = time.item() if time.numel() == 1 else float(time[0])
                        time = paddle.full_like(
                            x_t.n_atoms.cast("float32"),
                            t_val,
                        )
                        x_int = x_t.clone()
                        x_int.set_field(field_name, x)
                        model_result = model_function(x_int, time)
                        return model_result[b_data_field], model_result[eta_data_field]

                    # Cell data requires different batch indices.
                    if data_field == DataField.cell:
                        batch_indices = paddle.arange(len(x_0.n_atoms))
                    else:
                        batch_indices = x_0.batch

                    new_value = stochastic_interpolant.integrate(
                        model_prediction_fn,
                        x_t.get_field(field_name),
                        t,
                        dt,
                        batch_indices,
                    )
                    new_x_t.set_field(field_name, new_value)

                x_t = new_x_t.clone()

                if save_intermediate:
                    inter_list.append(x_t)

        if save_intermediate:
            return x_t, inter_list
        else:
            return x_t

    def get_stochastic_interpolant(self, data_field: str):
        """
        Return the stochastic interpolant associated with the data field.

        :param data_field:
            Data field for which the stochastic interpolant is requested.

        :return:
            Stochastic interpolant associated with the data field.
        """
        try:
            df = DataField[data_field.lower()]
        except KeyError:
            raise ValueError(f"Data field must be in {[d.value for d in DataField]}.")

        index = self._data_fields.index(df)
        return self._stochastic_interpolants[index]

_SI_MODULE = "ppmat.models.omatg.si"
_SUB_MODULES = [
    "ppmat.models.omatg.si.interpolants",
    "ppmat.models.omatg.si.core",
]


def _resolve_class(class_name: str, default_module: str):
    """Resolve a class by name using importlib resolution."""
    if "." in class_name:
        module_path, cls_name = class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    module = importlib.import_module(default_module)
    if hasattr(module, class_name):
        return getattr(module, class_name)

    for submodule_path in _SUB_MODULES:
        try:
            sub = importlib.import_module(submodule_path)
            if hasattr(sub, class_name):
                return getattr(sub, class_name)
        except ImportError:
            continue

    raise AttributeError(
        f"Class '{class_name}' not found in '{default_module}' or its submodules."
    )


def _build_object(cfg: dict, default_module: str):
    """Build a single object from a {__class_name__, __init_params__} config.

    Recursively builds nested objects when an __init_params__ entry is itself a
    config dict (i.e. contains __class_name__).
    """
    cls = _resolve_class(cfg["__class_name__"], default_module)
    params = cfg.get("__init_params__", {})
    built_params = {}
    for key, val in params.items():
        if isinstance(val, dict) and "__class_name__" in val:
            built_params[key] = _build_object(val, default_module)
        elif isinstance(val, list):
            built_params[key] = [
                _build_object(item, default_module)
                if isinstance(item, dict) and "__class_name__" in item
                else item
                for item in val
            ]
        else:
            built_params[key] = val
    return cls(**built_params)


def build_si_from_cfg(si_cfg: dict) -> StochasticInterpolants:
    """Build StochasticInterpolants from a config dict.

    Expected schema:
        stochastic_interpolants: list of {__class_name__, __init_params__}
        data_fields: list[str]
        integration_time_steps: int
        relative_si_costs: dict[str, float]  # optional
    """
    interpolants = [
        _build_object(cfg_item, _SI_MODULE)
        for cfg_item in si_cfg["stochastic_interpolants"]
    ]
    return StochasticInterpolants(
        stochastic_interpolants=interpolants,
        data_fields=si_cfg["data_fields"],
        integration_time_steps=si_cfg.get("integration_time_steps", 210),
    )


def build_sampler_from_cfg(sampler_cfg: dict):
    """Build IndependentSampler from a config dict.

    Expected schema (all keys optional):
        dataset_name: str | None
        mirror_species: bool
        mask_species: bool
    """
    from ppmat.models.omatg.model import IndependentSampler

    return IndependentSampler(
        dataset_name=sampler_cfg.get("dataset_name"),
        mirror_species=sampler_cfg.get("mirror_species", True),
        mask_species=sampler_cfg.get("mask_species", False),
    )


MAX_ATOM_NUM: int = 100


class DiscreteFlowMatchingMask(StochasticInterpolantSpecies):
    """Discrete flow matching between masked base p_0 and target p_1 for species.

    The base points x_0 are entirely in the masked state (token 0).
    The model prediction returns (sum(n_atoms), MAX_ATOM_NUM) logits.
    Loss is cross_entropy(pred, x_1 - 1).

    :param noise: noise parameter scaling added during integration.
    """

    def __init__(self, noise: float = 0.0) -> None:
        super().__init__()
        if noise < 0.0:
            raise ValueError("Noise parameter must be greater than or equal to 0.")
        self._mask_index = 0
        self._noise = noise

    def interpolate(
        self,
        t: paddle.Tensor,
        x_0: paddle.Tensor,
        x_1: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        assert paddle.all(x_1 != self._mask_index)
        x_t = x_0.clone()
        mask = paddle.rand(x_0.shape) < t
        x_t[mask] = x_1[mask]
        return x_t, paddle.zeros_like(x_t)

    def loss_keys(self) -> Iterable[str]:
        yield "loss"

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
        assert x_0.shape == x_1.shape
        assert paddle.all(x_0 == self._mask_index)
        assert paddle.all(x_1 != self._mask_index)
        pred = model_function(x_t)[0]
        assert pred.shape == (x_0.shape[0], MAX_ATOM_NUM)
        return {"loss": paddle.nn.functional.cross_entropy(input=pred, label=x_1 - 1)}

    def integrate(
        self,
        model_function: Callable[
            [paddle.Tensor, paddle.Tensor], Tuple[paddle.Tensor, paddle.Tensor]
        ],
        x_t: paddle.Tensor,
        time: paddle.Tensor,
        time_step: paddle.Tensor,
        batch_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        eps = paddle.finfo(paddle.float64).eps
        x_1_probs = paddle.nn.functional.softmax(
            model_function(time, x_t)[0], axis=-1
        )
        x_1_probs = x_1_probs.reshape((-1, MAX_ATOM_NUM))
        shifted_x_1 = paddle.multinomial(
            x_1_probs, num_samples=1, replacement=True
        ).squeeze(-1)
        shifted_x_t = x_t - 1
        assert shifted_x_1.shape == x_t.shape == shifted_x_t.shape
        shifted_x_1_hot = paddle.nn.functional.one_hot(
            shifted_x_1, num_classes=MAX_ATOM_NUM
        )
        dpt = shifted_x_1_hot - 1.0 / MAX_ATOM_NUM
        dpt_xt = dpt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        pt = time * shifted_x_1_hot + (1.0 - time) * (1.0 / MAX_ATOM_NUM)
        pt_xt = pt.gather(-1, shifted_x_t[:, None]).squeeze(-1)
        S = paddle.count_nonzero(x=pt, axis=-1)
        rate = paddle.nn.functional.relu(x=dpt - dpt_xt[:, None]) / (
            S * pt_xt
        )[:, None]
        rate[(pt_xt == 0.0)[:, None].expand([-1, MAX_ATOM_NUM])] = 0.0
        rate[pt == 0.0] = 0.0
        rate_db = paddle.zeros_like(rate)
        if self._noise > 0.0:
            rate_db[shifted_x_t == shifted_x_1] = 1.0
            rate_db[shifted_x_1 != shifted_x_t] = (MAX_ATOM_NUM * time + 1.0 - time) / (
                1.0 - time + eps
            )
            rate_db *= self._noise
        rate = rate + rate_db
        step_probs = (rate * time_step).clip(max=1.0)
        step_probs = (rate * time_step).clip(max=1.0)
        step_probs[paddle.arange(len(shifted_x_t)), shifted_x_t] = 0.0
        step_probs[paddle.arange(len(shifted_x_t)), shifted_x_t] = (
            (1.0 - step_probs.sum(axis=-1, keepdim=True)).clip(min=0.0)
        ).squeeze(-1)
        x_t = paddle.multinomial(
            step_probs, num_samples=1, replacement=True
        ).squeeze(-1) + 1
        return x_t

    def uses_masked_species(self) -> bool:
        return True
