# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""One-sided stochastic interpolant for VESBD/VPSBD variants.

Pure Paddle implementation using fixed-step Euler/Euler-Maruyama integration,
replacing the original torchdiffeq/torchsde dependencies. In the one-sided
case x_0 is Gaussian so the latent variable z merges with x_0.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import paddle
from paddle_scatter import scatter_mean

from .abstracts import Corrector, Epsilon, Interpolant, StochasticInterpolant
from .interpolants import (
    ScoreBasedDiffusionModelInterpolantVE,
    ScoreBasedDiffusionModelInterpolantVP,
)
from .single_stochastic_interpolant import DifferentialEquationType


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
        except AttributeError:
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
        self._integrator_kwargs = (
            integrator_kwargs if integrator_kwargs is not None else {}
        )
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
        raise NotImplementedError

    def _compute_mean_velocity(
        self, velocity: paddle.Tensor, batch_indices: paddle.Tensor
    ) -> paddle.Tensor:
        mean_vel = scatter_mean(velocity, batch_indices, dim=0)
        return mean_vel[batch_indices]

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
                    expected_velocity_p = expected_velocity_p - self._compute_mean_velocity(
                        expected_velocity_p, batch_indices
                    )
                    expected_velocity_m = expected_velocity_m - self._compute_mean_velocity(
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
                    expected_velocity = expected_velocity - self._compute_mean_velocity(
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
                    expected_velocity_p = expected_velocity_p - self._compute_mean_velocity(
                        expected_velocity_p, batch_indices
                    )
                    expected_velocity_m = expected_velocity_m - self._compute_mean_velocity(
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
                    expected_velocity = expected_velocity - self._compute_mean_velocity(
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
        return self._ode_integrate(*args, **kwargs)
