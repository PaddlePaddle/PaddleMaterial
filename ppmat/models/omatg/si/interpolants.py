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

"""Interpolant classes for Stochastic Interpolants.
"""

import math
from math import exp

import paddle

from .abstracts import Interpolant, Sigma, Tau
from .corrector import Corrector, IdentityCorrector, PeriodicBoundaryConditionsCorrector


class LinearInterpolant(Interpolant):
    """Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1."""

    def __init__(self) -> None:
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """Alpha function alpha(t) in the linear interpolant."""
        return 1.0 - t

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -paddle.ones_like(t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return t.clone()

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.ones_like(t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class PeriodicLinearInterpolant(LinearInterpolant):
    """
    Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1
    from two distributions p_0 and p_1 at times t with periodic boundary conditions.
    The coordinates are assumed to be in [0,1].
    """

    def __init__(self) -> None:
        """
        Construct PeriodicLinearInterpolant.
        """
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(
            min_value=0.0, max_value=1.0
        )

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector that corrects for periodic boundary conditions.
        :rtype: Corrector
        """
        return PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)


class TrigonometricInterpolant(Interpolant):
    """
    Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct trigonometric interpolant.
        """
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.cos(math.pi * t / 2.0)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -(math.pi / 2.0) * paddle.sin(math.pi * t / 2.0)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.sin(math.pi * t / 2.0)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function in the trigonometric interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return (math.pi / 2.0) * paddle.cos(math.pi * t / 2.0)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class ExponentialInterpolant(Interpolant):
    """
    Exponential interpolant I(t, x_0, x_1) = exp(-t) * x_0 + (1 - exp(-t)) * x_1
    between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        """
        Construct exponential interpolant.
        """
        super().__init__()

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) = exp(-t).

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.exp(-t)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the alpha function.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        return -paddle.exp(-t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Beta function beta(t) = 1 - exp(-t).

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return 1.0 - paddle.exp(-t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Time derivative of the beta function.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        return paddle.exp(-t)

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Identity corrector that does nothing.
        :rtype: Corrector
        """
        return IdentityCorrector()


class EncoderDecoderInterpolant(Interpolant):
    """Encoder-decoder interpolant with switch from x_0 to x_1 at switch_time."""

    def __init__(self, switch_time: float = 0.5, power: float = 1.0) -> None:
        super().__init__()
        if switch_time <= 0.0 or switch_time >= 1.0:
            raise ValueError("Switch time must be in (0,1).")
        if power < 0.5:
            raise ValueError("Power must be at least 0.5.")
        self._switch_time = switch_time
        self._power = power

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return paddle.where(
            t <= self._switch_time, paddle.cos(paddle.pi * a / b) ** 2, 0.0
        )

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        assert paddle.all((0.001 <= t) & (t <= 1.0 - 0.001))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = paddle.sin(2.0 * paddle.pi * a / (a + b))
        return paddle.where(
            t <= self._switch_time,
            self._power * paddle.pi * a * b * c / (t * (t - 1.0) * (a + b) ** 2),
            0.0,
        )

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return paddle.where(
            t > self._switch_time, paddle.cos(paddle.pi * a / b) ** 2, 0.0
        )

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        assert paddle.all((0.001 <= t) & (t <= 1.0 - 0.001))
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = paddle.sin(2.0 * paddle.pi * a / (a + b))
        return paddle.where(
            t > self._switch_time,
            self._power * paddle.pi * a * b * c / (t * (t - 1.0) * (a + b) ** 2),
            0.0,
        )

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class PeriodicEncoderDecoderInterpolant(EncoderDecoderInterpolant):
    """Encoder-decoder interpolant with periodic boundary conditions on [0,1]."""

    def __init__(self, switch_time: float = 0.5, power: float = 1.0) -> None:
        super().__init__(switch_time=switch_time, power=power)
        self._corrector = PeriodicBoundaryConditionsCorrector(
            min_value=0.0, max_value=1.0
        )

    def get_corrector(self) -> Corrector:
        return self._corrector


class ScoreBasedDiffusionModelInterpolantVP(Interpolant):
    """VP interpolant: I = sqrt(1 - tau^2) * x_0 + tau * x_1."""

    def __init__(self, tau: Tau) -> None:
        super().__init__()
        self._tau = tau
        self._one_over_e = 1.0 / exp(1.0)

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.sqrt(1.0 - self._tau.tau(t) ** 2)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        tau = self._tau.tau(t)
        t_sqrt = paddle.sqrt(1.0 - tau**2)
        tau_dot = self._tau.tau_dot(t)
        return -tau * tau_dot / t_sqrt

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return self._tau.tau(t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return self._tau.tau_dot(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class ScoreBasedDiffusionModelInterpolantVE(Interpolant):
    """VE interpolant: I = sqrt(sigma(1-t)^2 - sigma(0)^2) * x_0 + x_1."""

    def __init__(self, sigma: Sigma) -> None:
        super().__init__()
        self._sigma = sigma

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.sqrt(
            self._sigma.sigma(1.0 - t) ** 2
            - self._sigma.sigma(paddle.zeros_like(t)) ** 2
        )

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        sigma = self._sigma.sigma(1.0 - t)
        alpha = paddle.sqrt(sigma**2 - self._sigma.sigma(paddle.zeros_like(t)) ** 2)
        derivative = self._sigma.sigma_dot(1.0 - t)
        return -sigma * derivative / alpha

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.zeros_like(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()
