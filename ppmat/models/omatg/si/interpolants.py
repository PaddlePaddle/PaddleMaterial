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

"""Abstract classes for Stochastic Interpolants.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Tuple

import paddle


class TimeChecker:
    """Check that all times in a tensor are in [0,1]."""

    @staticmethod
    def _check_t(t: paddle.Tensor) -> paddle.Tensor:
        """Check that all times are in [0,1]."""
        return paddle.all((0.0 <= t) & (t <= 1.0))


class Corrector(ABC):
    """Abstract corrector function (e.g., for PBC wrapping)."""

    @abstractmethod
    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        """Correct the input x."""
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        """Correct x_1 based on reference x_0."""
        raise NotImplementedError


class Epsilon(ABC, TimeChecker):
    """
    Abstract class for defining an epsilon function epsilon(t).
    """

    @abstractmethod
    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the epsilon function at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Epsilon function epsilon(t).
        :rtype: paddle.Tensor
        """
        raise NotImplementedError


class Interpolant(ABC, TimeChecker):
    """
    Abstract class for defining an interpolant I(t, x_0, x_1) = alpha(t) * x_0 + beta(t) * x_1
    in a stochastic interpolant between points x_0 and x_1 from two distributions p_0 and p_1 at times t.
    """

    def interpolate(
        self, t: paddle.Tensor, x_0: paddle.Tensor, x_1: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: paddle.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: paddle.Tensor

        :return:
            Interpolated value.
        :rtype: paddle.Tensor
        """
        assert bool(self._check_t(t))
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime
        return self.get_corrector().correct(x_t)

    def interpolate_derivative(
        self, t: paddle.Tensor, x_0: paddle.Tensor, x_1: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Compute the derivative of the interpolant with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: paddle.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: paddle.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: paddle.Tensor
        """
        assert bool(self._check_t(t))
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime

    @abstractmethod
    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant.

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise NotImplementedError

    @abstractmethod
    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Alpha function alpha(t) in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Values of the alpha function at the given times.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: paddle.Tensor):
        """
        Time derivative of the beta function in the linear interpolant.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivatives of the beta function at the given times.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError


class LatentGamma(ABC, TimeChecker):
    """
    Abstract class for defining the gamma function gamma(t) in a latent variable gamma(t) * z.
    """

    @abstractmethod
    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Gamma function gamma(t).
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the gamma function gamma(t) with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivative of the gamma function.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges
        as t -> 0 or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        raise NotImplementedError


class StochasticInterpolant(ABC, TimeChecker):
    """
    Abstract class for defining a stochastic interpolant between points x_0 and x_1
    from two distributions p_0 and p_1 at times t.
    """

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def loss_keys(self) -> Iterable[str]:
        """
        Get the keys of the losses returned by the loss function.

        :return:
            Keys of the losses.
        :rtype: List[str]
        """
        raise NotImplementedError

    @abstractmethod
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

        :param model_function:
            Model function returning the velocity fields b and the denoisers eta given the current positions x_t.
        :type model_function: Callable[[paddle.Tensor, paddle.Tensor], tuple[paddle.Tensor, paddle.Tensor]]
        :param t:
            Times in [0,1].
        :type t: paddle.Tensor
        :param x_0:
            Points from p_0.
        :type x_0: paddle.Tensor
        :param x_1:
            Points from p_1.
        :type x_1: paddle.Tensor
        :param x_t:
            Stochastically interpolated points x_t.
        :type x_t: paddle.Tensor
        :param z:
            Random variable z that was used for the stochastic interpolation to get the model prediction.
        :type z: paddle.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: paddle.Tensor

        :return:
            Losses.
        :rtype: Dict[str, paddle.Tensor]
        """
        raise NotImplementedError

    @abstractmethod
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
            Model function returning the velocity fields b and the denoisers eta.
        :type model_function: Callable[[paddle.Tensor, paddle.Tensor], tuple[paddle.Tensor, paddle.Tensor]]
        :param x_t:
            Current positions.
        :type x_t: paddle.Tensor
        :param time:
            Initial time (0-dimensional tensor).
        :type time: paddle.Tensor
        :param time_step:
            Time step (0-dimensional tensor).
        :type time_step: paddle.Tensor
        :param batch_indices:
            Tensor containing the configuration index for every atom in the batch.
        :type batch_indices: paddle.Tensor

        :return:
            Integrated position.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant.

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise NotImplementedError


class StochasticInterpolantSpecies(StochasticInterpolant, ABC):
    """
    Abstract class for defining a stochastic interpolant between species x_0 and x_1.
    """

    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the stochastic interpolant.

        The stochastic interpolants for atom species should not define a corrector.

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise RuntimeError("Corrector not defined for StochasticInterpolantSpecies.")

    @abstractmethod
    def uses_masked_species(self) -> bool:
        """
        Whether the stochastic interpolant uses an additional masked species.

        :return:
            Whether the stochastic interpolant uses an additional masked species.
        :rtype: bool
        """
        raise NotImplementedError


class Sigma(ABC, TimeChecker):
    """
    Abstract class for defining a noise schedule sigma(s) for a one-sided variance-exploding interpolant.
    """

    @abstractmethod
    def sigma(self, s: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the sigma function at times s.

        :param s:
            Times in [0,1].
        :type s: paddle.Tensor

        :return:
            Sigma function sigma(s).
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def sigma_dot(self, s: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the sigma function with respect to time.

        :param s:
            Times in [0,1].
        :type s: paddle.Tensor

        :return:
            Derivative of the sigma function at the given times.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError


class Tau(ABC, TimeChecker):
    """
    Abstract class for defining the tau function tau(t) for a one-sided variance-preserving interpolant.

    The one-sided variance-preserving interpolant is defined as x_t = sqrt(1 - tau^2(t)) * x_0 + tau(t) * x_1.
    """

    @abstractmethod
    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Evaluate the tau function at times t.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Tau function tau(t).
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Compute the derivative of the tau function with respect to time.

        :param t:
            Times in [0,1].
        :type t: paddle.Tensor

        :return:
            Derivative of the tau function at the given times.
        :rtype: paddle.Tensor
        """
        raise NotImplementedError

import paddle


class IdentityCorrector(Corrector):
    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        return x

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        return x_1.clone()


class PeriodicBoundaryConditionsCorrector(Corrector):
    def __init__(self, min_value: float, max_value: float) -> None:
        super().__init__()
        if min_value >= max_value:
            raise ValueError("Minimum value must be less than maximum value.")
        self._min_value = min_value
        self._max_value = max_value
        self._range = max_value - min_value

    def correct(self, x: paddle.Tensor) -> paddle.Tensor:
        return paddle.remainder(x - self._min_value, self._range) + self._min_value

    def unwrap(self, x_0: paddle.Tensor, x_1: paddle.Tensor) -> paddle.Tensor:
        half = self._range / 2.0
        sep = x_1 - x_0
        return x_0 + paddle.remainder(sep + half, self._range) - half


class VanishingEpsilon(Epsilon):
    def __init__(self, c: float = 1.0, sigma: float = 0.01, mu: float = 0.075) -> None:
        super().__init__()
        self._c = c
        self._sigma = sigma
        self._mu = mu

    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        f1 = paddle.sigmoid((t - self._mu) / self._sigma)
        f2 = paddle.sigmoid((1 - self._mu - t) / self._sigma)
        return self._c * f1 * f2


class ConstantEpsilon(Epsilon):
    def __init__(self, c: float) -> None:
        super().__init__()
        self._c = c

    def epsilon(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.full_like(t, self._c)


class TauConstantSchedule(Tau):
    def tau(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return t.clone()

    def tau_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.ones_like(t)


class LinearInterpolant(Interpolant):
    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        return 1.0 - t

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return -paddle.ones_like(t)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return t.clone()

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class PeriodicLinearInterpolant(LinearInterpolant):
    def __init__(self) -> None:
        super().__init__()
        self._corrector = PeriodicBoundaryConditionsCorrector(0.0, 1.0)

    def get_corrector(self) -> Corrector:
        return PeriodicBoundaryConditionsCorrector(0.0, 1.0)


class ScoreBasedDiffusionModelInterpolantVP(Interpolant):
    def __init__(self, tau: Tau) -> None:
        super().__init__()
        self._tau = tau

    def alpha(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.sqrt(1.0 - self._tau.tau(t) ** 2)

    def alpha_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        tau = self._tau.tau(t)
        return -tau * self._tau.tau_dot(t) / paddle.sqrt(1.0 - tau**2)

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return self._tau.tau(t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return self._tau.tau_dot(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class ScoreBasedDiffusionModelInterpolantVE(Interpolant):
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
        return -sigma * self._sigma.sigma_dot(1.0 - t) / alpha

    def beta(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.ones_like(t)

    def beta_dot(self, t: paddle.Tensor) -> paddle.Tensor:
        return paddle.zeros_like(t)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class GeometricSigma(Sigma):
    def __init__(self, sigma_min: float, sigma_max: float) -> None:
        super().__init__()
        if sigma_min <= 0.0 or sigma_max <= 0.0 or sigma_max <= sigma_min:
            raise ValueError("sigma_min and sigma_max must be positive, sigma_max > sigma_min.")
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        from math import log
        self._log_ratio = log(sigma_max / sigma_min)

    def sigma(self, s: paddle.Tensor) -> paddle.Tensor:
        self._check_t(s)
        return self._sigma_min * (self._sigma_max / self._sigma_min) ** s

    def sigma_dot(self, s: paddle.Tensor) -> paddle.Tensor:
        self._check_t(s)
        return self._sigma_min * self._log_ratio * (self._sigma_max / self._sigma_min) ** s


class LatentGammaSqrt(LatentGamma):
    def __init__(self, a: float) -> None:
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        self._a = a

    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return paddle.sqrt(self._a * t * (1.0 - t))

    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        return self._a * (1.0 - 2.0 * t) / (2.0 * paddle.sqrt(self._a * t * (1.0 - t)))

    def requires_antithetic(self) -> bool:
        return True


class LatentGammaEncoderDecoder(LatentGamma):
    def __init__(self, a: float = 1.0, switch_time: float = 0.5, power: float = 1.0) -> None:
        super().__init__()
        if a <= 0.0 or switch_time <= 0.0 or switch_time >= 1.0 or power < 0.5:
            raise ValueError("Invalid parameters.")
        self._sqrt_a = a**0.5
        self._switch_time = switch_time
        self._power = power

    def gamma(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power + a
        return self._sqrt_a * paddle.sin(paddle.pi * a / b) ** 2

    def gamma_derivative(self, t: paddle.Tensor) -> paddle.Tensor:
        self._check_t(t)
        a = (t - self._switch_time * t) ** self._power
        b = (self._switch_time - self._switch_time * t) ** self._power
        c = paddle.sin(2.0 * paddle.pi * a / (a + b))
        return -self._sqrt_a * self._power * paddle.pi * a * b * c / (t * (t - 1.0) * (a + b) ** 2)

    def requires_antithetic(self) -> bool:
        return False
