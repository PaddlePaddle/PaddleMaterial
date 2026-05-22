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
        assert self._check_t(t).item()
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
        assert self._check_t(t).item()
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
