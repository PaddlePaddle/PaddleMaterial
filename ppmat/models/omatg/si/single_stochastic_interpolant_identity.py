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

"""Single Stochastic Interpolant Identity implementation.
"""

from typing import Callable, Dict, Iterable, Tuple

import paddle

from .abstracts import Corrector, StochasticInterpolantSpecies
from .corrector import IdentityCorrector


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
        assert paddle.equal_all(x_0, x_1).item()
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
        assert paddle.equal_all(x_0, x_1).item()
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
