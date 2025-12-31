# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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

from ppmat.models.common.e3nn.math import normalize2mom
from ppmat.models.common.e3nn.o3 import SO3Grid


class SO3Activation(paddle.nn.Layer):
    """Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax_in : int
        input lmax

    lmax_out : int
        output lmax

    act : function
        activation function :math:`\\phi`

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}
    """

    def __init__(
        self,
        lmax_in,
        lmax_out,
        act,
        resolution,
        *,
        normalization="component",
        aspect_ratio=2,
    ):
        super().__init__()
        self.grid_in = SO3Grid(
            lmax_in, resolution, normalization=normalization, aspect_ratio=aspect_ratio
        )
        self.grid_out = SO3Grid(
            lmax_out, resolution, normalization=normalization, aspect_ratio=aspect_ratio
        )
        self.act = normalize2mom(act)
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.lmax_in} -> {self.lmax_out})"

    def forward(self, features):
        """evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.irreps_in.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.irreps_out.dim)``
        """
        features = self.grid_in.to_grid(features)
        features = self.act(features)
        features = self.grid_out.from_grid(features)
        return features
