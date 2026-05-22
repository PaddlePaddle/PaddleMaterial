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

"""Model module for OMatG crystal structure prediction.
"""

from .omatg_cspnet import OMATGCSPNet
from ppmat.models.diffcsp.diffcsp import CSPLayer, SinusoidsEmbedding
from .omatg_model import OMATGCSPNetFull
from .utils import (
    lattice_params_to_matrix_paddle,
    frac_to_cart_coords,
    cart_to_frac_coords,
    radius_graph_pbc,
    repeat_blocks,
)

__all__ = [
    "OMATGCSPNet",
    "CSPLayer",
    "SinusoidsEmbedding",
    "OMATGCSPNetFull",
    "lattice_params_to_matrix_paddle",
    "frac_to_cart_coords",
    "cart_to_frac_coords",
    "radius_graph_pbc",
    "repeat_blocks",
]
