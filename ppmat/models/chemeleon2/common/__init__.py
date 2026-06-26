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

from ppmat.models.chemeleon2.common.utils import DiagonalGaussianDistribution
from ppmat.models.chemeleon2.common.utils import scatter_mean, scatter_sum, scatter_std
from ppmat.models.chemeleon2.common.utils import lattice_params_to_matrix, matrix_to_lattice_params
from ppmat.models.chemeleon2.common.utils import frac_to_cart_coords, cart_to_frac_coords
from ppmat.models.chemeleon2.common.utils import get_pbc_distances, lattice_vector_to_volume
from ppmat.models.chemeleon2.common.utils import to_dense_batch
from ppmat.models.chemeleon2.common.utils import apply_augmentation, apply_noise
from ppmat.models.chemeleon2.common.utils import get_index_embedding
from ppmat.models.chemeleon2.common.utils import LoRALayer, LoRALinear
from ppmat.models.chemeleon2.common.utils import apply_lora_to_linear, get_lora_parameters
from ppmat.models.chemeleon2.common.utils import merge_lora_weights, print_trainable_parameters
from ppmat.models.chemeleon2.common.schema import CrystalBatch

__all__ = [
    "DiagonalGaussianDistribution",
    "CrystalBatch",
    "scatter_mean", "scatter_sum", "scatter_std",
    "lattice_params_to_matrix", "matrix_to_lattice_params",
    "frac_to_cart_coords", "cart_to_frac_coords",
    "get_pbc_distances", "lattice_vector_to_volume",
    "to_dense_batch",
    "apply_augmentation", "apply_noise",
    "get_index_embedding",
    "LoRALayer", "LoRALinear",
    "apply_lora_to_linear", "get_lora_parameters",
    "merge_lora_weights", "print_trainable_parameters",
]
