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

# GaussianExpansion is kept local because framework variants have a different
# interface (initial/final/num_centers/width) than CHGNet's (min/max/step/var).
from ppmat.models.chgnet.chgnet import CutoffPolynomial, Fourier, RadialBessel
from ppmat.models.matterchat.chgnet.model._local_basis import GaussianExpansion

__all__ = ["CutoffPolynomial", "Fourier", "GaussianExpansion", "RadialBessel"]
