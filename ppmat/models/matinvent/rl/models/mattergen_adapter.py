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


class MatterGenRLAdapter:
    def __init__(self, base_model):
        self._base_model = base_model

    def __getattr__(self, name):
        return getattr(self._base_model, name)

    def noise_level_encoding(self, t):
        if hasattr(self._base_model, "noise_level_encoding"):
            return self._base_model.noise_level_encoding(t)
        from ppmat.models.common.sinusoidal_embedding import SinusoidalEmbeddings
        return SinusoidalEmbeddings(dim=self.time_dim)(t)

    @property
    def max_t(self): return getattr(self._base_model, "max_t", 1.0)
    @property
    def time_dim(self): return getattr(self._base_model, "time_dim", 256)
    @property
    def lattice_loss_weight(self): return getattr(self._base_model, "lattice_loss_weight", 1.0)
    @property
    def coord_loss_weight(self): return getattr(self._base_model, "coord_loss_weight", 0.1)
    @property
    def atom_loss_weight(self): return getattr(self._base_model, "atom_loss_weight", 1.0)
    @property
    def d3pm_hybrid_lambda(self): return getattr(self._base_model, "d3pm_hybrid_lambda", None)


def create_matinvent_adapter(model):
    return MatterGenRLAdapter(model)
