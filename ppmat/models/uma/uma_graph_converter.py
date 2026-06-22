# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import Any

from ppmat.models.uma._compat.graph import generate_graph


class UMAGraphConverter:
    """Build UMA edge fields from atomistic structure tensors.

    The converter follows the PaddleMaterials `build_graph_cfg` pattern while
    returning the edge fields expected by the UMA model.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        max_neighbors: int = 30,
        enforce_max_neighbors_strictly: bool = False,
        radius_pbc_version: int = 1,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.radius_pbc_version = radius_pbc_version

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        graph = generate_graph(
            data=sample,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            radius_pbc_version=self.radius_pbc_version,
            pbc=sample["pbc"],
        )
        return {
            "edge_index": graph["edge_index"],
            "cell_offsets": graph["cell_offsets"],
            "nedges": graph["neighbors"],
        }
