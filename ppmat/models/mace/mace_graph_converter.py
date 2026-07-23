# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""将晶体结构转换为 MACE 可用的原子图。"""

from __future__ import annotations

from typing import Optional

from pymatgen.core.structure import Structure

from ppmat.models.common.graph_converter import FindPointsInSpheres


class MACEGraphConverter:
    """基于截断半径邻域搜索构建 MACE 输入图。

    内部复用套件已有的 FindPointsInSpheres，保证与 Trainer / Predictor
    的图字段约定一致（atom_types、cart_coords、bond_dist、lattice 等）。

    Args:
        cutoff (float): 邻域截断半径（Å），默认 6.0，与 mace_mp0_medium 配置一致。
        num_elements (int): 元素种类数，默认 89（MACE-MP-0 覆盖范围）。
        pbc (tuple): 周期边界条件开关，默认全开。
        num_cpus (Optional[int]): 并行构建图的 CPU 数；None 表示自动。
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        num_elements: int = 89,
        pbc: tuple = (1, 1, 1),
        num_cpus: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
        self.num_elements = num_elements
        self.pbc = pbc
        self.num_cpus = num_cpus
        self._converter = FindPointsInSpheres(
            cutoff=cutoff,
            pbc=pbc,
            num_cpus=num_cpus,
        )

    def __call__(self, structure: Structure):
        """将单个或多个 Structure 转为 PGL Graph。"""
        return self._converter(structure)
