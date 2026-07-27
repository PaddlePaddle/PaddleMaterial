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

import os

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.deeph_graph import get_graph


class DeepHGraphConverter:
    """Convert a PaddleMaterials structure graph to a DeepH Hamiltonian graph."""

    def __init__(
        self,
        radius,
        max_num_nbr,
        default_dtype,
        interface,
        num_l,
        create_from_DFT=False,
        if_lcmp_graph=True,
        separate_onsite=False,
        target="hamiltonian",
        huge_structure=False,
        if_new_sp=False,
    ):
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.default_dtype = default_dtype
        self.interface = interface
        self.num_l = num_l
        self.create_from_DFT = create_from_DFT
        self.if_lcmp_graph = if_lcmp_graph
        self.separate_onsite = separate_onsite
        self.target = target
        self.huge_structure = huge_structure
        self.if_new_sp = if_new_sp

    @staticmethod
    def _to_structure(structure):
        if isinstance(structure, Structure):
            return structure
        return AseAtomsAdaptor.get_structure(structure)

    def __call__(self, structure, folder, converter_graph=None):
        structure = self._to_structure(structure)
        default_dtype = self.default_dtype
        return get_graph(
            np.asarray(structure.cart_coords, dtype=default_dtype),
            np.asarray(structure.frac_coords, dtype=default_dtype),
            np.asarray(structure.atomic_numbers, dtype=np.int64),
            os.path.basename(folder),
            r=self.radius,
            max_num_nbr=self.max_num_nbr,
            numerical_tol=1e-8,
            lattice=np.asarray(structure.lattice.matrix, dtype=default_dtype),
            default_dtype=default_dtype,
            tb_folder=folder,
            interface=self.interface,
            num_l=self.num_l,
            create_from_DFT=self.create_from_DFT,
            if_lcmp_graph=self.if_lcmp_graph,
            separate_onsite=self.separate_onsite,
            target=self.target,
            huge_structure=self.huge_structure,
            if_new_sp=self.if_new_sp,
            converter_graph=converter_graph,
        )
