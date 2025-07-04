# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numbers

import numpy as np
import paddle

from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.num_atom_dists import NUM_ATOMS_DISTRIBUTIONS
from ppmat.utils import paddle_aux  # noqa


class NumAtomsCrystalDataset(paddle.io.Dataset):
    def __init__(
        self,
        total_num,
        formula=None,
        dist_name="ALEX_MP_20",
        prop_names=None,
        prop_values=None,
    ):
        super().__init__()

        self.total_num = total_num
        self.formula = formula
        if formula is not None:
            # self.chem_list = self.get_structure(formula)
            raise NotImplementedError()
        else:
            self.chem_list = None

            assert dist_name in NUM_ATOMS_DISTRIBUTIONS
            distribution = NUM_ATOMS_DISTRIBUTIONS.get(dist_name)
            self.distribution = distribution

            self.num_atoms = np.random.choice(
                list(self.distribution.keys()),
                size=total_num,
                p=list(self.distribution.values()),
            )
        self.prop_names = prop_names
        self.prop_values = prop_values
        if prop_names is not None and prop_values is not None:
            assert len(prop_names) == len(prop_values)
            self.prop_flag = True
        else:
            self.prop_flag = False

    # def get_structure(self, formula):
    #     composition = chemparse.parse_formula(formula)
    #     chem_list = []
    #     for elem in composition:
    #         num_int = int(composition[elem])
    #         chem_list.extend([DEFAULT_ELEMENTS.index(elem)] * num_int)
    #     return chem_list

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        data = {}
        if self.chem_list is None:
            num_atom = self.num_atoms[index]
            data["structure_array"] = {
                "num_atoms": ConcatData(np.array([num_atom])),
            }

        else:
            data["structure_array"] = {
                "num_atoms": ConcatData(np.array([len(self.chem_list)])),
                "atom_types": ConcatData(np.array(self.chem_list)),
            }

        data["id"] = index

        if self.prop_flag:
            for prop_name, prop_value in zip(self.prop_names, self.prop_values):
                if isinstance(prop_value, numbers.Number):
                    data[prop_name] = np.array([prop_value]).astype("float32")
                else:
                    data[prop_name] = prop_value
        return data
