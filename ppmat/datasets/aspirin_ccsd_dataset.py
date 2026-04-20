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

import os
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from ase import units
from ase.io import read
from tqdm import tqdm

from ppmat.datasets.geometric_data_type import Data
from ppmat.datasets.geometric_data_type import Dataset

units.__setattr__("kcal/mol", units.kcal / units.mol)
units.__setattr__("kJ/mol", units.kJ / units.mol)


class AspirinCcsdDataset(Dataset):
    """
    This class is an dataset for molecular data.

    Args:
        root (str): The root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a
        data object and returns a transformed version. The data object will
        be transformed before every access. Default: None.
        pre_transform (callable, optional): A function/transform that takes
        in a data object and returns a transformed version. The data object
        will be transformed before being saved to disk. Default: None.
        pre_filter (callable, optional): A function that takes in a data object
        and returns a boolean value, indicating whether the data object should
        be included in the final dataset. Default: None.
        precision (paddle.dtype): The precision of the data. Default: paddle.float32.
    """

    def __init__(
        self,
        data_length_unit: str = "Ang",
        data_energy_unit: str = "eV",
        **kwargs,
    ) -> None:
        self.precision = kwargs.pop("precision", paddle.float32)
        dataset_size = kwargs.pop("dataset_size", None)
        self.units = {
            "length": getattr(units, data_length_unit),
            "energy": getattr(units, data_energy_unit),
        }
        super().__init__(**kwargs)
        self.data = paddle.load(self.processed_paths[0])
        if dataset_size is not None:
            self.data = self.data[:dataset_size]
        self._indices: Optional[Sequence] = None

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        names = [
            name
            for name in os.listdir(self.raw_dir)
            if name.endswith((".npz", ".xyz", ".extxyz"))
        ]
        return names

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self) -> None:
        data_list = []
        data_path = self.processed_paths[0]
        for raw_path in tqdm(self.raw_paths):
            if raw_path.endswith(".xyz") or raw_path.endswith(".extxyz"):
                data_list.extend(
                    parse_xyz(
                        raw_path,
                        self.pre_transform,
                        self.pre_filter,
                        self.precision,
                        self.units,
                    )
                )
        paddle.save(data_list, data_path)

    def len(self) -> int:
        return len(self.data)

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices

    def get(self, idx: int):
        return self.data[idx]


def parse_xyz(
    raw_path: str,
    pre_transform: Callable,
    pre_filter: Callable,
    precision: paddle.dtype,
    units: dict,
) -> List[Data]:
    data_list = []
    atoms_list = read(raw_path, index=":")
    for atoms in atoms_list:
        atoms.set_constraint()
        z = paddle.from_numpy(atoms.get_atomic_numbers()).int()
        pos = paddle.from_numpy(atoms.get_positions(wrap=True)).to(dtype=precision)
        cell = paddle.from_numpy(atoms.get_cell().array).to(dtype=precision)
        pbc = paddle.from_numpy(atoms.get_pbc()).bool()
        cell[~pbc] = 0.0
        energy = paddle.tensor(atoms.get_potential_energy(), dtype=precision)
        forces = paddle.from_numpy(atoms.get_forces()).to(dtype=precision)

        z = z.reshape(-1)
        pos = pos.reshape(-1, 3) * units["length"]
        cell = cell.reshape(1, 3, 3) * units["length"]
        energy = energy.reshape(1) * units["energy"]
        force = forces.reshape(-1, 3) * units["energy"] / units["length"]
        data = {
            "z": z,
            "pos": pos,
            "cell": cell,
            "energy": energy,
            "force": force,
        }
        if pre_filter is not None and not pre_filter(data):
            continue
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)
    return data_list
