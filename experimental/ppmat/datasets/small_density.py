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


import os

import numpy as np
import paddle

from ppmat.models.infgcn.paddle_utils import * # noqa: F403

from .paddle_geometric_data import Data

ATOM_TYPES = {
    "benzene": paddle.to_tensor(
        data=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        dtype="int64",
        place=paddle.CPUPlace(),
    ),
    "ethanol": paddle.to_tensor(
        data=[0, 0, 2, 1, 1, 1, 1, 1, 1], dtype="int64", place=paddle.CPUPlace()
    ),
    "phenol": paddle.to_tensor(
        data=[0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1],
        dtype="int64",
        place=paddle.CPUPlace(),
    ),
    "resorcinol": paddle.to_tensor(
        data=[0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1],
        dtype="int64",
        place=paddle.CPUPlace(),
    ),
    "ethane": paddle.to_tensor(
        data=[0, 0, 1, 1, 1, 1, 1, 1], dtype="int64", place=paddle.CPUPlace()
    ),
    "malonaldehyde": paddle.to_tensor(
        data=[2, 0, 0, 0, 2, 1, 1, 1, 1], dtype="int64", place=paddle.CPUPlace()
    ),
}


class SmallDensityDataset(paddle.io.Dataset):
    def __init__(self, root, mol_name, split):
        """
        Density dataset for small molecules in the MD datasets.
        Note that the validation and test splits are the same.
        :param root: data root
        :param mol_name: name of the molecule
        :param split: data split, can be 'train', 'validation', 'test'
        """
        super(SmallDensityDataset, self).__init__()
        assert mol_name in (
            "benzene",
            "ethanol",
            "phenol",
            "resorcinol",
            "ethane",
            "malonaldehyde",
        )
        self.root = root
        self.mol_name = mol_name
        self.split = split
        if split == "validation":
            split = "test"
        self.n_grid = 50
        self.grid_size = 20.0
        self.data_path = os.path.join(root, mol_name, f"{mol_name}_{split}")
        self.atom_type = ATOM_TYPES[mol_name]
        if self.atom_type.place.is_gpu_place():
            self.atom_type = self.atom_type.cpu()
        self.atom_coords = paddle.to_tensor(
            data=np.load(os.path.join(self.data_path, "structures.npy")),
            dtype="float32",
            place=paddle.CPUPlace(),
        )
        self.densities = self._convert_fft(
            np.load(os.path.join(self.data_path, "dft_densities.npy"))
        )
        self.grid_coord = self._generate_grid()

    def _convert_fft(self, fft_coeff):
        print(f"Precomputing {self.split} density from FFT coefficients ...")
        fft_coeff = paddle.to_tensor(
            data=fft_coeff, dtype="float32", place=paddle.CPUPlace()
        ).to("complex64")
        d = fft_coeff.view(-1, self.n_grid, self.n_grid, self.n_grid)
        hf = self.n_grid // 2
        d[:, :hf] = (d[:, :hf] - d[:, hf:] * 1.0j) / 2
        d[:, hf:] = paddle.flip(x=d[:, 1 : hf + 1], axis=[1]).conj()
        d = paddle.fft.ifft(x=d, axis=1)
        d[:, :, :hf] = (d[:, :, :hf] - d[:, :, hf:] * 1.0j) / 2
        d[:, :, hf:] = paddle.flip(x=d[:, :, 1 : hf + 1], axis=[2]).conj()
        d = paddle.fft.ifft(x=d, axis=2)
        d[..., :hf] = (d[..., :hf] - d[..., hf:] * 1.0j) / 2
        d[..., hf:] = paddle.flip(x=d[..., 1 : hf + 1], axis=[3]).conj()
        d = paddle.fft.ifft(x=d, axis=3)
        result = paddle.flip(x=d.real().view(-1, self.n_grid**3), axis=[-1]).detach()

        if result.place.is_gpu_place():
            result = result.cpu()
        return result

    def _generate_grid(self):
        x = paddle.linspace(
            start=self.grid_size / self.n_grid, stop=self.grid_size, num=self.n_grid
        )

        if x.place.is_gpu_place():
            x = x.cpu()
        grid = paddle.stack(x=paddle.meshgrid(x, x, x), axis=-1).view(-1, 3).detach()

        if grid.place.is_gpu_place():
            grid = grid.cpu()
        return grid

    def __getitem__(self, item):

        cell = paddle.eye(num_rows=3) * self.grid_size
        if cell.place.is_gpu_place():
            cell = cell.cpu()
        info = {
            "cell": cell,
            "shape": [self.n_grid, self.n_grid, self.n_grid],
        }

        def ensure_cpu(tensor):
            if paddle.is_tensor(tensor) and tensor.place.is_gpu_place():
                return tensor.cpu()
            return tensor

        return (
            Data(x=ensure_cpu(self.atom_type), pos=ensure_cpu(self.atom_coords[item])),
            ensure_cpu(self.densities[item]),
            ensure_cpu(self.grid_coord),
            info,
        )

    def __len__(self):
        return self.atom_coords.shape[0]
