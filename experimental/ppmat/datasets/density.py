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


import json
import os
import time

import numpy as np
import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.infgcn.paddle_utils import *

from .paddle_geometric_data import Data

Bohr = 0.529177


def pbc_expand(atom_type, atom_coord):
    """
    Expand the atoms by periodic boundary condition to eight directions in the neighboring cells.
    :param atom_type: atom types, tensor of shape (n_atom,)
    :param atom_coord: atom coordinates, tensor of shape (n_atom, 3)
    :return: expanded atom types and coordinates
    """
    exp_type, exp_coord = [], []
    exp_direction = paddle.to_tensor(
        data=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype="float32",
    )
    for a_type, a_coord in zip(atom_type, atom_coord):
        for direction in exp_direction:
            new_coord = a_coord + direction
            if (new_coord <= 1).astype("bool").all():
                exp_type.append(a_type)
                exp_coord.append(new_coord)
    return paddle.to_tensor(data=exp_type, dtype="int64"), paddle.stack(
        x=exp_coord, axis=0
    )


def rotate_voxel(shape, cell, density, rotated_grid):
    """
    Rotate the volumetric data using trilinear interpolation.
    :param shape: voxel shape, tensor of shape (3,)
    :param cell: cell vectors, tensor of shape (3, 3)
    :param density: original density, tensor of shape (n_grid,)
    :param rotated_grid: rotated grid coordinates, tensor of shape (n_grid, 3)
    :return: rotated density, tensor of shape (n_grid,)
    """
    density = density.view(1, 1, *shape)
    rotated_grid = rotated_grid.view(1, *shape, 3)
    shape = paddle.to_tensor(data=shape, dtype="float32")
    grid_cell = cell / shape.view(3, 1)
    normalized_grid = (
        2 * rotated_grid @ paddle.linalg.inv(x=grid_cell) - shape + 1
    ) / (shape - 1)
    return paddle.nn.functional.grid_sample(
        x=density,
        grid=paddle.flip(x=normalized_grid, axis=[-1]),
        mode="bilinear",
        align_corners=False,
    ).view(-1)


class DensityDataset(paddle.io.Dataset):
    def __init__(
        self,
        root,
        split,
        split_file,
        atom_file,
        extension="CHGCAR",
        compression="lz4",
        rotate=False,
        pbc=False,
    ):
        """
        The density dataset contains volumetric data of molecules.
        :param root: data root
        :param split: data split, can be 'train', 'validation', 'test'
        :param split_file: the data split file containing file names of the split
        :param atom_file: atom information file
        :param extension: raw data file extension, can be 'CHGCAR', 'cube', 'json'
        :param compression: raw data compression, can be 'lz4', 'xz', or None (no compression)
        :param rotate: whether to rotate the molecule and the volumetric data
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(DensityDataset, self).__init__()
        self.root = root
        self.split = split
        self.extension = extension
        self.compression = compression
        self.rotate = rotate
        self.pbc = pbc
        self.file_pattern = f".{extension}"
        if compression is not None:
            self.file_pattern += f".{compression}"
        with open(os.path.join(root, split_file)) as f:
            self.file_list = list(reversed(json.load(f)[split]))
        with open(atom_file) as f:
            atom_info = json.load(f)
        atom_list = [info["name"] for info in atom_info]
        self.atom_name2idx = {name: idx for idx, name in enumerate(atom_list)}
        self.atom_name2idx.update(
            {name.encode(): idx for idx, name in enumerate(atom_list)}
        )
        self.atom_num2idx = {
            info["atom_num"]: idx for idx, info in enumerate(atom_info)
        }
        self.idx2atom_num = {
            idx: info["atom_num"] for idx, info in enumerate(atom_info)
        }
        if extension == "CHGCAR":
            self.read_func = self.read_chgcar
        elif extension == "cube":
            self.read_func = self.read_cube
        elif extension == "json":
            self.read_func = self.read_json
        else:
            raise TypeError(f"Unknown extension {extension}")
        if compression == "lz4":
            import lz4.frame

            self.open = lz4.frame.open
        elif compression == "xz":
            import lzma

            self.open = lzma.open
        else:
            self.open = open

    def __getitem__(self, item):
        if self.compression == "lz4":
            file_name = f"{(self.file_list[item]+1):06}{self.file_pattern}"
        else:
            file_name = f"{(self.file_list[item])}{self.file_pattern}"

        try:
            with self.open(os.path.join(self.root, file_name)) as f:
                g, density, grid_coord, info = self.read_func(f)
        except EOFError:
            print("EOFError")
            print(f"Error reading {file_name} in {self.split} set, try again")
        except RuntimeError:
            print(f"Error reading {file_name} in {self.split} set")
            raise

        info["file_name"] = file_name
        if self.rotate:
            rot = o3.rand_matrix()
            center = info["cell"].sum(axis=0) / 2
            g.pos = (g.pos - center) @ rot.t() + center
            rotated_grid = (grid_coord - center) @ rot + center
            density = rotate_voxel(info["shape"], info["cell"], density, rotated_grid)
            info["rot"] = rot
        return g, density, grid_coord, info

    def __len__(self):
        return len(self.file_list)

    def read_cube(self, fileobj):
        """Read atoms and data from CUBE file."""
        if self.pbc:
            raise NotImplementedError("PBC not implemented for cube files")
        readline = fileobj.readline
        readline()
        readline()
        line = readline().split()
        n_atom = int(line[0])
        origin = paddle.to_tensor(data=[float(x) for x in line[1:]], dtype="float32")
        shape = []
        cell = paddle.empty(shape=[3, 3], dtype="float32")
        for i in range(3):
            n, x, y, z = [float(s) for s in readline().split()]
            shape.append(int(n))
            cell[i] = paddle.to_tensor(data=[x, y, z], dtype="float32")
        x_coord = paddle.arange(end=shape[0]).unsqueeze(axis=-1) * cell[0]
        y_coord = paddle.arange(end=shape[1]).unsqueeze(axis=-1) * cell[1]
        z_coord = paddle.arange(end=shape[2]).unsqueeze(axis=-1) * cell[2]
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        grid_coord = grid_coord.view(-1, 3) - origin
        atom_type = paddle.empty(shape=n_atom, dtype="int64")
        atom_coord = paddle.empty(shape=[n_atom, 3], dtype="float32")
        for i in range(n_atom):
            line = readline().split()
            atom_type[i] = self.atom_num2idx[int(line[0])]
            atom_coord[i] = paddle.to_tensor(
                data=[float(s) for s in line[2:]], dtype="float32"
            )
        g = Data(x=atom_type, pos=atom_coord)
        density = paddle.to_tensor(
            data=[float(s) for s in fileobj.read().split()], dtype="float32"
        )
        return g, density, grid_coord, {"shape": shape, "cell": cell, "origin": origin}

    def read_chgcar(self, fileobj):
        """Read atoms and data from CHGCAR file."""
        readline = fileobj.readline
        readline()
        scale = float(readline())
        cell = paddle.empty(shape=[3, 3], dtype="float32")
        for i in range(3):
            cell[i] = paddle.to_tensor(
                data=[float(s) for s in readline().split()], dtype="float32"
            )
        cell = cell * scale
        elements = readline().split()
        n_atoms = [int(s) for s in readline().split()]
        readline()
        tot_atoms = sum(n_atoms)
        atom_type = paddle.empty(shape=[tot_atoms], dtype="int64")
        atom_coord = paddle.empty(shape=[tot_atoms, 3], dtype="float32")
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type[idx : idx + n] = self.atom_name2idx[elem]
            for _ in range(n):
                atom_coord[idx] = paddle.to_tensor(
                    data=[float(s) for s in readline().split()], dtype="float32"
                )
                idx += 1
        if self.pbc:
            atom_type, atom_coord = pbc_expand(atom_type, atom_coord)
        atom_coord = atom_coord @ cell
        g = Data(x=atom_type, pos=atom_coord)
        readline()
        shape = [int(s) for s in readline().split()]
        n_grid = shape[0] * shape[1] * shape[2]
        x_coord = (
            paddle.linspace(start=0, stop=shape[0] - 1, num=shape[0]).unsqueeze(axis=-1)
            / shape[0]
            * cell[0]
        )
        y_coord = (
            paddle.linspace(start=0, stop=shape[1] - 1, num=shape[1]).unsqueeze(axis=-1)
            / shape[1]
            * cell[1]
        )
        z_coord = (
            paddle.linspace(start=0, stop=shape[2] - 1, num=shape[2]).unsqueeze(axis=-1)
            / shape[2]
            * cell[2]
        )
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        grid_coord = grid_coord.view(-1, 3)
        density = paddle.to_tensor(
            data=[float(s) for s in fileobj.read().split()[:n_grid]], dtype="float32"
        )
        volume = paddle.linalg.det(x=cell).abs()
        density = density / volume
        density = (
            density.view(shape[2], shape[1], shape[0])
            .transpose(
                perm=dim2perm(density.view(shape[2], shape[1], shape[0]).ndim, 0, 2)
            )
            .contiguous()
            .view(-1)
        )
        return g, density, grid_coord, {"shape": shape, "cell": cell}

    def read_json(self, fileobj):
        """Read atoms and data from JSON file."""

        def read_2d_tensor(s):
            return paddle.to_tensor(
                data=[[float(x) for x in line] for line in s], dtype="float32"
            )

        data = json.load(fileobj)
        scale = float(data["vector"][0][0])
        cell = read_2d_tensor(data["lattice"][0]) * scale
        elements = data["elements"][0]
        n_atoms = [int(s) for s in data["elements_number"][0]]
        tot_atoms = sum(n_atoms)
        atom_coord = read_2d_tensor(data["coordinates"][0])
        atom_type = paddle.empty(shape=[tot_atoms], dtype="int64")
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type[idx : idx + n] = self.atom_name2idx[elem]
            idx += n
        if self.pbc:
            atom_type, atom_coord = pbc_expand(atom_type, atom_coord)
        atom_coord = atom_coord @ cell
        g = Data(x=atom_type, pos=atom_coord)
        shape = [int(s) for s in data["FFTgrid"][0]]
        x_coord = (
            paddle.linspace(start=0, stop=shape[0] - 1, num=shape[0]).unsqueeze(axis=-1)
            / shape[0]
            * cell[0]
        )
        y_coord = (
            paddle.linspace(start=0, stop=shape[1] - 1, num=shape[1]).unsqueeze(axis=-1)
            / shape[1]
            * cell[1]
        )
        z_coord = (
            paddle.linspace(start=0, stop=shape[2] - 1, num=shape[2]).unsqueeze(axis=-1)
            / shape[2]
            * cell[2]
        )
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        grid_coord = grid_coord.view(-1, 3)
        n_grid = shape[0] * shape[1] * shape[2]
        n_line = (n_grid + 9) // 10
        density = paddle.to_tensor(
            data=[
                (float(s) if not s.startswith("*") else 0.0)
                for line in data["chargedensity"][0][:n_line]
                for s in line
            ],
            dtype="float32",
        ).view(-1)[:n_grid]
        volume = paddle.linalg.det(x=cell).abs()
        density = density / volume
        density = (
            density.view(shape[2], shape[1], shape[0])
            .transpose(
                perm=dim2perm(density.view(shape[2], shape[1], shape[0]).ndim, 0, 2)
            )
            .contiguous()
            .view(-1)
        )
        return g, density, grid_coord, {"shape": shape, "cell": cell}

    def write_cube(self, fileobj, atom_type, atom_coord, density, info):
        """Write a cube file."""
        fileobj.write("Cube file written on " + time.strftime("%c"))
        fileobj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        cell = info["cell"]
        shape = info["shape"]
        origin = info.get("origin", np.zeros(3))
        fileobj.write(
            "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(len(atom_type), *origin)
        )
        for s, c in zip(shape, cell):
            d = c / s
            fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(s, *d))
        for Z, (x, y, z) in zip(atom_type, atom_coord):
            Z = self.idx2atom_num[Z]
            fileobj.write(
                "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format(Z, Z, x, y, z)
            )
        density.tofile(fileobj, sep="\n", format="%e")
