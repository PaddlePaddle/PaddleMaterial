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


import gzip
import lzma
import time
from pathlib import Path

import numpy as np
import paddle


def write_cube_generic(
    fileobj, atom_type, atom_coord, density, info, idx2atom_num=None
):
    """Write a minimal Gaussian CUBE file for field prediction outputs."""
    fileobj.write("Cube file written on " + time.strftime("%c"))
    fileobj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    cell = info["cell"]
    shape = info["shape"]
    origin = info.get("origin", np.zeros(3, dtype=np.float32))
    fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(len(atom_type), *origin))
    for s, c in zip(shape, cell):
        d = c / s
        fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(s, *d))
    for Z, (x, y, z) in zip(atom_type, atom_coord):
        atomic_num = int(idx2atom_num[int(Z)]) if idx2atom_num is not None else int(Z)
        fileobj.write(
            "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format(
                atomic_num, float(atomic_num), x, y, z
            )
        )
    density.tofile(fileobj, sep="\n", format="%e")


def unavailable_cube_writer(*args, **kwargs):
    raise AttributeError("Cube writer not available for this dataset")


def open_text_maybe_compressed(path):
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(path, mode="rt")
    if suffixes.endswith(".xz"):
        return lzma.open(path, mode="rt")
    if suffixes.endswith(".gz"):
        return gzip.open(path, mode="rt")
    return path.open(mode="rt")


def read_cube_density(path):
    with open_text_maybe_compressed(path) as f:
        f.readline()
        f.readline()
        line = f.readline().split()
        if len(line) < 4:
            raise ValueError(f"Invalid CUBE header (line 3) in {path}")
        n_atom = int(line[0])
        origin = np.array([float(x) for x in line[1:4]], dtype=np.float32)

        shape = []
        cell = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            row = f.readline().split()
            if len(row) < 4:
                raise ValueError(f"Invalid CUBE axis line in {path}")
            n, x, y, z = [float(s) for s in row[:4]]
            shape.append(int(n))
            cell[i] = np.array([x, y, z], dtype=np.float32)

        x_coord = np.arange(shape[0], dtype=np.float32)[:, None] * cell[0][None, :]
        y_coord = np.arange(shape[1], dtype=np.float32)[:, None] * cell[1][None, :]
        z_coord = np.arange(shape[2], dtype=np.float32)[:, None] * cell[2][None, :]
        grid_coord = (
            x_coord.reshape(-1, 1, 1, 3)
            + y_coord.reshape(1, -1, 1, 3)
            + z_coord.reshape(1, 1, -1, 3)
        ).reshape(-1, 3)
        grid_coord = grid_coord + origin

        atom_coord_ref = []
        for _ in range(n_atom):
            row = f.readline().split()
            if len(row) < 5:
                raise ValueError(f"Invalid CUBE atom line in {path}")
            atom_coord_ref.append([float(row[2]), float(row[3]), float(row[4])])

        n_grid = shape[0] * shape[1] * shape[2]
        vals = []
        for line in f:
            parts = line.split()
            if parts:
                vals.extend(parts)
        if len(vals) < n_grid:
            raise ValueError(
                f"CUBE data too short in {path}: expect {n_grid}, got {len(vals)}"
            )
        density = np.array(vals[:n_grid], dtype=np.float32)

    return (
        paddle.to_tensor(density, dtype="float32"),
        paddle.to_tensor(grid_coord, dtype="float32"),
        {
            "shape": shape,
            "cell": paddle.to_tensor(cell, dtype="float32"),
            "origin": paddle.to_tensor(origin, dtype="float32"),
            "atom_coord_ref": np.asarray(atom_coord_ref, dtype=np.float32),
        },
    )


def prepare_info_cube(info, grid_coord):
    info_cube = {}
    shape = info.get("shape")
    cell = info.get("cell")
    origin = info.get("origin", None)
    grid_np_full = grid_coord.detach().cpu().numpy()

    if shape is not None and len(shape) == 3:
        try:
            shape_i = [int(s) for s in shape]
            grid_view = grid_np_full.reshape(shape_i[0], shape_i[1], shape_i[2], 3)
            origin_np = grid_view[0, 0, 0]
            step_x = (
                grid_view[1, 0, 0] - grid_view[0, 0, 0]
                if shape_i[0] > 1
                else np.zeros(3, dtype=np.float32)
            )
            step_y = (
                grid_view[0, 1, 0] - grid_view[0, 0, 0]
                if shape_i[1] > 1
                else np.zeros(3, dtype=np.float32)
            )
            step_z = (
                grid_view[0, 0, 1] - grid_view[0, 0, 0]
                if shape_i[2] > 1
                else np.zeros(3, dtype=np.float32)
            )
            cell_from_grid = np.stack(
                [step_x * shape_i[0], step_y * shape_i[1], step_z * shape_i[2]], axis=0
            )
        except Exception:
            origin_np = None
            cell_from_grid = None
    else:
        origin_np = None
        cell_from_grid = None

    if shape is not None:
        info_cube["shape"] = [int(s) for s in shape]
    if cell is not None:
        if hasattr(cell, "numpy"):
            info_cube["cell"] = cell.numpy()
        else:
            info_cube["cell"] = np.array(cell, dtype=np.float32)
    if cell_from_grid is not None:
        info_cube["cell"] = cell_from_grid
    if origin is not None:
        if hasattr(origin, "numpy"):
            info_cube["origin"] = origin.numpy()
        else:
            info_cube["origin"] = np.array(origin, dtype=np.float32)
    if origin_np is not None:
        info_cube["origin"] = origin_np
    return info_cube
