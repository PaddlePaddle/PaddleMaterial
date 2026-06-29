# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import datetime
import gzip
import hashlib
import json
import lzma
import os
import os.path as osp
import time
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import paddle

from ppmat.utils import logger


def count_samples_json_lines(path: str):
    """Fast count of samples in a line-delimited JSON file."""
    with open(path, "r") as f:
        return sum(1 for _ in f)


def read_json_lines(path):
    """
    Read all lines from a line-delimited JSON file,
    extracting all properties into a dictionary of lists.
    """
    property_data = {}

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            content = ast.literal_eval(line.strip())
            # if idx == 301:
            #     break
            if idx == 0:
                all_property_names = list(content.keys())
                # print("all_property_names:", all_property_names)
                property_data = {name: [] for name in all_property_names}

            for property_name in all_property_names:
                if property_name not in content:
                    raise ValueError(
                        f"'{property_name}' not found in line {idx + 1} of file"
                    )
                property_data[property_name].append(content[property_name])
    return property_data


def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "r") as f:
        content = json.load(f)
    return content


def list_files_by_suffix(path: str, suffix: str) -> List[str]:
    """List files under path with the given suffix."""
    if not osp.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    file_names = sorted(
        file_name for file_name in os.listdir(path) if file_name.endswith(suffix)
    )
    if not file_names:
        raise FileNotFoundError(f"No files ending with {suffix} found under {path}.")
    return file_names


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    def handler(obj: object) -> (int | object):
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=handler)


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


def calc_md5(fullname):
    md5 = hashlib.md5()
    fullname = os.path.expanduser(fullname)
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    return calc_md5sum


def append_timestamp_to_output_dir(
    config,
    now: Optional[datetime.datetime] = None,
):
    seed = config["Trainer"].get("seed", 42)
    timestamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_output_dir = config["Trainer"]["output_dir"]
    config["Trainer"]["output_dir"] = f"{base_output_dir}_t_{timestamp}_s_{seed}"
    return config


def find_file_in_package(package_path: str, file_name: str):
    logger.debug(f"Find file {file_name} in package path: {package_path}")
    if osp.isfile(package_path):
        if osp.basename(package_path) == file_name:
            logger.debug(f"Find file: {package_path}")
            return package_path
        logger.debug(f"No such file named {file_name} in {package_path}")
        raise FileNotFoundError(f"No such file named {file_name} in {package_path}")

    for root, _, files in os.walk(package_path):
        for name in files:
            if osp.basename(name) == file_name:
                file_path = osp.join(root, name)
                logger.debug(f"Find file: {file_path}")
                return file_path

    logger.debug(f"No such file named {file_name} in {package_path}")
    raise FileNotFoundError(f"No such file named {file_name} in {package_path}")


def find_config_file_in_package(model_name: str, package_path: str):
    logger.debug(f"Find config file for model {model_name} in {package_path}")
    for config_name in (f"{model_name}.yaml", f"{model_name}.yml"):
        try:
            return find_file_in_package(package_path, config_name)
        except FileNotFoundError:
            pass

    find_list = []
    for root, _, files in os.walk(package_path):
        for name in files:
            if name.endswith(".yaml") or name.endswith(".yml"):
                find_list.append(osp.join(root, name))

    if len(find_list) == 1:
        config_path = find_list[0]
        logger.warning(f"Find config file: {config_path}, using this file.")
        return config_path

    raise ValueError(f"Multiple yaml files found: {find_list}, must be only one")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MD5 hash of a file")
    parser.add_argument("filename", help="Path to the file to hash")
    args = parser.parse_args()

    md5 = calc_md5(args.filename)
    print(md5)
