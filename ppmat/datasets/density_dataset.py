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


import json
import os
import os.path as osp
import pickle
import time
import math

import numpy as np
import paddle
import paddle.distributed as dist

from ppmat.models.common.e3nn import o3
from ppmat.utils.paddle_aux import dim2perm
from ppmat.utils.misc import is_equal
from ppmat.utils import download
from ppmat.utils import logger

from ppmat.datasets.geometric_data_type.data import Data


Bohr = 0.529177


class DensityDataset(paddle.io.Dataset):
    """Density Dataset Handler

    Overview
    --------
    Generic volumetric electron-density reader with optional auto-download and
    cache. Supports CHGCAR / cube / json files (optionally compressed) with
    element metadata from ``atom_file``. Caching mirrors the mp20 dataset
    pattern: parsed samples are serialized for fast reuse and validated against
    config to avoid stale artifacts.

    Dataset layout
    --------------
    ``root/`` is expected to contain raw density files; ``split_file`` lists
    filenames for each split. Atom dictionary is provided via ``atom_file``.
    Compression is handled transparently based on the filename suffix.

    Auto-download
    -------------
    If ``root`` is missing and ``url`` is given (``auto_download=True``), the
    archive is fetched using :func:`ppmat.utils.download.get_datasets_path_from_url`
    (optional ``md5`` checksum).

    Caching
    -------
    Enable via ``enable_cache=True``. Samples are pickled under
    ``cache_path`` (default ``<root>_cache/<split>``) with a saved config. Set
    ``overwrite=True`` to rebuild or ``max_cache_samples`` to limit cached
    entries.

    Attributes for registry parity with other handlers
    --------------------------------------------------
    - name (str): optional dataset name for downstream reference
    - url (str): download URL if provided
    - md5 (str): md5 for download verification if provided

    Common ES datasets (paths/URLs)
    --------------------------------
    - QM9_ES: root ``dataset_ES/data_qm9``; data
      ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_es.tar``;
      atom dict ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9.json``;
      split file ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json``.
    - MP_ES (cubic): root ``dataset_ES/data_cubic``; data
      ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar``;
      atom dict ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json``;
      split file ``https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json``.
    """

    # Optional registry fields; can be overridden per config or subclass
    name: str | None = None
    url: str | None = None
    md5: str | None = None
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
        url=None,
        md5=None,
        auto_download=True,
        enable_cache=False,
        cache_path=None,
        overwrite=False,
        max_cache_samples=None,
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
        :param url: optional remote url for auto-download when root is missing
        :param md5: optional md5 for downloaded archive
        :param auto_download: download dataset automatically if root does not exist
        :param enable_cache: if True, read/save parsed samples to cache as pickle
        :param cache_path: cache directory; defaults to "<root>_cache/<split>"
        :param overwrite: force rebuilding cache even if it exists
        :param max_cache_samples: limit number of samples to cache (None means all)
        """
        super(DensityDataset, self).__init__()
        # Prefer explicit url/md5 from arguments; fall back to class attributes.
        dl_url = url if url is not None else self.url
        dl_md5 = md5 if md5 is not None else self.md5
        self.root = self._maybe_download_root(root, dl_url, dl_md5, auto_download)
        self.split = split
        self.extension = extension
        self.compression = compression
        self.rotate = rotate
        self.pbc = pbc
        self.enable_cache = enable_cache
        self.overwrite = overwrite
        self.max_cache_samples = max_cache_samples
        # Cache root, by default <root>_cache/<split>
        self.cache_path = (
            cache_path
            if cache_path is not None
            else osp.join(f"{self.root}_cache", split)
        )
        self.file_pattern = f".{extension}"
        if compression is not None:
            self.file_pattern += f".{compression}"
        with open(os.path.join(self.root, split_file)) as f:
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

        self._prepare_cache()

    def _maybe_download_root(self, root, url, md5, auto_download):
        if osp.exists(root):
            return root
        if not auto_download or url is None:
            logger.warning(
                f"Dataset root {root} not found and auto_download disabled or url missing."
            )
            return root
        logger.message(f"Dataset root {root} not found. Downloading from {url}.")
        downloaded_root = download.get_datasets_path_from_url(url, md5)
        logger.message(f"Downloaded dataset to {downloaded_root}")
        return downloaded_root

    def _prepare_cache(self):
        self.cache_samples = []
        if not self.enable_cache:
            return

        sample_dir = osp.join(self.cache_path, "samples")
        cache_cfg_path = osp.join(self.cache_path, "dataset_cfg.pkl")
        cache_exists = osp.exists(sample_dir)
        expected_cfg = {
            "root": self.root,
            "split": self.split,
            "extension": self.extension,
            "compression": self.compression,
            "pbc": self.pbc,
            "rotate": self.rotate,
            "file_pattern": self.file_pattern,
        }

        # If cache exists and no overwrite request, ensure configuration matches.
        if cache_exists and not self.overwrite:
            try:
                cached_cfg = self._load_from_cache(cache_cfg_path)
                if not is_equal(cached_cfg, expected_cfg):
                    logger.warning(
                        "Cache configuration differs from current settings, will rebuild."
                    )
                    self.overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning("Failed to read cached config, will rebuild.")
                self.overwrite = True

        # Rebuild cache when missing or outdated.
        if self.overwrite or not cache_exists:
            self._build_cache(sample_dir, cache_cfg_path, expected_cfg)

        if dist.is_initialized():
            dist.barrier()

        if osp.exists(sample_dir):
            self.cache_samples = sorted(
                [
                    osp.join(sample_dir, f)
                    for f in os.listdir(sample_dir)
                    if f.endswith(".pkl")
                ]
            )
        else:
            self.cache_samples = []

    def _build_cache(self, sample_dir, cache_cfg_path, expected_cfg):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        os.makedirs(sample_dir, exist_ok=True)
        self._save_to_cache(cache_cfg_path, expected_cfg)

        # Cap number of cached samples (default: cache all).
        num_to_cache = (
            len(self.file_list)
            if self.max_cache_samples is None
            else min(len(self.file_list), int(self.max_cache_samples))
        )
        logger.message(
            f"Caching {num_to_cache}/{len(self.file_list)} samples to {sample_dir}"
        )
        for idx in range(num_to_cache):
            file_name = self._resolve_file_name(idx)
            try:
                g, density, grid_coord, info = self._read_sample(file_name)
            except Exception as e:
                logger.warning(f"Failed to cache {file_name}: {e}")
                continue
            payload = self._serialize_sample(g, density, grid_coord, info)
            self._save_to_cache(osp.join(sample_dir, f"{idx:010d}.pkl"), payload)
        logger.info(f"Finished caching samples to {sample_dir}")

    def _serialize_sample(self, g, density, grid_coord, info):
        """将一次读取的结果打包为可pickle的轻量对象。"""
        payload = {
            "atom_type": np.asarray(g.x),
            "atom_coord": np.asarray(g.pos),
            "density": np.asarray(density),
            "grid_coord": np.asarray(grid_coord),
            "info": dict(info) if isinstance(info, dict) else info,
        }
        if isinstance(payload["info"], dict) and "cell" in payload["info"]:
            try:
                payload["info"]["cell"] = np.asarray(payload["info"]["cell"])
            except Exception:
                pass
        return payload

    def _deserialize_sample(self, payload):
        """从缓存还原张量与 Data 对象。"""
        atom_type = paddle.to_tensor(payload["atom_type"], dtype="int64")
        atom_coord = paddle.to_tensor(payload["atom_coord"], dtype="float32")
        density = paddle.to_tensor(payload["density"], dtype="float32")
        grid_coord = paddle.to_tensor(payload["grid_coord"], dtype="float32")
        info = payload.get("info", {})
        if isinstance(info, dict) and "cell" in info:
            info["cell"] = paddle.to_tensor(info["cell"], dtype="float32")
        g = Data(x=atom_type, pos=atom_coord)
        return g, density, grid_coord, info

    def _save_to_cache(self, cache_path: str, data):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _resolve_file_name(self, item):
        if self.compression == "lz4":
            return f"{(self.file_list[item]+1):06}{self.file_pattern}"
        return f"{(self.file_list[item])}{self.file_pattern}"

    def _read_sample(self, file_name):
        with self.open(os.path.join(self.root, file_name)) as f:
            g, density, grid_coord, info = self.read_func(f)
        info["file_name"] = file_name
        return g, density, grid_coord, info

    def __getitem__(self, item):
        file_name = self._resolve_file_name(item)

        # 优先从缓存读取，缓存缺失/异常时回退到原始文件
        if self.enable_cache and self.cache_samples and item < len(self.cache_samples):
            try:
                payload = self._load_from_cache(self.cache_samples[item])
                g, density, grid_coord, info = self._deserialize_sample(payload)
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_name}: {e}")
                g, density, grid_coord, info = self._read_sample(file_name)
        else:
            g, density, grid_coord, info = self._read_sample(file_name)

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
        x_coord = paddle.multiply(paddle.arange(end=shape[0], dtype="float32").unsqueeze(axis=-1), cell[0])
        y_coord = paddle.multiply(paddle.arange(end=shape[1], dtype="float32").unsqueeze(axis=-1), cell[1])
        z_coord = paddle.multiply(paddle.arange(end=shape[2], dtype="float32").unsqueeze(axis=-1), cell[2])
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        # In the CUBE format the origin marks the starting voxel; add it to the
        # axis-aligned coordinates so grid points align with atom positions.
        grid_coord = grid_coord.view(-1, 3) + origin
        atom_type = paddle.empty(shape=paddle.to_tensor(n_atom), dtype="int64")
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
        arr = _read_density_stream(fileobj, n_grid)
        density = paddle.to_tensor(arr, dtype="float32")

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


def _read_density_stream(fileobj, n_grid: int, chunk_tokens: int = 1_000_000):
    out = np.empty(n_grid, dtype=np.float32)
    filled = 0
    buf = []

    def _as_float(tok):
        try:
            return float(tok)
        except Exception:
            return math.nan  # 非数字用 NaN 标记

    for line in fileobj:
        if filled >= n_grid:
            break
        parts = line.split()
        if not parts:
            continue
        # 过滤出数字
        nums = [_as_float(t) for t in parts]
        # 丢弃 NaN（非数字 token）
        nums = [x for x in nums if not math.isnan(x)]
        if not nums:
            continue

        take = min(len(nums), n_grid - filled)
        out[filled:filled+take] = np.array(nums[:take], dtype=np.float32)
        filled += take

    if filled != n_grid:
        raise ValueError(f"Expected {n_grid} density values, got {filled}")
    return out
