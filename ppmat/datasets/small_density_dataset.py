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

from __future__ import absolute_import
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.distributed as dist

from ppmat.datasets.geometric_data_type.data import Data
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal

ATOM_TYPES: Dict[str, np.ndarray] = {
    # Atom order matches legacy dataset: 0=C, 1=H, 2=O.
    "benzene": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "ethanol": np.array([0, 0, 2, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "phenol": np.array([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "resorcinol": np.array(
        [0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1], dtype="int64"
    ),
    "ethane": np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype="int64"),
    "malonaldehyde": np.array([2, 0, 0, 0, 2, 1, 1, 1, 1], dtype="int64"),
}


class SmallDensityDataset(paddle.io.Dataset):
    """MD17 small-molecule electron-density dataset with caching and auto-download.

    The dataset stores FFT-domain electron-density coefficients for six MD17
    molecules (benzene, ethanol, phenol, resorcinol, ethane, malonaldehyde).
    This handler mirrors :mod:`mp20_dataset` patterns: automatic download if the
    data root is missing, conversion through factory-style builders, and per-
    sample caching into ``.pkl`` files (structures, densities, metadata) to
    avoid recomputing FFT inversions on every run.

    Raw layout (after extracting ``md17_es.tar.gz``):
        root/
            <mol_name>/
                <mol_name>_train/{structures.npy,dft_densities.npy,...}
                <mol_name>_test/{structures.npy,dft_densities.npy,...}

    Args:
        root: Dataset root. If missing and ``auto_download`` is True, the MD17
            archive is pulled and unpacked automatically.
        mol_name: One of the supported MD17 molecule names.
        split: ``train``, ``validation`` (alias of ``test``), or ``test``.
        n_grid: Cube grid resolution per axis.
        grid_size: Physical box size (Angstrom) for the grid.
        cache_path: Optional cache directory. Defaults to ``<root>/<mol>/<mol>_<split>_cache``.
        overwrite: Force rebuilding cache even if it exists.
        transforms: Optional callable applied to the (g, density, grid, info)
            tuple before returning.
        auto_download: Download and extract the dataset if ``root`` is missing.
    """

    name = "md17_es"
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz"
    md5 = None

    def __init__(
        self,
        root: str = "./data/data_md",
        mol_name: str = "ethanol",
        split: str = "train",
        n_grid: int = 50,
        grid_size: float = 20.0,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        transforms: Optional[Callable] = None,
        auto_download: bool = True,
        **kwargs,  # compatibility with unused config fields
    ):
        super().__init__()
        del kwargs

        if mol_name not in ATOM_TYPES:
            raise ValueError(f"Unsupported molecule {mol_name}. Options: {list(ATOM_TYPES)}")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of ['train', 'validation', 'test']")

        self.mol_name = mol_name
        self.user_split = split
        self.split = "test" if split == "validation" else split
        self.n_grid = int(n_grid)
        self.grid_size = float(grid_size)
        self.transforms = transforms

        self.root = self._prepare_root(root, auto_download)
        self.data_path = osp.join(self.root, mol_name, f"{mol_name}_{self.split}")
        if not osp.exists(self.data_path):
            raise FileNotFoundError(
                f"Data path {self.data_path} not found. "
                "Please check the root path or set auto_download=True."
            )

        if cache_path is None:
            self.cache_path = osp.join(
                self.root, mol_name, f"{mol_name}_{self.split}_cache"
            )
        else:
            self.cache_path = cache_path

        cell_np = np.eye(3, dtype="float32") * self.grid_size
        self.cell = paddle.to_tensor(cell_np, place=paddle.CPUPlace())

        self.samples: List[str] = []
        self.grid_coord: paddle.Tensor
        self._prepare_cache(overwrite)
        self.num_samples = len(self.samples)

    def _prepare_root(self, root: str, auto_download: bool) -> str:
        if osp.exists(root):
            return root
        if not auto_download:
            raise FileNotFoundError(
                f"Dataset root {root} not found and auto_download=False."
            )
        logger.message(
            f"Dataset root {root} not found. Downloading {self.name} from {self.url}."
        )
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        logger.message(f"Downloaded and extracted to {downloaded_root}")
        return downloaded_root

    def _prepare_cache(self, overwrite: bool) -> None:
        sample_dir = osp.join(self.cache_path, "samples")
        cache_cfg_path = osp.join(self.cache_path, "dataset_cfg.pkl")
        grid_cache_path = osp.join(self.cache_path, "grid.pkl")
        cache_exists = osp.exists(sample_dir)

        expected_cfg = {
            "mol_name": self.mol_name,
            "split": self.split,
            "n_grid": self.n_grid,
            "grid_size": self.grid_size,
            "data_path": self.data_path,
        }

        if cache_exists and not overwrite:
            try:
                cached_cfg = self._load_from_cache(cache_cfg_path)
                if not is_equal(cached_cfg, expected_cfg):
                    logger.warning(
                        "Cache configuration differs from current settings. "
                        "Will rebuild cache to avoid stale artifacts."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning("Failed to read cached config. Will rebuild cache.")
                overwrite = True

        if overwrite or not cache_exists:
            self._build_cache(sample_dir, cache_cfg_path, grid_cache_path, expected_cfg)

        if dist.is_initialized():
            dist.barrier()

        self.samples = sorted(
            [
                osp.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.endswith(".pkl")
            ]
        )
        if len(self.samples) == 0:
            raise RuntimeError(f"No cached samples found under {sample_dir}")

        grid_np = self._load_from_cache(grid_cache_path)
        self.grid_coord = paddle.to_tensor(
            grid_np, dtype="float32", place=paddle.CPUPlace()
        )

    def _build_cache(
        self,
        sample_dir: str,
        cache_cfg_path: str,
        grid_cache_path: str,
        expected_cfg: Dict[str, Any],
    ) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        os.makedirs(sample_dir, exist_ok=True)
        logger.message(f"Building cache at {self.cache_path}")

        structures_path = osp.join(self.data_path, "structures.npy")
        density_path = osp.join(self.data_path, "dft_densities.npy")
        if not osp.exists(structures_path) or not osp.exists(density_path):
            raise FileNotFoundError(
                f"Cannot locate expected files under {self.data_path}. "
                "Expected structures.npy and dft_densities.npy."
            )

        atom_type = ATOM_TYPES[self.mol_name]
        structures = np.load(structures_path).astype("float32")
        densities_fft = np.load(density_path)
        densities = self._convert_fft(densities_fft)
        num_samples = structures.shape[0]

        grid_coord = self._generate_grid()
        self._save_to_cache(grid_cache_path, grid_coord)
        self._save_to_cache(cache_cfg_path, expected_cfg)

        cell_np = np.eye(3, dtype="float32") * self.grid_size
        for idx in range(num_samples):
            sample = {
                "atom_type": atom_type,
                "atom_coord": structures[idx],
                "density": densities[idx],
                "shape": [self.n_grid, self.n_grid, self.n_grid],
                "cell": cell_np,
                "file_name": f"{self.mol_name}_{self.split}_{idx:06d}",
            }
            self._save_to_cache(osp.join(sample_dir, f"{idx:010d}.pkl"), sample)
        logger.info(f"Cached {num_samples} samples to {sample_dir}")

    def _convert_fft(self, fft_coeff: np.ndarray) -> np.ndarray:
        """Convert FFT coefficients to real-space densities."""
        logger.message(
            f"Precomputing {self.split} density from FFT coefficients with n_grid={self.n_grid} ..."
        )
        coeff = paddle.to_tensor(
            data=fft_coeff, dtype="float32", place=paddle.CPUPlace()
        ).to("complex64")
        d = coeff.reshape([-1, self.n_grid, self.n_grid, self.n_grid])
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

        result = paddle.flip(
            x=d.real().reshape([-1, self.n_grid**3]), axis=[-1]
        ).detach()
        return result.numpy()

    def _generate_grid(self) -> np.ndarray:
        x = np.linspace(
            start=self.grid_size / self.n_grid,
            stop=self.grid_size,
            num=self.n_grid,
            dtype="float32",
        )
        grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid

    def _save_to_cache(self, cache_path: str, data: Any) -> None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_path: str) -> Any:
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx: int) -> Tuple[Data, paddle.Tensor, paddle.Tensor, Dict]:
        sample = self._load_from_cache(self.samples[idx])
        atom_type = paddle.to_tensor(
            sample["atom_type"], dtype="int64", place=paddle.CPUPlace()
        )
        atom_coord = paddle.to_tensor(
            sample["atom_coord"], dtype="float32", place=paddle.CPUPlace()
        )
        density = paddle.to_tensor(sample["density"], dtype="float32", place=paddle.CPUPlace())

        g = Data(x=atom_type, pos=atom_coord)
        info = {
            "cell": self.cell,
            "shape": sample["shape"],
            "file_name": sample["file_name"],
        }

        data_tuple = (g, density, self.grid_coord, info)
        if self.transforms is not None:
            data_tuple = self.transforms(data_tuple)
        return data_tuple

    def __len__(self) -> int:
        return self.num_samples
