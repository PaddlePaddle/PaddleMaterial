"""
非对称单元（ASU） MP-20 数据集。

"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import paddle
import pandas as pd
from paddle.io import Dataset

from ppmat.models.sgequidiff.crystal_classes import ASUCrystal, ImmutableASUCrystal


def _get_data_directory() -> Path:
    """
    返回数据根目录：优先检查环境变量，否则 fallback 到项目 data/ 目录。
    """
    import os
    env = os.environ.get("SGEQUIDIFF_DATA_DIR", None)
    if env:
        return Path(env)
    # 尝试找项目 data 目录
    candidates = [
        Path(__file__).resolve().parents[3] / "data",
        Path("~").expanduser() / ".sgequidiff_data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "找不到数据目录。请设置环境变量 SGEQUIDIFF_DATA_DIR 或确保 data/ 目录存在。"
    )

class AsymmetricUnitDataset(Dataset):
    """
    ASU 表示下的 Materials Project 数据集（MP-20 / MPTS-52）。

    数据集在预处理脚本中将 CIF 格式晶体转换为 ASUCrystal 对象，
    并以压缩 .npz 格式保存。

    Args:
        name: 数据集名称，["mp_20", "mp_20_assumeP1", "mpts_52"]
        split: 数据集分割，["train", "val", "test"]
        data_directory: 自定义数据目录（None 则自动检测）
    """

    def __init__(
        self,
        name: str = "mp_20",
        split: str = "train",
        data_directory: Optional[Path] = None,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"unknown split: {split}"
        assert name in ["mp_20", "mp_20_assumeP1", "mpts_52"], f"unknown name: {name}"

        if name in ("mp_20", "mp_20_assumeP1"):
            self.max_atoms = 20
            self.max_elements = 7
        elif name == "mpts_52":
            self.max_atoms = 52
            self.max_elements = 7

        self.name = name
        self.split = split

        if data_directory is None:
            data_directory = _get_data_directory()

        data_path = Path(data_directory) / name / f"{split}.npz"
        properties_path = Path(data_directory) / name / f"{split}_properties.pkl"

        npz: dict = np.load(data_path)
        properties_df = pd.read_pickle(properties_path)  # 预留，将来可以使用

        self.indices_arr: np.ndarray = npz["indices"]
        self.packed: np.ndarray = npz["packed"]
        flat_crystals: List[np.ndarray] = np.split(self.packed, self.indices_arr)
        num_crystals = len(flat_crystals)

        self.data: List[ImmutableASUCrystal] = []
        _space_group_indices = []
        _composition_spaces = []
        _lattice_lengths = []
        _lattice_angles = []
        _n_atoms_per_asu = []

        self.padded_element_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_wyckoff_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_wyckoff_shape_indices = -1 * paddle.ones(
            [num_crystals, self.max_atoms], dtype=paddle.int64
        )
        self.padded_frac_coords = -1.0 * paddle.ones(
            [num_crystals, self.max_atoms, 3], dtype=paddle.float32
        )
        self.atoms_mask = paddle.zeros([num_crystals, self.max_atoms], dtype=paddle.bool)

        for i, flat in enumerate(flat_crystals):
            crystal: ASUCrystal = ASUCrystal.from_flat(flat)
            num_atoms: int = crystal.num_atoms
            self.data.append(crystal.to_ImmutableASUCrystal())

            _space_group_indices.append(crystal.space_group_number - 1)
            _composition_spaces.append(crystal.composition_space)
            _lattice_lengths.append(crystal.conventional_lattice_lengths)
            _lattice_angles.append(crystal.conventional_lattice_angles)
            _n_atoms_per_asu.append(num_atoms)

            # 按 wyckoff_index, element_index 字典序排序
            sorting_indices = paddle.to_tensor(
                sorted(
                    range(num_atoms),
                    key=lambda j: (
                        int(crystal.wyckoff_indices[j].item()),
                        int(crystal.element_indices[j].item()),
                    ),
                ),
                dtype=paddle.int64,
            )

            self.padded_element_indices[i, :num_atoms] = crystal.element_indices[sorting_indices]
            self.padded_wyckoff_indices[i, :num_atoms] = crystal.wyckoff_indices[sorting_indices]
            self.padded_wyckoff_shape_indices[i, :num_atoms] = crystal.wyckoff_shape_indices[sorting_indices]
            self.padded_frac_coords[i, :num_atoms] = crystal.conventional_frac_coords[sorting_indices]
            self.atoms_mask[i, :num_atoms] = True

        self.space_group_indices = paddle.to_tensor(_space_group_indices, dtype=paddle.int64)
        self.n_atoms_per_asu = paddle.to_tensor(_n_atoms_per_asu, dtype=paddle.int64)
        self.composition_spaces = paddle.stack(
            [paddle.to_tensor(c, dtype=paddle.float32) for c in _composition_spaces], axis=0
        )
        self.lattice_lengths = paddle.stack(
            [paddle.to_tensor(l, dtype=paddle.float32) for l in _lattice_lengths], axis=0
        )
        self.lattice_angles = paddle.stack(
            [paddle.to_tensor(a, dtype=paddle.float32) for a in _lattice_angles], axis=0
        )

    def __len__(self) -> int:
        return int(self.space_group_indices.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        """
        返回单个样本的标准 dict，兼容框架 DefaultCollator。

        DefaultCollator 对 dict 递归处理，对 paddle.Tensor 执行 paddle.stack，
        将 N 个 (max_atoms,) 堆叠为 (N, max_atoms)，即 training_wrapper 期望的格式。
        """
        if not isinstance(index, int):
            raise TypeError(
                f"Expected int index, got {type(index)}. "
                "DataLoader with BatchSampler calls __getitem__ with int."
            )

        return self._get_single_item(index)

    @paddle.no_grad()
    def _get_single_item(self, index: int) -> dict:
        """返回单个样本的 dict，所有 tensor 不带 batch 维度。"""
        return {
            "space_group_indices": self.space_group_indices[index],          # scalar
            "batch_chemistries": self.composition_spaces[index],             # (chem_dim,)
            "lattice_lengths": self.lattice_lengths[index],                  # (3,)
            "lattice_angles": self.lattice_angles[index],                    # (3,)
            "n_atoms_per_asu": self.n_atoms_per_asu[index],                  # scalar
            "element_indices": self.padded_element_indices[index],            # (max_atoms,)
            "wyckoff_indices": self.padded_wyckoff_indices[index],            # (max_atoms,)
            "wyckoff_shape_indices": self.padded_wyckoff_shape_indices[index],# (max_atoms,)
            "frac_coords": self.padded_frac_coords[index],                   # (max_atoms, 3)
            "atoms_mask": self.atoms_mask[index],                             # (max_atoms,)
        }

    @paddle.no_grad()
    def reindex(self, idxs: Optional[Union[List[int], paddle.Tensor]] = None):
        """
        重新排列数据集（随机打乱或按指定索引切片）。
        """
        num_crystals = len(self)
        if idxs is None:
            idxs = paddle.randperm(num_crystals)
        elif isinstance(idxs, list):
            idxs = paddle.to_tensor(idxs, dtype=paddle.int64)

        num_new = int(idxs.shape[0])
        self.data = [self.data[int(i)] for i in idxs.tolist()]
        self.space_group_indices = self.space_group_indices[idxs].reshape([-1])
        self.composition_spaces = self.composition_spaces[idxs].reshape([num_new, -1])
        self.lattice_lengths = self.lattice_lengths[idxs].reshape([num_new, -1])
        self.lattice_angles = self.lattice_angles[idxs].reshape([num_new, -1])
        self.n_atoms_per_asu = self.n_atoms_per_asu[idxs].reshape([num_new])
        self.padded_element_indices = self.padded_element_indices[idxs].reshape([num_new, self.max_atoms])
        self.padded_wyckoff_indices = self.padded_wyckoff_indices[idxs].reshape([num_new, self.max_atoms])
        self.padded_wyckoff_shape_indices = self.padded_wyckoff_shape_indices[idxs].reshape([num_new, self.max_atoms])
        self.padded_frac_coords = self.padded_frac_coords[idxs].reshape([num_new, self.max_atoms, 3])
        self.atoms_mask = self.atoms_mask[idxs].reshape([num_new, self.max_atoms])

    @property
    def empirical_space_group_probs(self) -> paddle.Tensor:
        counts = paddle.zeros([230], dtype=paddle.int64)
        for sg_idx in self.space_group_indices.tolist():
            counts[int(sg_idx)] += 1
        return counts.cast(paddle.float32) / len(self)
