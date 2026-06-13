# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle
from PIL import Image

from ppmat.utils import download
from ppmat.utils import logger


class STEMImageDataset(paddle.io.Dataset):
    """Paired STEM image dataset for spectrum enhancement tasks.

    Expected layout:

    ```text
    root/
      noisy/
      gt_enhance/    # optional for prediction-only datasets
      gt_detect/     # optional for prediction-only datasets
    ```

    The handler also accepts ``root/<split>/...`` when a split directory is
    prepared explicitly. If ``data_path`` is missing, the released SFIN archive
    is downloaded automatically according to the basename of ``data_path``.
    The model-facing keys are controlled by ``input_name`` and ``target_name``.
    """

    name = "stem_image"
    url = None
    md5 = None

    DATASET_URLS: Dict[str, str] = {
        "data": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN_datasets/haadf_data.zip"
        ),
        "data_test": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN_datasets/haadf_data_test.zip"
        ),
        "bf_data": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN_datasets/bf_data.zip"
        ),
        "bf_data_test": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN_datasets/bf_data_test.zip"
        ),
    }
    DATASET_MD5S: Dict[str, Optional[str]] = {
        "data": None,
        "data_test": None,
        "bf_data": None,
        "bf_data_test": None,
    }

    def __init__(
        self,
        data_path: str,
        target_subdir: Optional[str] = "gt_enhance",
        split: Optional[str] = None,
        input_name: str = "noisy",
        target_name: Optional[str] = None,
        noisy_subdir: str = "noisy",
        file_suffix: str = ".png",
        data_count: Optional[int] = None,
        strict_index_naming: bool = True,
        scale_to_unit: bool = False,
        transforms: Optional[Callable] = None,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()

        if split == "validation":
            split = "val"
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of None, 'train', 'val', or 'test'.")
        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        self.data_path = data_path
        self.split = split
        self.input_name = input_name
        self.target_name = target_name if target_name is not None else target_subdir
        self.noisy_subdir = noisy_subdir
        self.target_subdir = target_subdir
        self.file_suffix = file_suffix
        self.data_count = data_count
        self.strict_index_naming = strict_index_naming
        self.scale_to_unit = scale_to_unit
        self.transforms = transforms
        dataset_name = self._infer_dataset_name(data_path)
        self.url = url if url is not None else self._get_dataset_url(dataset_name)
        self.md5 = md5 if md5 is not None else self._get_dataset_md5(dataset_name)
        self.auto_download = auto_download

        self.root = self._prepare_root(data_path, self.auto_download)
        self.data_root, self.noisy_root, self.target_root = self._prepare_data_dirs()
        self.samples = self._build_samples()
        self.file_names = [sample["name"] for sample in self.samples]

    @classmethod
    def _infer_dataset_name(cls, data_path: str) -> str:
        return osp.basename(osp.normpath(data_path))

    @classmethod
    def _get_dataset_url(cls, dataset_name: str) -> Optional[str]:
        return cls.DATASET_URLS.get(dataset_name, cls.url)

    @classmethod
    def _get_dataset_md5(cls, dataset_name: str) -> Optional[str]:
        return cls.DATASET_MD5S.get(dataset_name, cls.md5)

    def _prepare_root(
        self,
        data_path: str,
        auto_download: bool,
    ) -> str:
        if osp.exists(data_path):
            return data_path
        if not auto_download or self.url is None:
            raise FileNotFoundError(
                f"Dataset path {data_path} not found. Please prepare data manually "
                "or enable auto download with a valid url."
            )
        logger.message(
            f"Dataset root {data_path} not found. "
            f"Downloading {self.name} from {self.url}."
        )
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        logger.info(f"Dataset downloaded to: {downloaded_root}")
        return downloaded_root

    def _prepare_data_dirs(self):
        data_root = self._resolve_data_root(self.root)
        noisy_root = osp.join(data_root, self.noisy_subdir)
        target_root = (
            osp.join(data_root, self.target_subdir)
            if self.target_subdir is not None
            else None
        )

        if not osp.isdir(noisy_root):
            raise FileNotFoundError(f"Noisy directory not found: {noisy_root}")
        if target_root is not None and not osp.isdir(target_root):
            raise FileNotFoundError(f"Target directory not found: {target_root}")
        return data_root, noisy_root, target_root

    def _contains_data_dirs(self, root: str) -> bool:
        noisy_root = osp.join(root, self.noisy_subdir)
        target_root = (
            osp.join(root, self.target_subdir)
            if self.target_subdir is not None
            else None
        )
        return osp.isdir(noisy_root) and (
            target_root is None or osp.isdir(target_root)
        )

    def _walk_candidate_roots(self, root: str, max_depth: int = 2):
        candidates = [root]
        frontier = [(root, 0)]
        while frontier:
            current_root, depth = frontier.pop(0)
            if depth >= max_depth or not osp.isdir(current_root):
                continue
            for child_name in sorted(os.listdir(current_root)):
                child_root = osp.join(current_root, child_name)
                if not osp.isdir(child_root):
                    continue
                candidates.append(child_root)
                frontier.append((child_root, depth + 1))
        return candidates

    def _resolve_data_root(self, root: str) -> str:
        candidates = self._walk_candidate_roots(root)

        if self.split is not None:
            for candidate in candidates:
                split_root = osp.join(candidate, self.split)
                if self._contains_data_dirs(split_root):
                    return split_root
            searched_roots = ", ".join(candidates)
            raise FileNotFoundError(
                f"Split '{self.split}' with data directories not found under "
                f"{root}. Searched roots: {searched_roots}."
            )

        for candidate in candidates:
            if not self._contains_data_dirs(candidate):
                continue
            return candidate

        searched_roots = ", ".join(candidates)
        raise FileNotFoundError(
            "Cannot locate dataset directories "
            f"'{self.noisy_subdir}' and '{self.target_subdir}' under {root}. "
            f"Searched roots: {searched_roots}."
        )

    @staticmethod
    def _build_index_map(file_names, directory: str, file_suffix: str):
        index_map = {}
        invalid_files = []
        duplicate_files = []
        for file_name in file_names:
            stem = osp.splitext(file_name)[0]
            if not stem.isdigit():
                invalid_files.append(file_name)
                continue
            index = int(stem)
            if index in index_map:
                duplicate_files.append((index_map[index], file_name))
                continue
            index_map[index] = file_name

        if invalid_files:
            raise ValueError(
                "Strict indexed naming requires files named like "
                f"'0{file_suffix}' under {directory}. "
                f"Invalid files: {invalid_files[:10]}."
            )
        if duplicate_files:
            raise ValueError(
                "Strict indexed naming requires one file per integer index under "
                f"{directory}. Duplicate indexed files: {duplicate_files[:10]}."
            )
        return index_map

    def _slice_by_data_count(self, samples):
        if self.data_count is None:
            return samples
        return samples[: self.data_count]

    def _build_samples(self):
        noisy_files = sorted(
            [
                file_name
                for file_name in os.listdir(self.noisy_root)
                if file_name.endswith(self.file_suffix)
            ]
        )
        if not noisy_files:
            raise FileNotFoundError(f"No noisy images found under {self.noisy_root}.")

        if self.target_root is None:
            return self._build_prediction_samples(noisy_files)

        target_files = sorted(
            [
                file_name
                for file_name in os.listdir(self.target_root)
                if file_name.endswith(self.file_suffix)
            ]
        )
        if not target_files:
            raise FileNotFoundError(f"No target images found under {self.target_root}.")

        if self.strict_index_naming:
            samples = self._build_indexed_pair_samples(noisy_files, target_files)
        else:
            samples = self._build_same_name_pair_samples(noisy_files, target_files)
        if not samples:
            raise FileNotFoundError(
                f"No paired samples found under {self.noisy_root} "
                f"and {self.target_root}."
            )
        return self._slice_by_data_count(samples)

    def _build_prediction_samples(self, noisy_files):
        samples = [
            {
                "noisy": file_name,
                "target": None,
                "name": file_name,
            }
            for file_name in noisy_files
        ]
        return self._slice_by_data_count(samples)

    def _build_indexed_pair_samples(self, noisy_files, target_files):
        noisy_map = self._build_index_map(
            noisy_files, self.noisy_root, self.file_suffix
        )
        target_map = self._build_index_map(
            target_files, self.target_root, self.file_suffix
        )
        if not noisy_map:
            raise FileNotFoundError(
                f"No indexed noisy images found under {self.noisy_root}."
            )
        if not target_map:
            raise FileNotFoundError(
                f"No indexed target images found under {self.target_root}."
            )

        common_indices = sorted(set(noisy_map.keys()) & set(target_map.keys()))
        missing_target = sorted(set(noisy_map.keys()) - set(target_map.keys()))
        missing_noisy = sorted(set(target_map.keys()) - set(noisy_map.keys()))
        if missing_target or missing_noisy:
            raise FileNotFoundError(
                "Noisy and target images are not paired. "
                f"Missing target indices: {missing_target[:10]}, "
                f"missing noisy indices: {missing_noisy[:10]}."
            )
        return [
            {
                "noisy": noisy_map[idx],
                "target": target_map[idx],
                "name": noisy_map[idx],
            }
            for idx in common_indices
        ]

    def _build_same_name_pair_samples(self, noisy_files, target_files):
        target_file_set = set(target_files)
        noisy_file_set = set(noisy_files)
        missing_target = sorted(noisy_file_set - target_file_set)
        missing_noisy = sorted(target_file_set - noisy_file_set)
        if missing_target or missing_noisy:
            raise FileNotFoundError(
                "Noisy and target images are not paired. "
                f"Missing target files: {missing_target[:10]}, "
                f"missing noisy files: {missing_noisy[:10]}."
            )
        return [
            {
                "noisy": file_name,
                "target": file_name,
                "name": file_name,
            }
            for file_name in noisy_files
            if file_name in target_file_set
        ]

    def _load_gray_image(self, file_path: str) -> paddle.Tensor:
        image = Image.open(file_path).convert("L")
        image_array = np.asarray(image, dtype=np.float32)
        if self.scale_to_unit:
            image_array = image_array / 255.0
        return paddle.to_tensor(image_array).unsqueeze(0)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        noisy = self._load_gray_image(osp.join(self.noisy_root, sample["noisy"]))

        data = {
            self.input_name: noisy,
            "name": sample["name"],
        }
        if self.target_root is not None:
            target = self._load_gray_image(osp.join(self.target_root, sample["target"]))
            data[self.target_name] = target
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.samples)
