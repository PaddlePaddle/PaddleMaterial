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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle
from PIL import Image

from ppmat.datasets.build_matched_name import build_matched_name_samples
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

    If the dataset root is missing, the released SFIN archive can be downloaded
    automatically. The model-facing keys are controlled by ``input_name`` and
    ``target_name``.
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
        data_path: Optional[str] = None,
        target_subdir: Optional[str] = "gt_enhance",
        input_name: str = "noisy",
        target_name: Optional[str] = None,
        noisy_subdir: str = "noisy",
        file_suffix: str = ".png",
        data_count: Optional[int] = None,
        build_samples_cfg: Optional[Dict[str, Any]] = None,
        scale_to_unit: bool = False,
        transforms: Optional[Callable] = None,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()
        if data_path is None:
            raise ValueError("STEMImageDataset requires `data_path`.")

        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        self.dataset_name = osp.basename(osp.normpath(data_path))
        self.url = url if url is not None else self.DATASET_URLS.get(self.dataset_name)
        self.md5 = md5 if md5 is not None else self.DATASET_MD5S.get(self.dataset_name)
        self.input_name = input_name
        self.target_name = target_name if target_name is not None else target_subdir
        self.noisy_subdir = noisy_subdir
        self.target_subdir = target_subdir
        self.file_suffix = file_suffix
        self.data_count = int(data_count) if data_count is not None else None
        self.scale_to_unit = scale_to_unit
        self.transforms = transforms
        self.auto_download = auto_download
        if build_samples_cfg is None:
            build_samples_cfg = {
                "__class_name__": "BuildIndexedNameSamples",
                "__init_params__": {},
            }
            logger.message(
                "The build_samples_cfg is not set, will use the default "
                f"configs: {build_samples_cfg}"
            )
        self.build_samples_cfg = build_samples_cfg
        self.path = self._prepare_root(data_path, auto_download)
        self.data_path = self.path
        self.sample_builder = build_matched_name_samples(self.build_samples_cfg)

        self.root = self.data_path
        self.data_root = self.root
        self.noisy_root = osp.join(self.data_root, self.noisy_subdir)
        self.target_root = (
            osp.join(self.data_root, self.target_subdir)
            if self.target_subdir is not None
            else None
        )
        if not osp.isdir(self.noisy_root):
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_root}")
        if self.target_root is not None and not osp.isdir(self.target_root):
            raise FileNotFoundError(f"Target directory not found: {self.target_root}")
        if self.target_root is None:
            self.samples = self._build_prediction_samples()
        else:
            self.samples = self._build_samples()
        self.file_names = [sample["name"] for sample in self.samples]

    def _prepare_root(self, root: str, auto_download: bool) -> str:
        if self._has_data_dirs(root):
            return root
        if not auto_download or self.url is None:
            raise FileNotFoundError(
                f"Dataset root {root} not found. "
                "Please check the path or enable auto_download with a valid url."
            )
        logger.message("The dataset is not found. Will download it now.")
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        if self._has_data_dirs(downloaded_root):
            return downloaded_root

        nested_root = osp.join(
            downloaded_root, osp.basename(osp.normpath(downloaded_root))
        )
        if self._has_data_dirs(nested_root):
            return nested_root

        return downloaded_root

    def _has_data_dirs(self, root: str) -> bool:
        if not osp.isdir(root):
            return False
        noisy_root = osp.join(root, self.noisy_subdir)
        target_root = (
            osp.join(root, self.target_subdir)
            if self.target_subdir is not None
            else None
        )
        return osp.isdir(noisy_root) and (
            target_root is None or osp.isdir(target_root)
        )

    def _build_samples(self):
        noisy_files = self._list_image_files(self.noisy_root)
        target_files = self._list_image_files(self.target_root)
        class_name = self.build_samples_cfg.get("__class_name__", "")
        if class_name.endswith("BuildMatchedNameSamples"):
            sample_data = self._prepare_matched_sample_data(noisy_files, target_files)
        elif class_name.endswith("BuildIndexedNameSamples"):
            sample_data = self._prepare_indexed_sample_data(noisy_files, target_files)
        else:
            raise ValueError(f"Unsupported sample builder class: {class_name}")
        if self.data_count is not None:
            sample_data = sample_data[: self.data_count]
        samples = self.sample_builder(sample_data)
        if not samples:
            raise FileNotFoundError(
                f"No paired samples found under {self.noisy_root} "
                f"and {self.target_root}."
            )
        return samples

    def _build_prediction_samples(self):
        noisy_files = self._list_image_files(self.noisy_root)
        if self.data_count is not None:
            noisy_files = noisy_files[: self.data_count]
        return [{"noisy": file_name, "name": file_name} for file_name in noisy_files]

    def _list_image_files(self, root: str):
        file_names = sorted(
            [
                file_name
                for file_name in os.listdir(root)
                if file_name.endswith(self.file_suffix)
            ]
        )
        if not file_names:
            raise FileNotFoundError(f"No images found under {root}.")
        return file_names

    def _build_index_map(self, file_names, root: str):
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
                f"'0{self.file_suffix}' under {root}. "
                f"Invalid files: {invalid_files[:10]}."
            )
        if duplicate_files:
            raise ValueError(
                "Strict indexed naming requires one file per integer index under "
                f"{root}. Duplicate indexed files: {duplicate_files[:10]}."
            )
        return index_map

    def _prepare_matched_sample_data(self, noisy_files, target_files):
        noisy_file_set = set(noisy_files)
        target_file_set = set(target_files)
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
                "noisy_file": file_name,
                "target_file": file_name,
            }
            for file_name in sorted(noisy_file_set & target_file_set)
        ]

    def _prepare_indexed_sample_data(self, noisy_files, target_files):
        noisy_map = self._build_index_map(noisy_files, self.noisy_root)
        target_map = self._build_index_map(target_files, self.target_root)
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
                "noisy_file": noisy_map[idx],
                "target_file": target_map[idx],
            }
            for idx in common_indices
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
            "id": idx,
        }
        if self.target_root is not None:
            target = self._load_gray_image(osp.join(self.target_root, sample["target"]))
            data[self.target_name] = target
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.samples)
