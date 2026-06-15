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

    If ``data_path`` is missing, the released SFIN archive is downloaded
    automatically according to the basename of ``data_path``. The model-facing
    keys are controlled by ``input_name`` and ``target_name``.
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
        input_name: str = "noisy",
        target_name: Optional[str] = None,
        noisy_subdir: str = "noisy",
        file_suffix: str = ".png",
        data_count: Optional[int] = None,
        strict_index_naming: bool = True,
        build_samples_cfg: Optional[Dict[str, Any]] = None,
        scale_to_unit: bool = False,
        transforms: Optional[Callable] = None,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
    ):
        super().__init__()

        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        self.data_path = data_path
        self.input_name = input_name
        self.target_name = target_name if target_name is not None else target_subdir
        self.noisy_subdir = noisy_subdir
        self.target_subdir = target_subdir
        self.file_suffix = file_suffix
        self.data_count = int(data_count) if data_count is not None else None
        self.strict_index_naming = strict_index_naming
        self.scale_to_unit = scale_to_unit
        self.transforms = transforms
        self.dataset_name = osp.basename(osp.normpath(data_path))
        self.url = url if url is not None else self.DATASET_URLS.get(self.dataset_name)
        self.md5 = md5 if md5 is not None else self.DATASET_MD5S.get(self.dataset_name)
        self.auto_download = auto_download
        if build_samples_cfg is None:
            class_name = (
                "BuildIndexedNameSamples"
                if strict_index_naming
                else "BuildMatchedNameSamples"
            )
            build_samples_cfg = {
                "__class_name__": class_name,
                "__init_params__": {},
            }
        self.sample_builder = build_matched_name_samples(build_samples_cfg)

        self.root = self._prepare_root(data_path, self.auto_download)
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
        self.samples = self._build_samples()
        self.file_names = [sample["name"] for sample in self.samples]

    def _prepare_root(
        self,
        data_path: str,
        auto_download: bool,
    ) -> str:
        def has_data_dirs(root: str) -> bool:
            noisy_root = osp.join(root, self.noisy_subdir)
            target_root = (
                osp.join(root, self.target_subdir)
                if self.target_subdir is not None
                else None
            )
            return osp.isdir(noisy_root) and (
                target_root is None or osp.isdir(target_root)
            )

        if osp.exists(data_path):
            if has_data_dirs(data_path):
                return data_path
            if (
                self.dataset_name not in self.DATASET_URLS
                or not auto_download
                or self.url is None
            ):
                return data_path
            logger.message(
                f"Dataset root {data_path} exists but does not contain "
                f"'{self.noisy_subdir}' and '{self.target_subdir}'. "
                f"Downloading {self.name} from {self.url}."
            )
        else:
            if not auto_download or self.url is None:
                raise FileNotFoundError(
                    f"Dataset path {data_path} not found. Please prepare data "
                    "manually or enable auto download with a valid url."
                )
            logger.message(
                f"Dataset root {data_path} not found. "
                f"Downloading {self.name} from {self.url}."
            )
        downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        if not osp.exists(downloaded_root):
            parent_root = osp.dirname(downloaded_root)
            if parent_root and osp.isdir(parent_root):
                downloaded_root = parent_root

        if self.dataset_name in self.DATASET_URLS and not has_data_dirs(downloaded_root):
            named_root = osp.join(downloaded_root, self.dataset_name)
            if has_data_dirs(named_root):
                downloaded_root = named_root

        logger.info(f"Dataset downloaded to: {downloaded_root}")
        return downloaded_root

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

        samples = self.sample_builder(
            noisy_files,
            target_files,
            noisy_root=self.noisy_root,
            target_root=self.target_root,
            file_suffix=self.file_suffix,
            data_count=self.data_count,
        )
        if not samples:
            raise FileNotFoundError(
                f"No paired samples found under {self.noisy_root} "
                f"and {self.target_root}."
            )
        return samples

    def _build_prediction_samples(self, noisy_files):
        samples = [
            {
                "noisy": file_name,
                "target": None,
                "name": file_name,
            }
            for file_name in noisy_files
        ]
        if self.data_count is not None:
            samples = samples[: self.data_count]
        return samples

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
