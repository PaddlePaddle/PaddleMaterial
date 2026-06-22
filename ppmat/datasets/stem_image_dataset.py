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

from __future__ import absolute_import
from __future__ import annotations

import os.path as osp
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image

from ppmat.datasets.build_matched_name import build_prediction_samples
from ppmat.datasets.build_matched_name import build_matched_name_samples
from ppmat.utils import download
from ppmat.utils import io
from ppmat.utils import logger


class STEMImageDataset(Dataset):
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
        path: Optional[str] = None,
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
        **kwargs,  # for compatibility
    ):
        super().__init__()
        if path is None:
            raise ValueError("STEMImageDataset requires `path`.")

        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        self.dataset_name = osp.basename(osp.normpath(path))
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
            build_samples_cfg = {"match_mode": "indexed"}
            logger.message(
                "The build_samples_cfg is not set, will use the default "
                f"configs: {build_samples_cfg}"
            )
        self.build_samples_cfg = build_samples_cfg
        self.sample_builder = build_matched_name_samples(build_samples_cfg)

        noisy_root = osp.join(path, self.noisy_subdir)
        target_root = (
            osp.join(path, self.target_subdir)
            if self.target_subdir is not None
            else None
        )
        has_data_dirs = osp.isdir(noisy_root) and (
            target_root is None or osp.isdir(target_root)
        )
        if not has_data_dirs:
            if not auto_download or self.url is None:
                raise FileNotFoundError(
                    f"Dataset root {path} not found. "
                    "Please check the path or enable auto_download with a valid url."
            )
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            for candidate in (
                root_path,
                osp.join(root_path, self.dataset_name),
                osp.join(root_path, osp.basename(osp.normpath(root_path))),
            ):
                noisy_root = osp.join(candidate, self.noisy_subdir)
                target_root = (
                    osp.join(candidate, self.target_subdir)
                    if self.target_subdir is not None
                    else None
                )
                if osp.isdir(noisy_root) and (
                    target_root is None or osp.isdir(target_root)
                ):
                    path = candidate
                    break
            else:
                path = root_path

        self.path = path
        self.root = self.path
        self.data_root = self.root
        self.noisy_root = osp.join(self.data_root, self.noisy_subdir)
        self.target_root = (
            osp.join(self.data_root, self.target_subdir)
            if self.target_subdir is not None
            else None
        )

        self.row_data, self.num_samples = self.read_data(self.path)
        self.samples = self.row_data["samples"]
        self.file_names = self.row_data["name"]
        logger.info(f"Load {self.num_samples} samples from {self.path}")

    def read_data(self, path: str) -> Tuple[Dict[str, List[Any]], int]:
        """Read STEM image file names and build sample metadata."""
        if not osp.isdir(self.noisy_root):
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_root}")
        if self.target_root is not None and not osp.isdir(self.target_root):
            raise FileNotFoundError(f"Target directory not found: {self.target_root}")

        noisy_files = io.list_files_by_suffix(self.noisy_root, self.file_suffix)
        if self.target_root is None:
            samples = build_prediction_samples(noisy_files)
        else:
            target_files = io.list_files_by_suffix(
                self.target_root, self.file_suffix
            )
            samples = self.sample_builder(
                noisy_files,
                target_files,
                self.noisy_root,
                self.target_root,
                self.file_suffix,
            )
        if self.data_count is not None:
            samples = samples[: self.data_count]
        if not samples and self.data_count != 0:
            raise FileNotFoundError(
                f"No paired samples found under {self.noisy_root} "
                f"and {self.target_root}."
            )

        row_data = {
            "samples": samples,
            "noisy": [sample["noisy"] for sample in samples],
            "name": [sample["name"] for sample in samples],
        }
        if self.target_root is not None:
            row_data["target"] = [sample["target"] for sample in samples]
        return row_data, len(samples)

    def _load_gray_image(self, file_path: str) -> paddle.Tensor:
        image = Image.open(file_path).convert("L")
        image_array = np.asarray(image, dtype=np.float32)
        if self.scale_to_unit:
            image_array = image_array / 255.0
        return paddle.to_tensor(image_array).unsqueeze(0)

    def __getitem__(self, idx: int):
        noisy = self._load_gray_image(
            osp.join(self.noisy_root, self.row_data["noisy"][idx])
        )

        data = {
            self.input_name: noisy,
            "name": self.row_data["name"][idx],
            "id": idx,
        }
        if self.target_root is not None:
            target = self._load_gray_image(
                osp.join(self.target_root, self.row_data["target"][idx])
            )
            data[self.target_name] = target
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
