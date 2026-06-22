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

import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
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


class SFINDataset(Dataset):
    """Paired SFIN image dataset for spectrum enhancement tasks.

    Expected layout:

    ```text
    root/
      train/
        noisy/
        gt_enhance/
        gt_detect/
      test/
        noisy/
        gt_enhance/
        gt_detect/
    ```

    Each noisy image is paired with both ``gt_enhance`` and ``gt_detect`` labels
    in the canonical SFIN layout. The model-facing input key is controlled by
    ``input_name``; ``target_name`` selects the label consumed by the current
    task while the other label remains available in the sample dictionary.
    """

    name = "sfin"
    url = None
    md5 = None

    DATASET_URLS: Dict[str, str] = {
        "sfin_haadf": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN/sfin_haadf.zip"
        ),
        "sfin_bf": (
            "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
            "SFIN/sfin_bf.zip"
        ),
    }
    DATASET_MD5S: Dict[str, Optional[str]] = {
        "sfin_haadf": None,
        "sfin_bf": None,
    }

    def __init__(
        self,
        path: Optional[str] = None,
        split: str = "train",
        target_subdir: Optional[str] = "gt_enhance",
        label_subdirs: Optional[Sequence[str]] = None,
        input_name: str = "noisy",
        target_name: Optional[str] = None,
        noisy_subdir: str = "noisy",
        file_suffix: str = ".png",
        data_count: Optional[int] = None,
        build_samples_cfg: Optional[Dict[str, Any]] = None,
        scale_to_unit: bool = False,
        transforms: Optional[Callable] = None,
        cache: bool = False,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        auto_download: bool = True,
        **kwargs,  # for compatibility
    ):
        super().__init__()
        if path is None:
            raise ValueError("SFINDataset requires `path`.")

        if data_count is not None and int(data_count) < 0:
            raise ValueError("data_count must be None or a non-negative integer.")

        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split '{split}', expected 'train' or 'test'.")

        self.dataset_name = osp.basename(osp.normpath(path))
        self.split = split
        self.url = url if url is not None else self.DATASET_URLS.get(self.dataset_name)
        self.md5 = md5 if md5 is not None else self.DATASET_MD5S.get(self.dataset_name)
        self.input_name = input_name
        self.target_name = target_name if target_name is not None else target_subdir
        self.noisy_subdir = noisy_subdir
        self.target_subdir = target_subdir
        if self.target_subdir is None:
            self.label_subdirs = tuple()
        else:
            label_subdirs = label_subdirs or ("gt_enhance", "gt_detect")
            self.label_subdirs = tuple(dict.fromkeys(label_subdirs))
            if self.target_subdir not in self.label_subdirs:
                self.label_subdirs = self.label_subdirs + (self.target_subdir,)
        self.file_suffix = file_suffix
        self.data_count = int(data_count) if data_count is not None else None
        self.scale_to_unit = scale_to_unit
        self.transforms = transforms
        self.cache = cache
        self.cache_path = cache_path
        self.overwrite = overwrite
        self.auto_download = auto_download
        if build_samples_cfg is None:
            build_samples_cfg = {"match_mode": "indexed"}
            logger.message(
                "The build_samples_cfg is not set, will use the default "
                f"configs: {build_samples_cfg}"
            )
        self.build_samples_cfg = build_samples_cfg
        self.sample_builder = build_matched_name_samples(build_samples_cfg)

        data_root = osp.join(path, self.split)
        noisy_root = osp.join(data_root, self.noisy_subdir)
        target_roots = {
            label_name: osp.join(data_root, label_name)
            for label_name in self.label_subdirs
        }
        has_data_dirs = osp.isdir(noisy_root) and all(
            osp.isdir(target_root) for target_root in target_roots.values()
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
                osp.join(root_path, self.split),
                osp.join(root_path, self.dataset_name),
                osp.join(root_path, self.dataset_name, self.split),
                root_path,
                osp.join(root_path, osp.basename(osp.normpath(root_path))),
            ):
                noisy_root = osp.join(candidate, self.noisy_subdir)
                target_roots = {
                    label_name: osp.join(candidate, label_name)
                    for label_name in self.label_subdirs
                }
                if osp.isdir(noisy_root) and all(
                    osp.isdir(target_root) for target_root in target_roots.values()
                ):
                    data_root = candidate
                    break
            else:
                data_root = root_path
        else:
            data_root = osp.join(path, self.split)

        self.path = path
        self.root = self.path
        self.data_root = data_root
        self.noisy_root = osp.join(self.data_root, self.noisy_subdir)
        self.target_roots = {
            label_name: osp.join(self.data_root, label_name)
            for label_name in self.label_subdirs
        }
        self.target_root = (
            self.target_roots.get(self.target_subdir)
            if self.target_subdir is not None
            else None
        )

        self.row_data, self.num_samples = self.read_data(self.path)
        self.samples = self.row_data["samples"]
        self.file_names = self.row_data["name"]
        self._prepare_cache()
        logger.info(f"Load {self.num_samples} samples from {self.path}")

    def read_data(self, path: str) -> Tuple[Dict[str, List[Any]], int]:
        """Read STEM image file names and build sample metadata."""
        if not osp.isdir(self.noisy_root):
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_root}")
        for label_name, target_root in self.target_roots.items():
            if not osp.isdir(target_root):
                raise FileNotFoundError(f"Target directory not found: {target_root}")

        noisy_files = io.list_files_by_suffix(self.noisy_root, self.file_suffix)
        if self.target_root is None:
            samples = build_prediction_samples(noisy_files)
        else:
            target_files = io.list_files_by_suffix(self.target_root, self.file_suffix)
            samples = self.sample_builder(
                noisy_files,
                target_files,
                self.noisy_root,
                self.target_root,
                self.file_suffix,
            )
            for label_name, target_root in self.target_roots.items():
                if label_name == self.target_subdir:
                    for sample in samples:
                        sample[label_name] = sample["target"]
                    continue
                label_files = io.list_files_by_suffix(target_root, self.file_suffix)
                label_samples = self.sample_builder(
                    noisy_files,
                    label_files,
                    self.noisy_root,
                    target_root,
                    self.file_suffix,
                )
                label_file_by_name = {
                    sample["name"]: sample["target"] for sample in label_samples
                }
                for sample in samples:
                    sample[label_name] = label_file_by_name[sample["name"]]
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
            for label_name in self.label_subdirs:
                row_data[label_name] = [sample[label_name] for sample in samples]
        return row_data, len(samples)

    def _load_gray_image(self, file_path: str) -> paddle.Tensor:
        image = Image.open(file_path).convert("L")
        image_array = np.asarray(image, dtype=np.float32)
        if self.scale_to_unit:
            image_array = image_array / 255.0
        return paddle.to_tensor(image_array).unsqueeze(0)

    def _prepare_cache(self):
        self.cache_files = []
        if not self.cache:
            return

        target_name = self.target_name if self.target_name is not None else "predict"
        if self.cache_path is None:
            self.cache_path = osp.join(
                f"{self.path}_cache",
                self.split,
                str(target_name),
            )
        sample_cache_path = osp.join(self.cache_path, "samples")
        os.makedirs(sample_cache_path, exist_ok=True)

        self.cache_files = [
            osp.join(sample_cache_path, f"{idx:010d}.pkl")
            for idx in range(self.num_samples)
        ]
        cache_ready = all(osp.exists(cache_file) for cache_file in self.cache_files)
        if cache_ready and not self.overwrite:
            logger.info(f"Using cached STEM image samples from {sample_cache_path}")
            return

        logger.info(
            f"Caching {self.num_samples} STEM image samples to {sample_cache_path}"
        )
        for idx, cache_file in enumerate(self.cache_files):
            data = self._build_item(idx)
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)

    def _build_item(self, idx: int):
        noisy = self._load_gray_image(
            osp.join(self.noisy_root, self.row_data["noisy"][idx])
        )

        data = {
            self.input_name: noisy,
            "name": self.row_data["name"][idx],
            "id": idx,
        }
        for label_name, target_root in self.target_roots.items():
            data[label_name] = self._load_gray_image(
                osp.join(target_root, self.row_data[label_name][idx])
            )
        if (
            self.target_subdir is not None
            and self.target_name not in data
            and self.target_subdir in data
        ):
            data[self.target_name] = data[self.target_subdir]
        return data

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx: int):
        if self.cache and self.cache_files and idx < len(self.cache_files):
            data = self.load_from_cache(self.cache_files[idx])
        else:
            data = self._build_item(idx)
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
