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
import paddle.distributed as dist
from paddle.io import Dataset
from PIL import Image

from ppmat.datasets.build_matched_name import build_prediction_samples
from ppmat.datasets.build_matched_name import build_matched_name_samples
from ppmat.utils import download
from ppmat.utils import io
from ppmat.utils import logger
from ppmat.utils.misc import is_equal


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
        "sfin_haadf": "f96dea9ac1f722d6ca55c7e49c1b3a41",
        "sfin_bf": "74ec1c1959e162669cc8cbbc8713bda0",
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
            self._normalize_extracted_paths(root_path)
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

    @staticmethod
    def _normalize_extracted_paths(root_path: str):
        """Normalize Windows-style zip entries after automatic download.

        Some zip files generated on Windows may contain ``\\`` in entry names.
        On Linux, ``zipfile.extract`` treats those backslashes as ordinary
        filename characters instead of directory separators. Move such files to
        their normalized nested paths so the canonical SFIN layout can be read.
        """
        if not osp.isdir(root_path):
            return

        file_move_pairs = []
        dir_move_pairs = []
        for current_root, dir_names, file_names in os.walk(root_path, topdown=False):
            for file_name in file_names:
                normalized_name = file_name.replace("\\", os.sep)
                if normalized_name == file_name:
                    continue
                src_path = osp.join(current_root, file_name)
                dst_path = osp.normpath(osp.join(current_root, normalized_name))
                if osp.abspath(src_path) == osp.abspath(dst_path):
                    continue
                file_move_pairs.append((src_path, dst_path))

            for dir_name in dir_names:
                normalized_name = dir_name.replace("\\", os.sep)
                if normalized_name == dir_name:
                    continue
                src_path = osp.join(current_root, dir_name)
                dst_path = osp.normpath(osp.join(current_root, normalized_name))
                if osp.abspath(src_path) == osp.abspath(dst_path):
                    continue
                dir_move_pairs.append((src_path, dst_path))

        for src_path, dst_path in file_move_pairs + dir_move_pairs:
            os.makedirs(osp.dirname(dst_path), exist_ok=True)
            if osp.exists(dst_path):
                continue
            os.replace(src_path, dst_path)

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
        sample_done_flag = osp.join(sample_cache_path, "completed.flag")
        build_samples_cfg_path = osp.join(self.cache_path, "build_samples_cfg.pkl")
        image_cfg_path = osp.join(self.cache_path, "image_cfg.pkl")
        image_cfg = {
            "input_name": self.input_name,
            "target_name": self.target_name,
            "target_subdir": self.target_subdir,
            "label_subdirs": self.label_subdirs,
            "noisy_subdir": self.noisy_subdir,
            "file_suffix": self.file_suffix,
            "scale_to_unit": self.scale_to_unit,
        }
        logger.info(f"Cache path: {self.cache_path}")

        cache_exists = osp.exists(self.cache_path)
        overwrite = self.overwrite
        if cache_exists and not self.overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_samples_cfg_cache = self.load_from_cache(build_samples_cfg_path)
                if is_equal(build_samples_cfg_cache, self.build_samples_cfg):
                    logger.info(
                        "The cached build_samples_cfg configuration matches "
                        "the current settings. Reusing cached STEM image samples."
                    )
                else:
                    logger.warning(
                        "build_samples_cfg is different from "
                        "build_samples_cfg_cache. Will rebuild the samples."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load build_samples_cfg.pkl from cache. "
                    "Will rebuild the samples."
                )
                overwrite = True

            if not overwrite:
                try:
                    image_cfg_cache = self.load_from_cache(image_cfg_path)
                    if is_equal(image_cfg_cache, image_cfg):
                        logger.info(
                            "The cached image_cfg configuration matches "
                            "the current settings."
                        )
                    else:
                        logger.warning(
                            "image_cfg is different from image_cfg_cache. "
                            "Will rebuild the samples."
                        )
                        overwrite = True
                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load image_cfg.pkl from cache. "
                        "Will rebuild the samples."
                    )
                    overwrite = True

            if not overwrite:
                num_cached = self._count_cache_files(sample_cache_path)
                is_complete = osp.exists(sample_done_flag)
                if is_complete and num_cached == self.num_samples:
                    logger.info(
                        f"Using cached STEM image samples ({num_cached}) "
                        f"from {sample_cache_path}."
                    )
                else:
                    logger.warning(
                        f"Cached STEM image samples are incomplete "
                        f"(cached={num_cached}, expected={self.num_samples}, "
                        f"complete={is_complete}). Will rebuild the samples."
                    )
                    overwrite = True

        if overwrite or not cache_exists:
            self._build_cache(
                sample_cache_path,
                sample_done_flag,
                build_samples_cfg_path,
                image_cfg_path,
                image_cfg,
            )

        if dist.is_initialized():
            dist.barrier()

        self.cache_files = [
            osp.join(sample_cache_path, f"{idx:010d}.pkl")
            for idx in range(self.num_samples)
        ]
        if not all(osp.exists(cache_file) for cache_file in self.cache_files):
            raise RuntimeError(
                f"No complete cached STEM image samples found under "
                f"{sample_cache_path}."
            )

    def _build_cache(
        self,
        sample_cache_path: str,
        sample_done_flag: str,
        build_samples_cfg_path: str,
        image_cfg_path: str,
        image_cfg: Dict[str, Any],
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(sample_cache_path, exist_ok=True)
        self._clean_cache_dir(sample_cache_path)

        self.save_to_cache(build_samples_cfg_path, self.build_samples_cfg)
        self.save_to_cache(image_cfg_path, image_cfg)
        logger.message(
            f"Caching {self.num_samples} STEM image samples to {sample_cache_path}"
        )
        for idx in range(self.num_samples):
            data = self._build_item(idx)
            payload = self._serialize_item(data)
            self.save_to_cache(osp.join(sample_cache_path, f"{idx:010d}.pkl"), payload)
        with open(sample_done_flag, "w") as f:
            f.write("done")
        logger.info(f"Finished caching STEM image samples to {sample_cache_path}")

    @staticmethod
    def _count_cache_files(cache_path: str) -> int:
        if not osp.isdir(cache_path):
            return 0
        return sum(
            1
            for file_name in os.listdir(cache_path)
            if file_name.endswith(".pkl")
        )

    @staticmethod
    def _clean_cache_dir(cache_path: str):
        if not osp.isdir(cache_path):
            return
        for file_name in os.listdir(cache_path):
            if file_name.endswith(".pkl") or file_name.endswith(".flag"):
                os.remove(osp.join(cache_path, file_name))

    def _serialize_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {}
        for key, value in data.items():
            if isinstance(value, paddle.Tensor):
                payload[key] = value.detach().cpu().numpy()
            else:
                payload[key] = value
        return payload

    def _deserialize_item(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                data[key] = paddle.to_tensor(value)
            else:
                data[key] = value
        return data

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

    def save_to_cache(self, cache_path: str, data: Any):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, idx: int):
        if self.cache and self.cache_files and idx < len(self.cache_files):
            try:
                payload = self.load_from_cache(self.cache_files[idx])
                data = self._deserialize_item(payload)
            except Exception as e:
                logger.warning(f"Failed to load cached STEM image sample {idx}: {e}")
                data = self._build_item(idx)
        else:
            data = self._build_item(idx)
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
