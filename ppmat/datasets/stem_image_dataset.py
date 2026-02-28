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

import copy
import importlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
from PIL import Image

from ppmat.utils import download as download_utils
from ppmat.utils import logger


def _locate_class(class_name: str):
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    return globals()[class_name]


def _build_component(
    cfg: Optional[Dict],
    *,
    default_class_name: str,
    required_methods: List[str],
):
    if cfg is None:
        class_name = default_class_name
        init_params = {}
    else:
        cfg = copy.deepcopy(cfg)
        class_name = cfg.pop("__class_name__", None)
        if not class_name:
            raise ValueError(
                "Factory cfg must include '__class_name__', e.g. "
                "{'__class_name__': 'StrictIndexSampleBuilder', '__init_params__': {}}"
            )
        init_params = cfg.pop("__init_params__", {})
        if cfg:
            raise ValueError(
                f"Unsupported keys in factory cfg for '{class_name}': {list(cfg.keys())}"
            )

    cls = _locate_class(class_name)
    component = cls(**init_params)
    for method_name in required_methods:
        if not hasattr(component, method_name):
            raise TypeError(
                f"Component '{class_name}' must implement method '{method_name}'."
            )
    return component


class StrictIndexSampleBuilder:
    def __init__(self):
        pass

    def build(
        self,
        noisy_dir: Path,
        target_dir: Path,
        file_suffix: str,
        data_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        if data_count is None:
            available = sorted(noisy_dir.glob(f"*{file_suffix}"))
            data_count = len(available)

        for idx in range(int(data_count)):
            name = f"{idx}{file_suffix}"
            noisy_path = noisy_dir / name
            target_path = target_dir / name
            if not noisy_path.exists():
                raise FileNotFoundError(f"Noisy image not found: {noisy_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
            samples.append(
                {
                    "name": name,
                    "noisy_path": str(noisy_path),
                    "target_path": str(target_path),
                }
            )
        return samples


class MatchedNameSampleBuilder:
    def __init__(self):
        pass

    def build(
        self,
        noisy_dir: Path,
        target_dir: Path,
        file_suffix: str,
        data_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        noisy_files = {
            p.name: p for p in noisy_dir.glob(f"*{file_suffix}") if p.is_file()
        }
        target_files = {
            p.name: p for p in target_dir.glob(f"*{file_suffix}") if p.is_file()
        }
        common_names = sorted(set(noisy_files.keys()) & set(target_files.keys()))
        if data_count is not None:
            common_names = common_names[: int(data_count)]

        for name in common_names:
            samples.append(
                {
                    "name": name,
                    "noisy_path": str(noisy_files[name]),
                    "target_path": str(target_files[name]),
                }
            )
        return samples


class DefaultSTEMDatasetDownloader:
    def __init__(self, datasets_home: Optional[str] = None):
        self.datasets_home = datasets_home or download_utils.DATASETS_HOME

    def download(
        self, url: str, md5: Optional[str] = None, force_download: bool = False
    ) -> Path:
        if force_download:
            downloaded_root = download_utils.get_path_from_url(
                url,
                self.datasets_home,
                md5sum=md5,
                check_exist=False,
                decompress=True,
            )
        else:
            downloaded_root = download_utils.get_datasets_path_from_url(url, md5)
        return Path(downloaded_root)


class PairDirectoryDataRootResolver:
    def __init__(self, max_depth: int = 2):
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, but got {max_depth}")
        self.max_depth = int(max_depth)

    @staticmethod
    def _contains_pair_dirs(root: Path, noisy_subdir: str, target_subdir: str) -> bool:
        return (
            root.is_dir()
            and (root / noisy_subdir).exists()
            and (root / target_subdir).exists()
        )

    def find_data_roots(
        self,
        base_root: Path,
        split: Optional[str],
        noisy_subdir: str,
        target_subdir: str,
    ) -> List[Path]:
        if not base_root.exists():
            return []

        candidate_roots: List[Path] = [base_root]
        frontier: List[Path] = [base_root]
        for _ in range(self.max_depth):
            next_frontier: List[Path] = []
            for root in frontier:
                for child in root.iterdir():
                    if child.is_dir():
                        candidate_roots.append(child)
                        next_frontier.append(child)
            frontier = next_frontier

        matches: List[Path] = []
        for root in candidate_roots:
            if split is not None:
                split_root = root / split
                if self._contains_pair_dirs(split_root, noisy_subdir, target_subdir):
                    matches.append(split_root)
            if self._contains_pair_dirs(root, noisy_subdir, target_subdir):
                matches.append(root)

        seen = set()
        unique_matches = []
        for path in matches:
            path_str = str(path)
            if path_str in seen:
                continue
            seen.add(path_str)
            unique_matches.append(path)
        return unique_matches


def build_stem_sample_builder(
    cfg: Optional[Dict],
    *,
    strict_index_naming: bool,
):
    default_class_name = (
        "StrictIndexSampleBuilder"
        if strict_index_naming
        else "MatchedNameSampleBuilder"
    )
    sample_builder = _build_component(
        cfg,
        default_class_name=default_class_name,
        required_methods=["build"],
    )
    logger.debug(f"Use sample builder: {sample_builder.__class__.__name__}")
    return sample_builder


def build_stem_downloader(cfg: Optional[Dict]):
    downloader = _build_component(
        cfg,
        default_class_name="DefaultSTEMDatasetDownloader",
        required_methods=["download"],
    )
    logger.debug(f"Use downloader: {downloader.__class__.__name__}")
    return downloader


def build_stem_data_root_resolver(cfg: Optional[Dict]):
    resolver = _build_component(
        cfg,
        default_class_name="PairDirectoryDataRootResolver",
        required_methods=["find_data_roots"],
    )
    logger.debug(f"Use data root resolver: {resolver.__class__.__name__}")
    return resolver


class STEMImageDataset(paddle.io.Dataset):
    """Dataset for paired STEM image restoration/enhancement.

    Supports automatic download and extraction (zip/tar/tar.gz) through
    ``ppmat.utils.download.get_datasets_path_from_url``.

    Expected directory layout after extraction:
        data_path/
          train/
            noisy/
              0.png
              1.png
              ...
            gt_enhance/
              0.png
              1.png
              ...
          val/
            noisy/
              ...
            gt_enhance/
              ...
          test/
            noisy/
              ...
            gt_enhance/
              ...

    Or legacy format (backward compatible):
        data_path/
          noisy/
            0.png
            ...
          gt_enhance/
            0.png
            ...
    """

    name = "stem_enhancement"
    url = None
    md5 = None
    _DEFAULT_URL_MAP = {
        "data": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data.zip",
        "data_test": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data_test.zip",
        "bf_data": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data.zip",
        "bf_data_test": "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data_test.zip",
    }
    _DEFAULT_MD5_MAP: Dict[str, str] = {}

    def __init__(
        self,
        data_path: str,
        split: Optional[str] = None,
        data_count: int | None = None,
        noisy_subdir: str = "noisy",
        target_subdir: str = "gt_enhance",
        file_suffix: str = ".png",
        strict_index_naming: bool = True,
        sample_builder_cfg: Optional[Dict] = None,
        downloader_cfg: Optional[Dict] = None,
        data_root_resolver_cfg: Optional[Dict] = None,
        scale_to_unit: bool = False,
        url: Optional[str] = None,
        md5: Optional[str] = None,
        download: bool = True,
        force_download: bool = False,
    ):
        """Initialize STEMImageDataset.

        Args:
            data_path: Root directory for the dataset.
            split: Dataset split, one of 'train', 'val', 'test', or None.
                   If None, uses legacy format without split subdirectories.
            data_count: Maximum number of samples to load. None means all.
            noisy_subdir: Subdirectory name for noisy images.
            target_subdir: Subdirectory name for target/ground truth images.
            file_suffix: File extension for images (e.g., '.png', '.tif').
            strict_index_naming: If True, expects files named as {idx}{suffix}.
            sample_builder_cfg: Sample builder config in factory style:
                {
                    "__class_name__": "StrictIndexSampleBuilder",
                    "__init_params__": {}
                }
            downloader_cfg: Downloader config in factory style.
                Default class is "DefaultSTEMDatasetDownloader".
            data_root_resolver_cfg: Data root resolver config in factory style.
                Default class is "PairDirectoryDataRootResolver".
            scale_to_unit: If True, scales pixel values to [0, 1].
            url: URL to download the dataset from. Overrides default URL.
            md5: MD5 checksum for downloaded file. Optional.
            download: Whether to automatically download if data not found.
            force_download: If True, re-download even if data exists.
        """
        super().__init__()

        self.split = split
        self.data_count = data_count
        self.noisy_subdir = noisy_subdir
        self.target_subdir = target_subdir
        self.file_suffix = file_suffix
        self.strict_index_naming = strict_index_naming
        self.scale_to_unit = scale_to_unit
        self.sample_builder = build_stem_sample_builder(
            sample_builder_cfg,
            strict_index_naming=strict_index_naming,
        )
        self.downloader = build_stem_downloader(downloader_cfg)
        self.data_root_resolver = build_stem_data_root_resolver(data_root_resolver_cfg)

        self.url = url if url is not None else self._infer_default_url(data_path)
        self.md5 = (
            md5 if md5 is not None else self._infer_default_md5(data_path) or self.md5
        )
        self.data_root = Path(data_path)
        self.downloaded_root: Optional[Path] = None

        if self._locate_data_root(self.data_root) is None and (
            download or force_download
        ):
            self.downloaded_root = self._download_dataset(force_download)

        # Determine actual data directory based on split
        self.data_dir = self._resolve_data_dir()

        # Set up noisy and target directories
        self.noisy_dir = self.data_dir / noisy_subdir
        self.target_dir = self.data_dir / target_subdir

        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        self.samples = self._build_samples(data_count)

    def _resolve_data_dir(self) -> Path:
        """Resolve the actual data directory based on split configuration."""
        candidate_roots = [self.data_root]
        if self.downloaded_root is not None:
            for root in [self.downloaded_root, self.downloaded_root.parent]:
                if root != self.data_root and root not in candidate_roots:
                    candidate_roots.append(root)

        for candidate_root in candidate_roots:
            matches = self._find_data_roots(candidate_root)
            if not matches:
                continue
            if (
                self.downloaded_root is not None
                and candidate_root == self.downloaded_root.parent
                and len(matches) > 1
            ):
                raise FileNotFoundError(
                    "Multiple candidate dataset roots were found under "
                    f"'{candidate_root}': {[str(m) for m in matches]}. "
                    "Please provide a more specific local `data_path` or explicit `url`."
                )

            data_root_candidate = matches[0]
            if self.split is not None and data_root_candidate == candidate_root:
                logger.warning(
                    f"Split '{self.split}' requested but legacy format detected. "
                    f"Using data directly from {candidate_root}"
                )
            return data_root_candidate

        searched_roots = ", ".join([str(path) for path in candidate_roots])
        if self.split is not None:
            raise FileNotFoundError(
                f"Split '{self.split}' not found under: {searched_roots}"
            )
        raise FileNotFoundError(
            "Cannot locate dataset directories "
            f"'{self.noisy_subdir}' and '{self.target_subdir}' under: {searched_roots}"
        )

    def _download_dataset(self, force_download: bool = False) -> Path:
        """Download dataset with built-in ppmat factory utility."""
        if not self.url:
            candidate = ", ".join(sorted(self._DEFAULT_URL_MAP.keys()))
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_root}', and no download URL provided. "
                f"Auto-url is only inferred for data_path basename in [{candidate}]."
            )

        logger.message(
            f"Dataset root {self.data_root} not found. Will download it now."
        )
        downloaded_root = self.downloader.download(
            self.url,
            self.md5,
            force_download=force_download,
        )
        logger.info(f"Dataset downloaded to: {downloaded_root}")
        return Path(downloaded_root)

    @classmethod
    def _infer_default_url(cls, data_path: str) -> Optional[str]:
        key = Path(data_path).name
        url = cls._DEFAULT_URL_MAP.get(key)
        if url is not None:
            logger.info(
                f"Infer dataset download URL by data_path='{data_path}': {url}"
            )
        return url

    @classmethod
    def _infer_default_md5(cls, data_path: str) -> Optional[str]:
        key = Path(data_path).name
        md5 = cls._DEFAULT_MD5_MAP.get(key)
        if md5 is not None:
            logger.info(
                f"Infer dataset md5 by data_path='{data_path}': {md5}"
            )
        return md5

    def _locate_data_root(self, base_root: Path) -> Optional[Path]:
        matches = self._find_data_roots(base_root)
        if not matches:
            return None
        return matches[0]

    def _find_data_roots(self, base_root: Path) -> List[Path]:
        return self.data_root_resolver.find_data_roots(
            base_root=base_root,
            split=self.split,
            noisy_subdir=self.noisy_subdir,
            target_subdir=self.target_subdir,
        )

    def _build_samples(self, data_count: int | None) -> List[Dict[str, str]]:
        """Build list of sample dictionaries."""
        return self.sample_builder.build(
            noisy_dir=self.noisy_dir,
            target_dir=self.target_dir,
            file_suffix=self.file_suffix,
            data_count=data_count,
        )

    def __len__(self):
        return len(self.samples)

    def _load_gray_image(self, path: str) -> paddle.Tensor:
        """Load image as grayscale tensor."""
        image = Image.open(path).convert("L")
        image_array = np.asarray(image, dtype=np.float32)
        if self.scale_to_unit:
            image_array = image_array / 255.0
        return paddle.to_tensor(image_array).unsqueeze(0)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        noisy = self._load_gray_image(sample["noisy_path"])
        target = self._load_gray_image(sample["target_path"])

        output = {
            "noisy": noisy,
            self.target_subdir: target,
            "target": target,
            "name": sample["name"],
        }
        # Backward compatibility for legacy code paths that read `gt_enhance`.
        if self.target_subdir != "gt_enhance":
            output["gt_enhance"] = target
        return output
