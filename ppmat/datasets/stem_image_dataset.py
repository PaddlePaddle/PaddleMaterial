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

from __future__ import annotations

import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
import paddle.distributed as dist
from PIL import Image

from ppmat.utils import logger


class STEMImageDataset(paddle.io.Dataset):
    """Dataset for paired STEM image restoration/enhancement.

    Supports automatic download and extraction of zip/tar/tar.gz datasets.

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

    def __init__(
        self,
        data_path: str,
        split: Optional[str] = None,
        data_count: int | None = None,
        noisy_subdir: str = "noisy",
        target_subdir: str = "gt_enhance",
        file_suffix: str = ".png",
        strict_index_naming: bool = True,
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

        self.url = url
        self.md5 = md5

        # Set up paths
        self.data_root = Path(data_path)
        self.raw_dir = self.data_root / "raw"
        self.extracted_dir = self.data_root / "extracted"

        # Handle download and extraction
        if download or force_download:
            self._prepare_data(force_download)

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
        data_root_candidate = self._locate_data_root(self.data_root)
        if data_root_candidate is not None:
            if self.split is not None and data_root_candidate == self.data_root:
                logger.warning(
                    f"Split '{self.split}' requested but legacy format detected. "
                    f"Using data directly from {self.data_root}"
                )
            return data_root_candidate

        extracted_candidate = self._locate_data_root(self.extracted_dir)
        if extracted_candidate is not None:
            return extracted_candidate

        if self.split is not None:
            raise FileNotFoundError(
                f"Split '{self.split}' not found under '{self.data_root}' or "
                f"'{self.extracted_dir}'."
            )
        return self.data_root

    def _prepare_data(self, force_download: bool = False) -> None:
        """Download and extract dataset if necessary."""
        # Check if data already exists
        if not force_download and self._data_exists():
            logger.info(f"Dataset already exists at {self.data_root}")
            return

        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.extracted_dir, exist_ok=True)

        # Download
        tar_path = self._download_data()

        # Extract
        self._extract_data(tar_path)

    def _data_exists(self) -> bool:
        """Check if extracted data already exists."""
        return (
            self._locate_data_root(self.data_root) is not None
            or self._locate_data_root(self.extracted_dir) is not None
        )

    def _download_data(self) -> Path:
        """Download dataset from URL."""
        archive_name = os.path.basename(self.url)
        archive_path = self.raw_dir / archive_name

        if archive_path.exists():
            logger.info(f"Archive already downloaded: {archive_path}")
            return archive_path

        if dist.get_rank() == 0:
            logger.info(f"Downloading dataset from {self.url}...")
            try:
                urllib.request.urlretrieve(self.url, archive_path)
                logger.info(f"Downloaded to {archive_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")

        if dist.is_initialized():
            dist.barrier()

        return archive_path

    def _extract_data(self, archive_path: Path) -> None:
        """Extract downloaded archive (zip/tar/tar.gz)."""
        if dist.get_rank() == 0:
            logger.info(f"Extracting {archive_path}...")

            try:
                suffix = archive_path.suffix.lower()
                if suffix == ".zip":
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        zf.extractall(path=self.extracted_dir)
                elif tarfile.is_tarfile(archive_path):
                    with tarfile.open(archive_path, "r:*") as tf:
                        tf.extractall(path=self.extracted_dir)
                else:
                    raise RuntimeError(
                        f"Unsupported archive format for '{archive_path.name}'. "
                        "Only zip/tar/tar.gz/tgz are supported."
                    )
                logger.info(f"Extracted to {self.extracted_dir}")
            except (tarfile.TarError, zipfile.BadZipFile) as e:
                raise RuntimeError(f"Failed to extract archive: {e}")

        if dist.is_initialized():
            dist.barrier()

    def _contains_pair_dirs(self, root: Path) -> bool:
        return (
            root.is_dir()
            and (root / self.noisy_subdir).exists()
            and (root / self.target_subdir).exists()
        )

    def _locate_data_root(self, base_root: Path) -> Optional[Path]:
        if not base_root.exists():
            return None

        # Try current directory then its first-level subdirectories.
        candidate_roots = [base_root]
        candidate_roots.extend(p for p in base_root.iterdir() if p.is_dir())

        for root in candidate_roots:
            if self.split is not None:
                split_root = root / self.split
                if self._contains_pair_dirs(split_root):
                    return split_root
            if self._contains_pair_dirs(root):
                return root
        return None

    def _build_samples(self, data_count: int | None) -> List[Dict[str, str]]:
        """Build list of sample dictionaries."""
        samples: List[Dict[str, str]] = []

        if self.strict_index_naming:
            if data_count is None:
                available = sorted(self.noisy_dir.glob(f"*{self.file_suffix}"))
                data_count = len(available)

            for idx in range(int(data_count)):
                name = f"{idx}{self.file_suffix}"
                noisy_path = self.noisy_dir / name
                target_path = self.target_dir / name
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

        noisy_files = {
            p.name: p
            for p in self.noisy_dir.glob(f"*{self.file_suffix}")
            if p.is_file()
        }
        target_files = {
            p.name: p
            for p in self.target_dir.glob(f"*{self.file_suffix}")
            if p.is_file()
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
