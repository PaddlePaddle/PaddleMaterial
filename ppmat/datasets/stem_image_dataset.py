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

from typing import Any
from typing import Callable
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import paddle
from PIL import Image

from ppmat.utils import download
from ppmat.utils import logger


class STEMImageDataset(paddle.io.Dataset):
    """Dataset for paired STEM image restoration/enhancement.

    Supports automatic download and extraction through ``ppmat.utils.download``.

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
        scale_to_unit: bool = False,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
            transforms: Optional per-sample transforms built by ``build_dataloader``.
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
        self.transforms = transforms

        self.url = url if url is not None else self._infer_default_url(data_path)
        self.md5 = (
            md5 if md5 is not None else self._infer_default_md5(data_path) or self.md5
        )
        if data_count is not None and int(data_count) < 0:
            raise ValueError("`data_count` must be None or a non-negative integer.")
        self.data_root = Path(data_path)
        self.downloaded_root = self._maybe_download_dataset(
            download=download, force_download=force_download
        )

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
        candidate_roots = self._get_candidate_roots()

        for candidate_root in candidate_roots:
            matches = self._find_data_roots(candidate_root)
            selected_root = self._select_data_root(candidate_root, matches)
            if selected_root is None:
                continue
            data_root_candidate = selected_root
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

    def _maybe_download_dataset(
        self, download: bool = True, force_download: bool = False
    ) -> Optional[Path]:
        """Download dataset when local data root cannot be resolved."""
        if not force_download and self._locate_data_root(self.data_root) is not None:
            return None
        if not (download or force_download):
            return None
        return self._download_dataset(force_download=force_download)

    def _download_dataset(self, force_download: bool = False) -> Path:
        """Delegate dataset download to the shared ppmat download utility."""
        if not self.url:
            candidate = ", ".join(sorted(self._DEFAULT_URL_MAP.keys()))
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_root}', and no download URL provided. "
                f"Auto-url is only inferred for data_path basename in [{candidate}]."
            )

        logger.message(
            f"Dataset root {self.data_root} not found. Will download it now."
        )
        if force_download:
            downloaded_root = download.get_path_from_url(
                self.url,
                download.DATASETS_HOME,
                md5sum=self.md5,
                check_exist=False,
                decompress=True,
            )
        else:
            downloaded_root = download.get_datasets_path_from_url(self.url, self.md5)
        logger.info(f"Dataset downloaded to: {downloaded_root}")
        return Path(downloaded_root)

    def _get_candidate_roots(self) -> List[Path]:
        candidate_roots = [self.data_root]
        if self.downloaded_root is not None:
            for root in (self.downloaded_root, self.downloaded_root.parent):
                if root != self.data_root and root not in candidate_roots:
                    candidate_roots.append(root)
        return candidate_roots

    def _select_data_root(
        self, candidate_root: Path, matches: List[Path]
    ) -> Optional[Path]:
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        preferred_names = self._get_preferred_root_names()
        for preferred_name in preferred_names:
            preferred_matches = [path for path in matches if path.name == preferred_name]
            if len(preferred_matches) == 1:
                logger.info(
                    "Resolved dataset root '%s' under '%s' from multiple candidates: %s"
                    % (
                        preferred_matches[0],
                        candidate_root,
                        [str(path) for path in matches],
                    )
                )
                return preferred_matches[0]

        raise FileNotFoundError(
            "Multiple candidate dataset roots were found under "
            f"'{candidate_root}': {[str(path) for path in matches]}. "
            f"Tried preferred names: {preferred_names or ['<none>']}. "
            "Please provide a more specific local `data_path` or explicit `url`."
        )

    def _get_preferred_root_names(self) -> List[str]:
        preferred_names: List[str] = []
        for name in (self.data_root.name, self._infer_download_root_name()):
            if name and name not in preferred_names:
                preferred_names.append(name)
        return preferred_names

    def _infer_download_root_name(self) -> Optional[str]:
        if not self.url:
            return None
        parsed_path = urlparse(self.url).path
        archive_name = Path(parsed_path).name
        if not archive_name:
            return None
        return Path(archive_name).stem

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
        return self._select_data_root(base_root, matches)

    def _find_data_roots(self, base_root: Path) -> List[Path]:
        if not base_root.exists():
            return []

        candidate_roots: List[Path] = [base_root]
        frontier: List[Path] = [base_root]
        for _ in range(2):
            next_frontier: List[Path] = []
            for root in frontier:
                for child in sorted(root.iterdir()):
                    if child.is_dir():
                        candidate_roots.append(child)
                        next_frontier.append(child)
            frontier = next_frontier

        matches: List[Path] = []
        for root in candidate_roots:
            if self.split is not None:
                split_root = root / self.split
                if self._contains_pair_dirs(split_root):
                    matches.append(split_root)
            if self._contains_pair_dirs(root):
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

    def _contains_pair_dirs(self, root: Path) -> bool:
        return (
            root.is_dir()
            and (root / self.noisy_subdir).exists()
            and (root / self.target_subdir).exists()
        )

    def _list_image_files(self, directory: Path) -> List[Path]:
        return sorted(
            path for path in directory.glob(f"*{self.file_suffix}") if path.is_file()
        )

    def _build_samples(self, data_count: int | None) -> List[Dict[str, str]]:
        """Build list of sample dictionaries."""
        if not self.strict_index_naming:
            return self._build_matched_name_samples(data_count)
        return self._build_indexed_samples(data_count)

    def _build_indexed_samples(self, data_count: int | None) -> List[Dict[str, str]]:
        noisy_files = self._list_image_files(self.noisy_dir)
        target_files = self._list_image_files(self.target_dir)
        noisy_index_map = self._build_index_map(noisy_files, self.noisy_dir)
        target_index_map = self._build_index_map(target_files, self.target_dir)

        if data_count is None:
            common_indices = sorted(set(noisy_index_map) & set(target_index_map))
            if not common_indices:
                raise FileNotFoundError(
                    "No matched indexed image pairs were found under "
                    f"'{self.noisy_dir}' and '{self.target_dir}'."
                )
            max_index = common_indices[-1]
            missing_indices = [
                idx
                for idx in range(max_index + 1)
                if idx not in noisy_index_map or idx not in target_index_map
            ]
            if missing_indices:
                raise FileNotFoundError(
                    "Strict indexed naming expects contiguous pairs from 0. "
                    f"Missing indices: {missing_indices[:10]}. "
                    "Use `strict_index_naming=False` for arbitrary filenames."
                )
            expected_indices = list(range(max_index + 1))
        else:
            expected_indices = list(range(int(data_count)))

        samples: List[Dict[str, str]] = []
        for idx in expected_indices:
            noisy_path = noisy_index_map.get(idx)
            target_path = target_index_map.get(idx)
            if noisy_path is None:
                raise FileNotFoundError(
                    f"Noisy image for index {idx} not found in '{self.noisy_dir}'."
                )
            if target_path is None:
                raise FileNotFoundError(
                    f"Target image for index {idx} not found in '{self.target_dir}'."
                )
            if not noisy_path.exists():
                raise FileNotFoundError(f"Noisy image not found: {noisy_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
            samples.append(
                {
                    "name": noisy_path.name,
                    "noisy_path": str(noisy_path),
                    "target_path": str(target_path),
                }
            )
        return samples

    def _build_index_map(
        self, files: List[Path], directory: Path
    ) -> Dict[int, Path]:
        index_map: Dict[int, Path] = {}
        invalid_names: List[str] = []
        for path in files:
            if not path.stem.isdigit():
                invalid_names.append(path.name)
                continue
            index = int(path.stem)
            if index in index_map:
                raise ValueError(
                    f"Duplicate indexed file '{path.name}' found in '{directory}'."
                )
            index_map[index] = path

        if invalid_names:
            raise ValueError(
                "Strict indexed naming requires filenames like '0"
                f"{self.file_suffix}'. Invalid files in '{directory}': "
                f"{invalid_names[:10]}"
            )
        return index_map

    def _build_matched_name_samples(
        self, data_count: int | None
    ) -> List[Dict[str, str]]:
        noisy_files = {
            path.name: path
            for path in self._list_image_files(self.noisy_dir)
        }
        target_files = {
            path.name: path
            for path in self._list_image_files(self.target_dir)
        }
        common_names = sorted(set(noisy_files.keys()) & set(target_files.keys()))
        if not common_names:
            raise FileNotFoundError(
                "No matched image pairs were found under "
                f"'{self.noisy_dir}' and '{self.target_dir}'."
            )
        if data_count is not None:
            common_names = common_names[: int(data_count)]

        return [
            {
                "name": name,
                "noisy_path": str(noisy_files[name]),
                "target_path": str(target_files[name]),
            }
            for name in common_names
        ]

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
        if self.transforms is not None:
            output = self.transforms(output)
        return output
