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
from typing import Dict
from typing import List
from typing import Optional

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
