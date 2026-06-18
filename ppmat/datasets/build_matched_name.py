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
import os.path as osp
from typing import Dict
from typing import List
from typing import Optional


def build_matched_name_samples(cfg: Dict):
    """Build sample matcher from config."""
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    init_params = cfg.pop("__init_params__")
    cls = _locate_class(class_name)
    return cls(**init_params)


class BuildMatchedNameSamples:
    """Match noisy and target samples by identical file names."""

    def __init__(self):
        pass

    @staticmethod
    def build_one(file_name: str) -> Dict[str, str]:
        return {
            "noisy": file_name,
            "target": file_name,
            "name": file_name,
        }

    def __call__(
        self,
        noisy_files: List[str],
        target_files: List[str],
        *,
        noisy_root: str,
        target_root: str,
        data_count: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        del kwargs

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

        common_names = sorted(noisy_file_set & target_file_set)
        if data_count is not None:
            common_names = common_names[: int(data_count)]

        return [
            BuildMatchedNameSamples.build_one(file_name) for file_name in common_names
        ]


class BuildIndexedNameSamples:
    """Match noisy and target samples by integer file stem."""

    def __init__(self):
        pass

    @staticmethod
    def build_one(noisy_file: str, target_file: str) -> Dict[str, str]:
        return {
            "noisy": noisy_file,
            "target": target_file,
            "name": noisy_file,
        }

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

    def __call__(
        self,
        noisy_files: List[str],
        target_files: List[str],
        *,
        noisy_root: str,
        target_root: str,
        file_suffix: str,
        data_count: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        del kwargs

        noisy_map = self._build_index_map(noisy_files, noisy_root, file_suffix)
        target_map = self._build_index_map(target_files, target_root, file_suffix)
        if not noisy_map:
            raise FileNotFoundError(
                f"No indexed noisy images found under {noisy_root}."
            )
        if not target_map:
            raise FileNotFoundError(
                f"No indexed target images found under {target_root}."
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

        if data_count is not None:
            common_indices = common_indices[: int(data_count)]

        return [
            BuildIndexedNameSamples.build_one(noisy_map[idx], target_map[idx])
            for idx in common_indices
        ]


def _locate_class(class_name: str):
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    if class_name not in globals():
        raise ValueError(f"Unknown sample matcher class: {class_name}")
    return globals()[class_name]
