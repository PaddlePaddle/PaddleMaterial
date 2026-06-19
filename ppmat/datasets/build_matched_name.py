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
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union


def build_matched_name_samples(cfg: Dict):
    """Build sample matcher from config."""
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__")
    if not class_name:
        raise ValueError(
            "Sample matcher class name is not specified in the configuration."
        )
    init_params = cfg.pop("__init_params__")
    cls = _locate_class(class_name)
    return cls(**init_params)


class BuildMatchedNameSamples:
    """Match noisy and target samples by identical file names."""

    def __init__(
        self,
        noisy_file_key: str = "noisy_file",
        target_file_key: str = "target_file",
        file_key: str = "file_name",
        noisy_key: str = "noisy",
        target_key: str = "target",
        name_key: str = "name",
    ):
        self.noisy_file_key = noisy_file_key
        self.target_file_key = target_file_key
        self.file_key = file_key
        self.noisy_key = noisy_key
        self.target_key = target_key
        self.name_key = name_key

    @staticmethod
    def build_one(
        file_data: Union[Dict[str, str], str],
        noisy_file_key: str,
        target_file_key: str,
        file_key: str,
        noisy_key: str,
        target_key: str,
        name_key: str,
    ) -> Dict[str, str]:
        if isinstance(file_data, dict):
            if file_key in file_data:
                noisy_file = file_data.get(file_key)
                target_file = noisy_file
            else:
                noisy_file = file_data.get(noisy_file_key)
                target_file = file_data.get(target_file_key)
        else:
            noisy_file = file_data
            target_file = file_data
        if not isinstance(noisy_file, str) or not noisy_file:
            raise ValueError(
                f"Expected non-empty noisy file name, but got {noisy_file}."
            )
        if not isinstance(target_file, str) or not target_file:
            raise ValueError(
                f"Expected non-empty target file name, but got {target_file}."
            )
        if noisy_file != target_file:
            raise ValueError(
                "Matched-name samples require identical noisy and target file names, "
                f"but got {noisy_file} and {target_file}."
            )
        return {
            noisy_key: noisy_file,
            target_key: target_file,
            name_key: noisy_file,
        }

    def __call__(
        self,
        file_names: Union[
            Sequence[Union[Dict[str, str], str]],
            Dict[str, str],
            str,
        ],
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if isinstance(file_names, (list, tuple)):
            if len(file_names) == 0:
                return []
            return [
                BuildMatchedNameSamples.build_one(
                    file_name,
                    self.noisy_file_key,
                    self.target_file_key,
                    self.file_key,
                    self.noisy_key,
                    self.target_key,
                    self.name_key,
                )
                for file_name in file_names
            ]
        return BuildMatchedNameSamples.build_one(
            file_names,
            self.noisy_file_key,
            self.target_file_key,
            self.file_key,
            self.noisy_key,
            self.target_key,
            self.name_key,
        )


class BuildIndexedNameSamples:
    """Match noisy and target samples by integer file stem."""

    def __init__(
        self,
        noisy_file_key: str = "noisy_file",
        target_file_key: str = "target_file",
        noisy_key: str = "noisy",
        target_key: str = "target",
        name_key: str = "name",
    ):
        self.noisy_file_key = noisy_file_key
        self.target_file_key = target_file_key
        self.noisy_key = noisy_key
        self.target_key = target_key
        self.name_key = name_key

    @staticmethod
    def build_one(
        sample_data: Dict[str, str],
        noisy_file_key: str,
        target_file_key: str,
        noisy_key: str,
        target_key: str,
        name_key: str,
    ) -> Dict[str, str]:
        if not isinstance(sample_data, dict):
            raise TypeError(
                f"Indexed sample data must be a dict, but got {type(sample_data)}."
            )
        noisy_file = sample_data.get(noisy_file_key)
        target_file = sample_data.get(target_file_key)
        if not isinstance(noisy_file, str) or not noisy_file:
            raise ValueError(
                f"Expected non-empty noisy file name, but got {noisy_file}."
            )
        if not isinstance(target_file, str) or not target_file:
            raise ValueError(
                f"Expected non-empty target file name, but got {target_file}."
            )
        noisy_stem = osp.splitext(noisy_file)[0]
        target_stem = osp.splitext(target_file)[0]
        if not noisy_stem.isdigit() or not target_stem.isdigit():
            raise ValueError(
                "Indexed-name samples require integer file stems, but got "
                f"{noisy_file} and {target_file}."
            )
        if int(noisy_stem) != int(target_stem):
            raise ValueError(
                "Indexed-name samples require matching integer file stems, but got "
                f"{noisy_file} and {target_file}."
            )
        return {
            noisy_key: noisy_file,
            target_key: target_file,
            name_key: noisy_file,
        }

    def __call__(
        self,
        sample_data_list: Union[
            Sequence[Dict[str, str]],
            Dict[str, str],
        ],
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if isinstance(sample_data_list, (list, tuple)):
            if len(sample_data_list) == 0:
                return []
            return [
                BuildIndexedNameSamples.build_one(
                    sample_data,
                    self.noisy_file_key,
                    self.target_file_key,
                    self.noisy_key,
                    self.target_key,
                    self.name_key,
                )
                for sample_data in sample_data_list
            ]
        return BuildIndexedNameSamples.build_one(
            sample_data_list,
            self.noisy_file_key,
            self.target_file_key,
            self.noisy_key,
            self.target_key,
            self.name_key,
        )


def _locate_class(class_name: str) -> Any:
    """Resolve 'pkg.mod.Class' or a bare class name in the current globals()."""
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    return globals()[class_name]
