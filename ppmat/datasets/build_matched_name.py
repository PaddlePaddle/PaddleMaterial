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
        self, file_names: Union[Sequence[str], str]
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if isinstance(file_names, (list, tuple)):
            return [
                BuildMatchedNameSamples.build_one(file_name) for file_name in file_names
            ]
        return BuildMatchedNameSamples.build_one(file_names)


class BuildIndexedNameSamples:
    """Match noisy and target samples by integer file stem."""

    def __init__(self):
        pass

    @staticmethod
    def build_one(file_pair: Sequence[str]) -> Dict[str, str]:
        noisy_file, target_file = file_pair
        return {
            "noisy": noisy_file,
            "target": target_file,
            "name": noisy_file,
        }

    def __call__(
        self, file_pairs: Union[Sequence[Sequence[str]], Sequence[str]]
    ) -> Union[List[Dict[str, str]], Dict[str, str]]:
        if (
            isinstance(file_pairs, (list, tuple))
            and len(file_pairs) > 0
            and isinstance(file_pairs[0], (list, tuple))
        ):
            return [
                BuildIndexedNameSamples.build_one(file_pair)
                for file_pair in file_pairs
            ]
        return BuildIndexedNameSamples.build_one(file_pairs)


def _locate_class(class_name: str) -> Any:
    """Resolve 'pkg.mod.Class' or a bare class name in the current globals()."""
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    return globals()[class_name]
