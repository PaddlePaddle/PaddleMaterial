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

import os
import os.path as osp

from ppmat.utils import logger


def find_file_in_package(package_path: str, file_name: str):
    if osp.isfile(package_path):
        if osp.basename(package_path) == file_name:
            return package_path
        raise FileNotFoundError(f"No such file named {file_name} in {package_path}")

    for root, _, files in os.walk(package_path):
        for name in files:
            if osp.basename(name) == file_name:
                return osp.join(root, name)

    raise FileNotFoundError(f"No such file named {file_name} in {package_path}")


def find_config_file_in_package(model_name: str, package_path: str):
    for config_name in (f"{model_name}.yaml", f"{model_name}.yml"):
        try:
            return find_file_in_package(package_path, config_name)
        except FileNotFoundError:
            pass

    find_list = []
    for root, _, files in os.walk(package_path):
        for name in files:
            if name.endswith(".yaml") or name.endswith(".yml"):
                find_list.append(osp.join(root, name))

    if len(find_list) == 1:
        config_path = find_list[0]
        logger.warning(f"Find config file: {config_path}, using this file.")
        return config_path

    raise ValueError(f"Multiple yaml files found: {find_list}, must be only one")
