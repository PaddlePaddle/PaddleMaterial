# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import ast

import numpy as np


def read_json_lines(path):
    """
    Read all lines from a line-delimited Python-dict-like file,
    extracting all properties into a dictionary of lists.
    """
    property_data = {}
    
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            content = ast.literal_eval(line.strip())

            if idx == 0:
                all_property_names = list(content.keys())
                # print("all_property_names:", all_property_names)
                property_data = {name: [] for name in all_property_names}

            for property_name in all_property_names:
                if property_name not in content:
                    raise ValueError(f"'{property_name}' not found in line {idx + 1} of file")
                property_data[property_name].append(content[property_name])    
    return property_data



def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    def handler(obj: object) -> (int | object):
        """Convert numpy int64 to int.

        Fixes TypeError: Object of type int64 is not JSON serializable
        reported in https://github.com/CederGroupHub/chgnet/issues/168.

        Returns:
            int | object: object for serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=handler)


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


def calc_md5(fullname):
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    return calc_md5sum


if __name__ == "__main__":
    md5 = calc_md5("yourfile.zip")
    print(md5)
