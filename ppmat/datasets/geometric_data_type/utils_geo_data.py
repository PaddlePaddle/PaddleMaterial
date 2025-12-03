# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from https://github.com/jediofgever/pytorch_geometric

import os
import ssl
import urllib
import zipfile


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    """Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)
    if osp.exists(path):
        if log:
            print("Using exist file", filename)
        return path
    if log:
        print("Downloading", url)
    makedirs(folder)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(path, "wb") as f:
        f.write(data.read())
    return path


def extract_zip(path, folder, log=True):
    """Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)
