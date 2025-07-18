# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import os
import os.path as osp
import shutil
import tarfile
import time
import zipfile

import requests
import tqdm

from mattergen.utils import logger
from mattergen.utils import misc

__all__ = ["get_weights_path_from_url"]

WEIGHTS_HOME = osp.expanduser("~/.paddlemat/weights")

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.

    Args:
        path (str): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def get_weights_path_from_url(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.

    Args:
        url (str): Download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded weights.
    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True, decompress=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): Download url
        root_dir (str): Root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """
    if not is_url(url):
        raise ValueError(f"Given url({url}) is not valid")
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)
    # Mainly used to solve the problem of downloading data from different
    # machines in the case of multiple machines. Different nodes will download
    # data, and the same node will only download data once.
    rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.message(f"Found {fullpath} already in {WEIGHTS_HOME}, skip downloading.")
    else:
        with misc.RankZeroOnly(rank_id_curr_node) as is_master:
            if is_master:
                fullpath = _download(url, root_dir, md5sum)

    if decompress and (tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath)):
        with misc.RankZeroOnly(rank_id_curr_node) as is_master:
            if is_master:
                fullpath = _decompress(fullpath)

    return fullpath


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): Download url
    path (str): Download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(f"Download from {url} failed. " "Retry limit reached")

        logger.message(f"Downloading {fname} from {url}")

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.warning(
                f"Downloading {fname} from {url} failed {retry_cnt + 1} times with "
                f"exception {str(e)}"
            )
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError(
                f"Downloading from {url} failed with code " f"{req.status_code}!"
            )

        # For protecting download interrupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm.tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
        logger.message(f"Finish downloading pretrained model and saved to {fullname}")

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.message(f"File {fullname} md5 checking...")
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.error(
            f"File {fullname} md5 check failed, {calc_md5sum}(calc) != "
            f"{md5sum}(base)"
        )
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.message(f"Decompressing {fname}...")

    # For protecting decompressing interrupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # succeed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError(f"Unsupported compress file type {fname}")

    return uncompressed_path


def _uncompress_file_zip(filepath):
    with zipfile.ZipFile(filepath, "r") as files:
        file_list = files.namelist()

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)

            for item in file_list:
                files.extract(item, file_dir)

        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)

            for item in file_list:
                files.extract(item, file_dir)

        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)
            for item in file_list:
                files.extract(item, os.path.join(file_dir, rootpath))

    return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    with tarfile.open(filepath, mode) as files:
        file_list = files.getnames()

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            for item in file_list:
                files.extract(item, file_dir)
        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            for item in file_list:
                files.extract(item, file_dir)
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)

            for item in file_list:
                files.extract(item, os.path.join(file_dir, rootpath))

    return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if "/" in file_path:
            file_path = file_path.replace("/", os.sep)
        elif "\\" in file_path:
            file_path = file_path.replace("\\", os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True
