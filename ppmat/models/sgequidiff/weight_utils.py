# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGEquiDiff weight download and loading utilities."""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Optional

import paddle
import requests

from ppmat.models.sgequidiff.constants import PRETRAINED_WEIGHT_URLS
from ppmat.utils import logger

SGEQUIDIFF_WEIGHTS_HOME = osp.join(
    osp.expanduser("~/.paddlemat/weights"), "sgequidiff"
)
DOWNLOAD_RETRY_LIMIT = 3


def download_weight_file(url: str, dataset_name: str, sub_module: str) -> str:
    """Download a single weight file to local cache."""
    cache_dir = osp.join(SGEQUIDIFF_WEIGHTS_HOME, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    fname = url.rstrip("/").split("/")[-1]
    local_path = osp.join(cache_dir, fname)

    if osp.exists(local_path):
        logger.info(f"Cached: {local_path}")
        return local_path

    retry_cnt = 0
    while retry_cnt < DOWNLOAD_RETRY_LIMIT:
        try:
            logger.info(f"Downloading {dataset_name}/{sub_module} from {url}")
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            logger.info(f"Done: {local_path}")
            return local_path
        except requests.RequestException as e:
            retry_cnt += 1
            if retry_cnt >= DOWNLOAD_RETRY_LIMIT:
                raise RuntimeError(
                    f"Download failed (retried {DOWNLOAD_RETRY_LIMIT} times): {url}\nError: {e}"
                )
            logger.warning(f"Retry {retry_cnt}/{DOWNLOAD_RETRY_LIMIT} ...")

    raise RuntimeError(f"Download failed: {url}")


# 4 sub-modules (diffusion/lattice/space_group/wyckoff), not single-file like framework load_pretrain
def download_all_weights(
    dataset_name: str = "mp_20",
) -> Dict[str, str]:
    """Download all sub-module weights for a given dataset."""
    if dataset_name not in PRETRAINED_WEIGHT_URLS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}, supported: {list(PRETRAINED_WEIGHT_URLS.keys())}"
        )

    urls = PRETRAINED_WEIGHT_URLS[dataset_name]
    local_paths = {}
    for sub_module, url in urls.items():
        local_paths[sub_module] = download_weight_file(url, dataset_name, sub_module)

    return local_paths


def load_pretrained_weights(
    model: paddle.nn.Layer,
    dataset_name: str = "mp_20",
    weight_dir: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Load pretrained weights into a model."""
    if weight_dir is not None:
        if not osp.isdir(weight_dir):
            raise FileNotFoundError(f"Weight directory does not exist: {weight_dir}")
        local_paths = {}
        urls = PRETRAINED_WEIGHT_URLS.get(dataset_name, {})
        for sub_module_key, url in urls.items():
            fname = url.rstrip("/").split("/")[-1]
            candidate = osp.join(weight_dir, fname)
            if osp.exists(candidate):
                local_paths[sub_module_key] = candidate
        if len(local_paths) == 0:
            for f in os.listdir(weight_dir):
                if f.endswith(".pdparams") and dataset_name in f:
                    for key in ["diffusion", "lattice", "space_group", "wyckoff"]:
                        if key in f:
                            local_paths[key] = osp.join(weight_dir, f)
    else:
        local_paths = download_all_weights(dataset_name)

    if verbose:
        logger.info(f"Dataset: {dataset_name}, found {len(local_paths)} weight files")

    submodule_attr_map = {
        "diffusion": "atom_coord_diffusion_model",
        "lattice": "lattice_sampler",
        "space_group": "space_group_sampler",
        "wyckoff": "wyckoff_and_element_sampler",
    }

    loaded_count = 0
    search_roots = [model]
    if hasattr(model, "diffusion_model"):
        search_roots.append(model.diffusion_model)

    for sub_module_key, local_path in local_paths.items():
        attr_name = submodule_attr_map.get(sub_module_key)
        if attr_name is None:
            if verbose:
                logger.warning(f"Unknown sub-module: {sub_module_key}")
            continue

        submodule = None
        for root in search_roots:
            submodule = getattr(root, attr_name, None)
            if submodule is not None:
                break

        if submodule is None:
            if verbose:
                logger.warning(f"Attribute {attr_name} not found in model")
            continue

        try:
            state_dict = paddle.load(local_path)
            submodule.set_state_dict(state_dict)
            loaded_count += 1
            if verbose:
                n_params = len(state_dict)
                logger.info(f"Loaded {sub_module_key:12s} -> {attr_name:30s} ({n_params} params)")
        except Exception as e:
            if verbose:
                logger.warning(f"Failed {sub_module_key}: {e}")

    if verbose:
        logger.info(f"Successfully loaded {loaded_count}/{len(local_paths)} weight files")

    return loaded_count
