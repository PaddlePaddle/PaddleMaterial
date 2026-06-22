#!/usr/bin/env bash
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -euo pipefail

DATA_ROOT="${1:-./data}"
PYTHON_BIN="${PYTHON_BIN:-}"
UMA_DIRECT_DOWNLOAD="${UMA_DIRECT_DOWNLOAD:-1}"

if [[ "${UMA_DIRECT_DOWNLOAD}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY
  unset http_proxy https_proxy all_proxy ftp_proxy
fi

find_python() {
  local candidates=()
  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=("python3")
  candidates+=("python")

  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" -c "import ase, lmdb" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      return
    fi
  done

  echo "Cannot find a Python environment satisfying PaddleMaterials requirements.txt; set PYTHON_BIN." >&2
  exit 1
}

find_python

mkdir -p "${DATA_ROOT}/omat24/raw" "${DATA_ROOT}/omat24/train" "${DATA_ROOT}/omat24/val"
mkdir -p "${DATA_ROOT}/oc20/raw"

download() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    if tar -tf "${output}" >/dev/null 2>&1; then
      echo "Found complete ${output}, skip download."
      return
    fi
    echo "Found incomplete ${output}, resume download."
  fi
  curl -L -C - "${url}" -o "${output}"
}

extract_once() {
  local archive="$1"
  local destination="$2"
  local marker="$3"
  if [[ -f "${marker}" ]]; then
    echo "Found ${marker}, skip extraction."
    return
  fi
  mkdir -p "${destination}"
  tar -xf "${archive}" -C "${destination}"
  touch "${marker}"
}

download \
  "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz" \
  "${DATA_ROOT}/omat24/raw/rattled-500-train.tar.gz"
download \
  "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz" \
  "${DATA_ROOT}/omat24/raw/rattled-500-val.tar.gz"

extract_once \
  "${DATA_ROOT}/omat24/raw/rattled-500-train.tar.gz" \
  "${DATA_ROOT}/omat24/train" \
  "${DATA_ROOT}/omat24/train/.rattled-500.extracted"
extract_once \
  "${DATA_ROOT}/omat24/raw/rattled-500-val.tar.gz" \
  "${DATA_ROOT}/omat24/val" \
  "${DATA_ROOT}/omat24/val/.rattled-500.extracted"

download \
  "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar" \
  "${DATA_ROOT}/oc20/raw/s2ef_train_200K.tar"
download \
  "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar" \
  "${DATA_ROOT}/oc20/raw/s2ef_val_id.tar"

extract_once \
  "${DATA_ROOT}/oc20/raw/s2ef_train_200K.tar" \
  "${DATA_ROOT}/oc20/raw" \
  "${DATA_ROOT}/oc20/raw/.s2ef_train_200K.extracted"
extract_once \
  "${DATA_ROOT}/oc20/raw/s2ef_val_id.tar" \
  "${DATA_ROOT}/oc20/raw" \
  "${DATA_ROOT}/oc20/raw/.s2ef_val_id.extracted"

"${PYTHON_BIN}" test/uma/prepare_oc20_s2ef_aselmdb.py \
  --raw-dir "${DATA_ROOT}/oc20/raw/s2ef_train_200K/s2ef_train_200K" \
  --val-raw-dir "${DATA_ROOT}/oc20/raw/s2ef_val_id/s2ef_val_id" \
  --out-dir "${DATA_ROOT}/oc20/uma_budget_aselmdb" \
  --train 50000 \
  --val 5000 \
  --test 5000

echo "Budget UMA data is ready under ${DATA_ROOT}/omat24 and ${DATA_ROOT}/oc20/uma_budget_aselmdb."
