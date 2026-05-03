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

# Prepare CrystalLLM deliverables for Baidu Netdisk upload.
# Downloads all 11 pretrained checkpoints from Zenodo, converts to Paddle,
# and organizes into an upload-ready directory structure.
#
# Usage:
#   bash tools/prepare_netdisk.sh [--output-dir DIR] [--skip-download]
#
# Requirements: wget, tar, python3 with torch and paddle installed

set -euo pipefail

ZENODO_BASE="https://zenodo.org/records/10642388/files"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/netdisk_upload"
SKIP_DOWNLOAD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# All 11 checkpoints from Zenodo record 10642388
# NOTE: Zenodo filenames use underscores (perov_5, carbon_24, mp_20, mpts_52)
CHECKPOINTS=(
    "crystallm_v1_small"
    "crystallm_v1_large"
    "crystallm_v1_minus_mpts_52_small"
    "crystallm_perov_5_small"
    "crystallm_perov_5_large"
    "crystallm_carbon_24_small"
    "crystallm_carbon_24_large"
    "crystallm_mp_20_small"
    "crystallm_mp_20_large"
    "crystallm_mpts_52_small"
    "crystallm_mpts_52_large"
)

echo "=== CrystalLLM Netdisk Preparation ==="
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"/{pytorch_checkpoints,paddle_checkpoints,eval_results}

# Step 1: Download PyTorch checkpoints from Zenodo
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "--- Step 1: Downloading checkpoints from Zenodo ---"
    for ckpt in "${CHECKPOINTS[@]}"; do
        tarfile="${ckpt}.tar.gz"
        dest="$OUTPUT_DIR/pytorch_checkpoints/$tarfile"
        if [ -f "$dest" ] && [ -s "$dest" ]; then
            echo "  [skip] $tarfile (already exists)"
        else
            echo "  [download] $tarfile ..."
            wget -O "$dest" "${ZENODO_BASE}/${tarfile}"
        fi
    done
    echo ""

    # Extract all tarballs
    echo "--- Extracting checkpoints ---"
    for ckpt in "${CHECKPOINTS[@]}"; do
        tarfile="$OUTPUT_DIR/pytorch_checkpoints/${ckpt}.tar.gz"
        dest_dir="$OUTPUT_DIR/pytorch_checkpoints/${ckpt}"
        if [ -d "$dest_dir" ] && [ -f "$dest_dir/ckpt.pt" ]; then
            echo "  [skip] $ckpt (already extracted)"
        else
            echo "  [extract] $ckpt ..."
            mkdir -p "$dest_dir"
            tar -xzf "$tarfile" -C "$dest_dir" --strip-components=1
        fi
    done
    echo ""
fi

# Step 2: Convert all checkpoints to Paddle format
echo "--- Step 2: Converting PyTorch → Paddle ---"
CONVERTER="$REPO_ROOT/structure_generation/convert_weights.py"

for ckpt in "${CHECKPOINTS[@]}"; do
    pt_file="$OUTPUT_DIR/pytorch_checkpoints/${ckpt}/ckpt.pt"
    pd_dir="$OUTPUT_DIR/paddle_checkpoints/${ckpt}"
    pd_file="$pd_dir/ckpt.pdparams"

    if [ -f "$pd_file" ]; then
        echo "  [skip] $ckpt (already converted)"
        continue
    fi

    if [ ! -f "$pt_file" ]; then
        echo "  [WARN] $ckpt: no ckpt.pt found, skipping"
        continue
    fi

    echo "  [convert] $ckpt ..."
    mkdir -p "$pd_dir"
    python3 "$CONVERTER" --input "$pt_file" --output "$pd_file"
done
echo ""

# Step 3: Copy eval results if available
echo "--- Step 3: Collecting evaluation results ---"
for f in "$REPO_ROOT"/eval_*.json "$REPO_ROOT"/eval_*.log; do
    if [ -f "$f" ]; then
        cp "$f" "$OUTPUT_DIR/eval_results/"
        echo "  [copy] $(basename "$f")"
    fi
done
echo ""

# Step 4: Print summary
echo "=== Upload Directory Structure ==="
echo ""
find "$OUTPUT_DIR" -type f | sort | while read -r f; do
    size=$(du -sh "$f" | cut -f1)
    echo "  $size  ${f#$OUTPUT_DIR/}"
done
echo ""

TOTAL=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "Total size: $TOTAL"
echo ""
echo "=== Ready for Baidu Netdisk Upload ==="
echo "Upload the entire '$OUTPUT_DIR' directory to 百度网盘."
echo "Share the link (with password) in the PR comments."
echo ""
echo "Recommended Netdisk path: /PaddleMaterials/CrystalLLM/"
