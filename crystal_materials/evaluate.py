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

import argparse
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

from ppmat.metrics import PSNRMetric
from ppmat.metrics import SSIMMetric


def load_gray_tensor(path: Path) -> paddle.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return paddle.to_tensor(arr).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="../sfin/data_test/gt_enhance",
        help="Ground truth image directory.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Predicted image directory.",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=".png",
        help="Image file suffix.",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    gt_files = sorted([p for p in gt_dir.glob(f"*{args.file_suffix}") if p.is_file()])

    psnr_metric = PSNRMetric(data_range=255.0)
    ssim_metric = SSIMMetric(data_range=255.0)

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = load_gray_tensor(gt_path)
        pred = load_gray_tensor(pred_path)
        psnr_sum += float(psnr_metric(pred, gt))
        ssim_sum += float(ssim_metric(pred, gt))
        count += 1

    if count == 0:
        raise RuntimeError("No matched prediction files found for evaluation.")

    print(f"Matched images: {count}")
    print(f"Average PSNR: {psnr_sum / count:.6f}")
    print(f"Average SSIM: {ssim_sum / count:.6f}")
