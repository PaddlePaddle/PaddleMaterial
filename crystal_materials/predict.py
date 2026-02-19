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
from omegaconf import OmegaConf
from PIL import Image

from ppmat.models import build_model
from ppmat.utils import save_load


def load_gray_tensor(path: Path) -> paddle.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return paddle.to_tensor(arr).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="./crystal_materials/configs/sfin/sfin_tem_enhance.yaml",
        help="Path to model config yaml.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint (*.pdparams).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../sfin/data_test/noisy",
        help="Directory of noisy input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/sfin_tem_enhance/predictions",
        help="Directory to save enhanced images.",
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=".png",
        help="File suffix for image scanning.",
    )
    parser.add_argument(
        "--data_count",
        type=int,
        default=-1,
        help="Max number of images to process, -1 means all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu" if paddle.device.cuda.device_count() > 0 else "cpu",
        choices=["cpu", "gpu"],
        help="Device to run inference.",
    )
    args = parser.parse_args()

    paddle.set_device(args.device)

    config = OmegaConf.load(args.config_path)
    config = OmegaConf.to_container(config, resolve=True)
    model_cfg = config["Model"]
    model = build_model(model_cfg)
    try:
        # Compatible with checkpoints saved as:
        # 1) pure state_dict
        # 2) {"model_state_dict": ..., ...}
        ckpt = paddle.load(args.checkpoint_path)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.set_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.set_state_dict(ckpt)
        else:
            save_load.load_pretrain(model, args.checkpoint_path)
    except Exception:
        # Fallback to standard PaddleMaterials loading behavior
        save_load.load_pretrain(model, args.checkpoint_path)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.glob(f"*{args.file_suffix}") if p.is_file()])
    if args.data_count > 0:
        image_paths = image_paths[: args.data_count]

    with paddle.no_grad():
        for image_path in image_paths:
            x = load_gray_tensor(image_path)
            out = model(x)
            out = paddle.clip(out, min=0.0, max=255.0)
            out_np = out.squeeze().numpy().astype(np.uint8)
            Image.fromarray(out_np).save(output_dir / image_path.name)

    print(f"Saved {len(image_paths)} enhanced images to {output_dir}")
