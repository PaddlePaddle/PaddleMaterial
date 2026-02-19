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

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import paddle
from PIL import Image


class STEMImageDataset(paddle.io.Dataset):
    """Dataset for paired STEM image restoration/enhancement.

    Expected directory layout:
        data_path/
          noisy/
            0.png
            1.png
            ...
          gt_enhance/
            0.png
            1.png
            ...
    """

    def __init__(
        self,
        data_path: str,
        data_count: int | None = None,
        noisy_subdir: str = "noisy",
        target_subdir: str = "gt_enhance",
        file_suffix: str = ".png",
        strict_index_naming: bool = True,
        scale_to_unit: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_path)
        self.noisy_dir = self.data_root / noisy_subdir
        self.target_dir = self.data_root / target_subdir
        self.file_suffix = file_suffix
        self.strict_index_naming = strict_index_naming
        self.scale_to_unit = scale_to_unit

        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        self.samples = self._build_samples(data_count)

    def _build_samples(self, data_count: int | None) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []

        if self.strict_index_naming:
            if data_count is None:
                available = sorted(self.noisy_dir.glob(f"*{self.file_suffix}"))
                data_count = len(available)

            for idx in range(int(data_count)):
                name = f"{idx}{self.file_suffix}"
                noisy_path = self.noisy_dir / name
                target_path = self.target_dir / name
                if not noisy_path.exists():
                    raise FileNotFoundError(f"Noisy image not found: {noisy_path}")
                if not target_path.exists():
                    raise FileNotFoundError(f"Target image not found: {target_path}")
                samples.append(
                    {
                        "name": name,
                        "noisy_path": str(noisy_path),
                        "target_path": str(target_path),
                    }
                )
            return samples

        noisy_files = {
            p.name: p for p in self.noisy_dir.glob(f"*{self.file_suffix}") if p.is_file()
        }
        target_files = {
            p.name: p for p in self.target_dir.glob(f"*{self.file_suffix}") if p.is_file()
        }
        common_names = sorted(set(noisy_files.keys()) & set(target_files.keys()))
        if data_count is not None:
            common_names = common_names[: int(data_count)]

        for name in common_names:
            samples.append(
                {
                    "name": name,
                    "noisy_path": str(noisy_files[name]),
                    "target_path": str(target_files[name]),
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_gray_image(self, path: str) -> paddle.Tensor:
        image = Image.open(path).convert("L")
        image_array = np.asarray(image, dtype=np.float32)
        if self.scale_to_unit:
            image_array = image_array / 255.0
        return paddle.to_tensor(image_array).unsqueeze(0)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        noisy = self._load_gray_image(sample["noisy_path"])
        target = self._load_gray_image(sample["target_path"])

        return {
            "noisy": noisy,
            "gt_enhance": target,
            "name": sample["name"],
        }
