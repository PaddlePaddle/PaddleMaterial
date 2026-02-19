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

import paddle


def calc_psnr(
    pred: paddle.Tensor,
    label: paddle.Tensor,
    data_range: float = 255.0,
    eps: float = 1e-12,
) -> paddle.Tensor:
    """Compute batch PSNR for image tensors with shape [N, C, H, W]."""
    pred = pred.astype("float64")
    label = label.astype("float64")
    diff = (pred - label) / data_range
    mse = paddle.mean(diff * diff)
    mse = paddle.maximum(mse, paddle.to_tensor(eps, dtype=mse.dtype))
    return -10.0 * paddle.log10(mse)


def _gaussian_window_2d(
    channels: int,
    win_size: int = 11,
    win_sigma: float = 1.5,
    dtype: str = "float32",
) -> paddle.Tensor:
    coords = paddle.arange(win_size, dtype=dtype) - (win_size // 2)
    gauss = paddle.exp(-(coords**2) / (2.0 * (win_sigma**2)))
    gauss = gauss / paddle.sum(gauss)
    window_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = window_2d.reshape([1, 1, win_size, win_size])
    return paddle.tile(window, [channels, 1, 1, 1])


def calc_ssim(
    pred: paddle.Tensor,
    label: paddle.Tensor,
    data_range: float = 255.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    nonnegative_ssim: bool = False,
) -> paddle.Tensor:
    """Compute batch SSIM for image tensors with shape [N, C, H, W]."""
    if pred.shape != label.shape:
        raise ValueError(
            f"Input images should have the same dimensions, got {pred.shape} and {label.shape}."
        )
    if len(pred.shape) != 4:
        raise ValueError(
            f"Input images should be 4-d tensors [N, C, H, W], got shape {pred.shape}."
        )
    if win_size % 2 != 1:
        raise ValueError("win_size must be odd.")

    pred = pred.astype("float32")
    label = label.astype("float32")

    channels = pred.shape[1]
    window = _gaussian_window_2d(
        channels=channels,
        win_size=win_size,
        win_sigma=win_sigma,
        dtype=pred.dtype,
    )

    mu1 = paddle.nn.functional.conv2d(pred, window, stride=1, padding=0, groups=channels)
    mu2 = paddle.nn.functional.conv2d(label, window, stride=1, padding=0, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = paddle.nn.functional.conv2d(
        pred * pred, window, stride=1, padding=0, groups=channels
    ) - mu1_sq
    sigma2_sq = paddle.nn.functional.conv2d(
        label * label, window, stride=1, padding=0, groups=channels
    ) - mu2_sq
    sigma12 = paddle.nn.functional.conv2d(
        pred * label, window, stride=1, padding=0, groups=channels
    ) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c1 = paddle.to_tensor(c1, dtype=pred.dtype)
    c2 = paddle.to_tensor(c2, dtype=pred.dtype)

    cs_map = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2.0 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if nonnegative_ssim:
        ssim_map = paddle.nn.functional.relu(ssim_map)

    return paddle.mean(ssim_map)


class PSNRMetric:
    def __init__(self, data_range: float = 255.0, eps: float = 1e-12):
        self.data_range = data_range
        self.eps = eps

    def __call__(self, pred: paddle.Tensor, label: paddle.Tensor):
        return calc_psnr(
            pred=pred,
            label=label,
            data_range=self.data_range,
            eps=self.eps,
        )


class SSIMMetric:
    def __init__(
        self,
        data_range: float = 255.0,
        win_size: int = 11,
        win_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        nonnegative_ssim: bool = False,
    ):
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.k1 = k1
        self.k2 = k2
        self.nonnegative_ssim = nonnegative_ssim

    def __call__(self, pred: paddle.Tensor, label: paddle.Tensor):
        return calc_ssim(
            pred=pred,
            label=label,
            data_range=self.data_range,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            k1=self.k1,
            k2=self.k2,
            nonnegative_ssim=self.nonnegative_ssim,
        )
