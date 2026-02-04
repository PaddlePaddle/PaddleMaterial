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

"""
SFIN: Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement
Paper: CVPR 2025 - https://arxiv.org/pdf/2504.02555
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Dict, Optional


class FourierUnit(nn.Layer):
    """Fourier Unit for processing frequency domain features."""

    def __init__(self, in_channels: int, out_channels: int):
        super(FourierUnit, self).__init__()
        self.conv_layer = nn.Conv2D(
            in_channels=in_channels * 2 + 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False
        )
        self.bn = nn.BatchNorm2D(out_channels * 2, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        
        # Real FFT
        ffted = paddle.fft.rfftn(x, axes=fft_dim, norm='ortho')
        
        # Stack real and imaginary parts
        ffted_real = paddle.real(ffted)
        ffted_imag = paddle.imag(ffted)
        ffted = paddle.stack([ffted_real, ffted_imag], axis=-1)
        
        # Permute to (batch, c, 2, h, w/2+1)
        ffted = ffted.transpose([0, 1, 4, 2, 3])
        ffted = ffted.reshape([batch, -1] + list(ffted.shape[3:]))
        
        # Create coordinate grids
        height, width = ffted.shape[-2:]
        coords_vert = paddle.linspace(0, 1, height).reshape([1, 1, height, 1])
        coords_vert = coords_vert.expand([batch, 1, height, width])
        coords_hor = paddle.linspace(0, 1, width).reshape([1, 1, 1, width])
        coords_hor = coords_hor.expand([batch, 1, height, width])
        
        # Concatenate coordinates and FFT features
        ffted = paddle.concat([coords_vert, coords_hor, ffted], axis=1)
        
        # Process through convolution
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        # Reshape back to complex format
        ffted = ffted.reshape([batch, -1, 2] + list(ffted.shape[2:]))
        ffted = ffted.transpose([0, 1, 3, 4, 2])
        
        # Convert back to complex tensor
        ffted_real = ffted[..., 0]
        ffted_imag = ffted[..., 1]
        ffted = paddle.complex(ffted_real, ffted_imag)
        
        # Inverse FFT
        ifft_shape_slice = x.shape[-2:]
        output = paddle.fft.irfftn(ffted, s=ifft_shape_slice, axes=fft_dim, norm='ortho')
        
        return output


class SpectralTransform(nn.Layer):
    """Spectral Transform block combining spatial and frequency domain processing."""

    def __init__(self, in_channels: int):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2D(in_channels // 2, in_channels // 2, 3, padding=1)
        self.fu = FourierUnit(in_channels // 2, in_channels // 2)
        self.conv2 = nn.Conv2D(in_channels, in_channels // 2, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(paddle.concat([x, x2], axis=1))
        return x


class FFC(nn.Layer):
    """Fast Fourier Convolution block for spatial-frequency interaction."""

    def __init__(self, in_channels: int):
        super(FFC, self).__init__()
        self.convl2l = nn.Conv2D(in_channels // 2, in_channels // 2, 3, padding=1)
        self.convl2g = nn.Conv2D(in_channels // 2, in_channels // 2, 3, padding=1)
        self.convg2l = nn.Conv2D(in_channels // 2, in_channels // 2, 3, padding=1)
        self.convg2g = SpectralTransform(in_channels)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, paddle.zeros_like(x))
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class SFIB(nn.Layer):
    """Spatial-Frequency Interactive Block."""

    def __init__(self, in_channels: int):
        super(SFIB, self).__init__()
        self.ffc = FFC(in_channels)
        self.bn_l = nn.BatchNorm2D(in_channels // 2, momentum=0.1)
        self.bn_g = nn.BatchNorm2D(in_channels // 2, momentum=0.1)
        self.act_l = nn.ReLU()
        self.act_g = nn.ReLU()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class ResnetBlock(nn.Layer):
    """Residual block with SFIB."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = SFIB(in_channels)
        self.conv2 = SFIB(in_channels)

    def forward(self, x):
        x_l, x_g = paddle.split(x, [self.in_channels // 2, self.in_channels // 2], axis=1)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = paddle.concat([x_l, x_g], axis=1)
        return out


class SFIN(nn.Layer):
    """
    SFIN: Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale images)
        base_channels (int): Base number of channels (default: 64)
        num_blocks (int): Number of ResNet blocks (default: 8)
    
    Reference:
        Li et al., "Noise Calibration and Spatial-Frequency Interactive Network for 
        STEM Image Enhancement", CVPR 2025.
        https://arxiv.org/pdf/2504.02555
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 8
    ):
        super(SFIN, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        # Build ResNet blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResnetBlock(base_channels))
        self.body = nn.Sequential(*blocks)

        # Head and tail convolutions
        self.head_conv = nn.Conv2D(in_channels, base_channels, 3, padding=1)
        self.tail_conv = nn.Conv2D(base_channels, in_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming Uniform."""
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                paddle.nn.initializer.KaimingUniform(
                    negative_slope=5**0.5,  # a=sqrt(5)
                    nonlinearity='leaky_relu'
                )(m.weight)
                if m.bias is not None:
                    # Bias initialization
                    fan_in = m._in_channels * m._kernel_size[0] * m._kernel_size[1]
                    bound = 1.0 / (fan_in ** 0.5)
                    paddle.nn.initializer.Uniform(-bound, bound)(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                # BatchNorm initialization
                paddle.nn.initializer.Constant(value=1.0)(m.weight)
                paddle.nn.initializer.Constant(value=0.0)(m.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass of SFIN.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Enhanced image tensor of shape (B, C, H, W)
        """
        x = self.head_conv(x)
        shortcut = x
        x = self.body(x)
        x = x + shortcut
        x = self.tail_conv(x)
        return x

    def predict(self, batch: Dict) -> Dict:
        """
        Prediction interface for BasePredictor.
        
        Args:
            batch: Dictionary containing 'image' key with input tensor
        
        Returns:
            Dictionary containing 'pred' key with enhanced image
        """
        if isinstance(batch, dict):
            x = batch.get('image', batch.get('noisy', None))
        else:
            x = batch
        
        enhanced = self.forward(x)
        
        if isinstance(batch, dict):
            return {'pred': enhanced}
        else:
            return enhanced