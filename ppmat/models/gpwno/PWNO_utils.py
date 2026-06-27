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

import numpy as np
import paddle


class SpectralConv3d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights2 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights3 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights4 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )

    def compl_mul3d(self, input, weights):
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Pass explicit dimensions for the 3D real FFT.
        x_ft = paddle.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        out_ft = paddle.zeros(
            [
                batchsize,
                self.out_channels,
                x.shape[-3],
                x.shape[-2],
                x.shape[-1] // 2 + 1,
            ],
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )
        # Use the original spatial shape for the inverse 3D real FFT.
        x = paddle.fft.irfftn(
            out_ft,
            s=(x.shape[-3], x.shape[-2], x.shape[-1]),
            dim=[-3, -2, -1],
            norm="ortho",
        )
        return x


class SpectralConv3d_FFNO(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_FFNO, self).__init__()
        """
        3D Fourier layer for FFNO-style factorized spectral convolution.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes1
        self.modes_y = modes2
        self.modes_z = modes3
        self.fourier_weight = paddle.nn.ParameterList(parameters=[])
        for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
            weight = paddle.randn([in_channels, out_channels, n_modes, 2], dtype=paddle.float32)
            paddle.nn.init.xavier_normal_(weight)
            self.fourier_weight.append(paddle.nn.Parameter(weight))

    def forward(self, x):
        B, I, S1, S2, S3 = x.shape  # [batch, in_ch, x, y, z]

        # Spectral convolution along the z axis.
        x_ftz = paddle.fft.rfftn(x, dim=[-1,], norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        out_ft[:, :, :, :, : self.modes_z] = paddle.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, : self.modes_z],
            paddle.view_as_complex(self.fourier_weight[2])
        )
        xz = paddle.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")

        # Spectral convolution along the y axis.
        x_fty = paddle.fft.rfftn(x, dim=[-2,], norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        out_ft[:, :, :, : self.modes_y, :] = paddle.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, : self.modes_y, :],
            paddle.view_as_complex(self.fourier_weight[1])
        )
        xy = paddle.fft.irfft(out_ft, n=S2, dim=-2, norm="ortho")

        # Spectral convolution along the x axis.
        x_ftx = paddle.fft.rfftn(x, dim=[-3,], norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        out_ft[:, :, : self.modes_x, :, :] = paddle.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, : self.modes_x, :, :],
            paddle.view_as_complex(self.fourier_weight[0])
        )
        xx = paddle.fft.irfft(out_ft, n=S1, dim=-3, norm="ortho")

        x = xx + xy + xz
        return x


def plane_wave_sum(x):
    N = x.shape[-1]
    n = paddle.arange(N).float().to(x.device)
    k = n.unsqueeze(1)
    M = paddle.exp(-2.0j * np.pi * k * n)
    return paddle.matmul(M, x)
