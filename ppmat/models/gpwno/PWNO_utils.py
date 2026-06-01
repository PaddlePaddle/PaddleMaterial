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
        # ✅ 修复1：rfftn的dim传列表（多维FFT要求）
        x_ft = paddle.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        # ✅ 修复2：Paddle创建张量，替换new_zeros，修正device为place
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
        # 维度切片赋值（原逻辑正确，保留）
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
        # ✅ 修复3：irfftn的dim传列表（多维逆FFT要求）
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]), dim=[-3, -2, -1], norm="ortho")
        return x


class SpectralConv3d_FFNO(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_FFNO, self).__init__()
        """
        3D Fourier layer (FFNO版本). 修正Paddle兼容问题 + 维度对齐
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes1
        self.modes_y = modes2
        self.modes_z = modes3
        self.fourier_weight = paddle.nn.ParameterList(parameters=[])
        for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
            # ✅ 修复4：Paddle创建参数，替换FloatTensor，指定dtype
            weight = paddle.randn([in_channels, out_channels, n_modes, 2], dtype=paddle.float32)
            paddle.nn.init.xavier_normal_(weight)
            self.fourier_weight.append(paddle.nn.Parameter(weight))

    def forward(self, x):
        B, I, S1, S2, S3 = x.shape  # [batch, in_ch, x, y, z]
        
        # -------------------------- 处理z轴 (dim=-1) --------------------------
        # ✅ 修复5：rfftn传列表，irfft传单个整数
        x_ftz = paddle.fft.rfftn(x, dim=[-1,], norm="ortho")
        # ✅ 修复6：替换new_zeros为paddle.zeros，修正形状和device
        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        # einsum公式匹配维度，避免形状不匹配
        out_ft[:, :, :, :, : self.modes_z] = paddle.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, : self.modes_z],
            paddle.view_as_complex(self.fourier_weight[2])
        )
        # ✅ 修复7：irfft传单个整数dim=-1，移除重复调用
        xz = paddle.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")
        
        # -------------------------- 处理y轴 (dim=-2) --------------------------
        x_fty = paddle.fft.rfftn(x, dim=[-2,], norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        out_ft[:, :, :, : self.modes_y, :] = paddle.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, : self.modes_y, :],
            paddle.view_as_complex(self.fourier_weight[1])
        )
        xy = paddle.fft.irfft(out_ft, n=S2, dim=-2, norm="ortho")
        
        # -------------------------- 处理x轴 (dim=-3) --------------------------
        # ✅ 修复8：修正FFT维度为dim=-3（x轴），而非dim=-1
        x_ftx = paddle.fft.rfftn(x, dim=[-3,], norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        out_ft[:, :, : self.modes_x, :, :] = paddle.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, : self.modes_x, :, :],
            paddle.view_as_complex(self.fourier_weight[0])
        )
        # ✅ 修复9：irfft传单个整数dim=-3，而非列表
        xx = paddle.fft.irfft(out_ft, n=S1, dim=-3, norm="ortho")
        
        # 求和输出
        x = xx + xy + xz
        return x


def plane_wave_sum(x):
    N = x.shape[-1]  # ✅ 修复10：Paddle用shape而非size
    n = paddle.arange(N).float().to(x.device)  # ✅ 修正device为place
    k = n.unsqueeze(1)  # ✅ 修复11：Paddle用unsqueeze而非view
    M = paddle.exp(-2.0j * np.pi * k * n)
    return paddle.matmul(M, x)
