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
"""
AlloyGAN — GAN and CGAN for inverse metallic glass design.

Implements the generative models from:
    "Inverse Materials Design by Large Language Model-Assisted
     Generative Framework" (Hao et al., arXiv:2502.18127)

Architecture:
    GAN:  G(z[100]) → comp[40],  D(comp[40]) → real/fake
    CGAN: G(z[5]||cond[26]) → comp[40],  D(comp[40]||cond[26]) → real/fake
"""

import paddle
import paddle.nn as nn


class AlloyGenerator(nn.Layer):
    """Generator network for AlloyGAN.

    Produces 40-dimensional alloy compositions from latent noise.
    """

    def __init__(self, input_dim=100, hidden_dim=512, output_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class AlloyDiscriminator(nn.Layer):
    """Discriminator network for AlloyGAN.

    Classifies 40-dimensional (or 66-dimensional for CGAN) inputs as real/fake.
    Output is probability via Sigmoid (clamped for numerical stability).
    """

    def __init__(self, input_dim=40, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AlloyGAN(nn.Layer):
    """Standard GAN for alloy composition generation.

    Generator:     z(100) → composition(40)
    Discriminator: composition(40) → real/fake probability

    The model wraps both G and D as sublayers for checkpoint management.
    Training uses a custom adversarial loop (not BaseTrainer).
    """

    def __init__(
        self,
        noise_dim=100,
        hidden_dim_g=512,
        hidden_dim_d=1024,
        comp_dim=40,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.comp_dim = comp_dim

        self.generator = AlloyGenerator(
            input_dim=noise_dim,
            hidden_dim=hidden_dim_g,
            output_dim=comp_dim,
        )
        self.discriminator = AlloyDiscriminator(
            input_dim=comp_dim,
            hidden_dim=hidden_dim_d,
        )

    def forward(self, batch_data):
        """Forward pass for compatibility with PaddleMaterials.

        For GAN training, use train_step_d / train_step_g instead.
        This forward method wraps a single G step for inference or
        combined loss reporting.
        """
        if isinstance(batch_data, dict):
            real_data = batch_data["data"]  # (B, 66)
            real_comp = real_data[:, :self.comp_dim]
        else:
            real_comp = batch_data[:, :self.comp_dim]

        batch_size = real_comp.shape[0]
        z = paddle.rand([batch_size, self.noise_dim])
        fake_comp = self.generator(z)

        # D on real
        d_real = self.discriminator(real_comp)
        # D on fake (detach G)
        d_fake = self.discriminator(fake_comp.detach())

        ones = paddle.ones_like(d_real)
        zeros = paddle.zeros_like(d_fake)

        loss_fn = nn.BCELoss()
        d_loss = loss_fn(d_real, ones) + loss_fn(d_fake, zeros)

        # G loss
        d_fake_for_g = self.discriminator(fake_comp)
        g_loss = loss_fn(d_fake_for_g, ones)

        return {
            "loss_dict": {
                "loss": d_loss + g_loss,
                "d_loss": d_loss,
                "g_loss": g_loss,
            },
            "pred_dict": {
                "fake_comp": fake_comp,
            },
        }

    def generate(self, num_samples=100):
        """Generate alloy compositions from random noise."""
        z = paddle.rand([num_samples, self.noise_dim])
        return self.generator(z)


class AlloyCGAN(nn.Layer):
    """Conditional GAN for alloy composition generation.

    Generator:     (z[5] || conditions[26]) → composition[40]
    Discriminator: (composition[40] || conditions[26]) → real/fake probability

    Conditions are 26-dimensional: Tg, Tx, Tl + 23 GFA criteria.
    """

    def __init__(
        self,
        noise_dim=5,
        cond_dim=26,
        hidden_dim_g=512,
        hidden_dim_d=1024,
        comp_dim=40,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.comp_dim = comp_dim

        self.generator = AlloyGenerator(
            input_dim=noise_dim + cond_dim,  # 5 + 26 = 31
            hidden_dim=hidden_dim_g,
            output_dim=comp_dim,
        )
        self.discriminator = AlloyDiscriminator(
            input_dim=comp_dim + cond_dim,  # 40 + 26 = 66
            hidden_dim=hidden_dim_d,
        )

    def forward(self, batch_data):
        """Forward pass for compatibility with PaddleMaterials.

        batch_data: dict with "data" key → tensor of shape (B, 66)
            columns 0-39: composition
            columns 40-65: conditions (Tg, Tx, Tl + 23 GFA)
        """
        if isinstance(batch_data, dict):
            data = batch_data["data"]  # (B, 66)
        else:
            data = batch_data

        real_comp = data[:, :self.comp_dim]     # (B, 40)
        conditions = data[:, self.comp_dim:]    # (B, 26)

        batch_size = real_comp.shape[0]
        z = paddle.rand([batch_size, self.noise_dim])

        # G: generate fake composition conditioned on properties
        g_input = paddle.concat([z, conditions], axis=1)  # (B, 31)
        fake_comp = self.generator(g_input)                # (B, 40)

        # D on real
        d_real_input = paddle.concat([real_comp, conditions], axis=1)  # (B, 66)
        d_real = self.discriminator(d_real_input)

        # D on fake (detach G)
        d_fake_input = paddle.concat([fake_comp.detach(), conditions], axis=1)
        d_fake = self.discriminator(d_fake_input)

        ones = paddle.ones_like(d_real)
        zeros = paddle.zeros_like(d_fake)

        loss_fn = nn.BCELoss()
        d_loss = loss_fn(d_real, ones) + loss_fn(d_fake, zeros)

        # G loss
        d_fake_for_g = self.discriminator(
            paddle.concat([fake_comp, conditions], axis=1)
        )
        g_loss = loss_fn(d_fake_for_g, ones)

        return {
            "loss_dict": {
                "loss": d_loss + g_loss,
                "d_loss": d_loss,
                "g_loss": g_loss,
            },
            "pred_dict": {
                "fake_comp": fake_comp,
            },
        }

    def generate(self, conditions, num_samples=None):
        """Generate alloy compositions conditioned on target properties.

        Args:
            conditions: (N, 26) tensor of target properties
            num_samples: if provided, randomly sample this many conditions

        Returns:
            (N, 40) tensor of generated alloy compositions
        """
        if num_samples is not None and num_samples < conditions.shape[0]:
            idx = paddle.randperm(conditions.shape[0])[:num_samples]
            conditions = conditions[idx]

        z = paddle.randn([conditions.shape[0], self.noise_dim])
        g_input = paddle.concat([z, conditions], axis=1)
        return self.generator(g_input)
