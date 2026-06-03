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
Loss functions for GDI-NN with Gibbs-Duhem constraints.

This module provides loss functions that combine prediction loss with
Gibbs-Duhem thermodynamic constraint enforcement.
"""


import paddle
import paddle.nn as nn


class GibbsDuhemLoss(nn.Layer):
    """Gibbs-Duhem constraint loss.

    This loss enforces the Gibbs-Duhem thermodynamic constraint for binary mixtures:
    x1 * d(ln(gamma1))/dx1 + x2 * d(ln(gamma2))/dx1 = 0

    This ensures thermodynamic consistency of the predicted activity coefficients.

    Args:
        lambda_gd: Weight for Gibbs-Duhem loss (default: 1.0)
        loss_type: Type of loss to use ('mse', 'mae', 'huber') (default: 'mse')
        create_graph: If True, enables higher-order gradients (default: True)
                     Set to False when using dropout to avoid gradient issues
    """

    def __init__(
        self, lambda_gd: float = 1.0, loss_type: str = "mse", create_graph: bool = True
    ):
        super().__init__()
        self.lambda_gd = lambda_gd
        self.loss_type = loss_type
        self.create_graph = create_graph

    def forward(
        self,
        ln_gamma1: paddle.Tensor,
        ln_gamma2: paddle.Tensor,
        x1: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute Gibbs-Duhem constraint loss.

        Args:
            ln_gamma1: Predicted ln(gamma1) [batch_size, 1]
            ln_gamma2: Predicted ln(gamma2) [batch_size, 1]
            x1: Composition of solvent 1 [batch_size, 1]

        Returns:
            Gibbs-Duhem constraint loss (scalar)
        """
        x1.stop_gradient = False

        # Compute d(ln(gamma1))/dx1
        dln_gamma1_dx1 = paddle.grad(
            outputs=ln_gamma1,
            inputs=x1,
            create_graph=self.create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]

        # Compute d(ln(gamma2))/dx1
        dln_gamma2_dx1 = paddle.grad(
            outputs=ln_gamma2,
            inputs=x1,
            create_graph=self.create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]

        # Handle None gradients (when prediction mode or no gradient flow)
        if dln_gamma1_dx1 is None:
            dln_gamma1_dx1 = paddle.zeros_like(x1)
        if dln_gamma2_dx1 is None:
            dln_gamma2_dx1 = paddle.zeros_like(x1)

        # Gibbs-Duhem constraint: x1*dln_gamma1/dx1 + (1-x1)*dln_gamma2/dx1 = 0
        gd_constraint = x1 * dln_gamma1_dx1 + (1 - x1) * dln_gamma2_dx1

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = paddle.mean(gd_constraint**2)
        elif self.loss_type == "mae":
            loss = paddle.mean(paddle.abs(gd_constraint))
        elif self.loss_type == "huber":
            loss = paddle.nn.functional.smooth_l1_loss(
                gd_constraint, paddle.zeros_like(gd_constraint)
            )
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        loss = self.lambda_gd * loss

        return loss
