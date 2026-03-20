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
MCM (Multi-Component Model) for predicting binary activity coefficients.

This module implements the MCM model which uses embedding layers and MLPs
to predict activity coefficients for binary solvent mixtures, based on
solvent/solute IDs rather than molecular graphs.

Reference:
    Chen, G., Song, Z., Qi, Z., & Sundmacher, K. (2021). Neural recommender
    system for the activity coefficient prediction and UNIFAC model extension
    of ionic liquid‐solute systems. AIChE Journal, 67(4), e17171.
"""

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.losses.gibbs_duhem_loss import GibbsDuhemLoss
from ppmat.models.gdinn.utils.layers import get_activation


class MLPModule(nn.Layer):
    """MLP module with embedding layer for solvent/solute encoding.

    This module creates an embedding layer followed by multiple linear layers
    with ReLU activation and dropout.

    Args:
        dim_in: Input dimension (vocabulary size for embedding)
        dim_hidden: Hidden dimension
        dropout: Dropout rate
    """

    def __init__(self, dim_in: int, dim_hidden: int, dropout: float = 0.05):
        super().__init__()

        self.embedding = nn.Embedding(dim_in, dim_hidden)
        self.dropout = nn.Dropout(dropout)

        # Build MLP layers matching PyTorch get_mlp_module
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of indices [batch_size]

        Returns:
            Output tensor [batch_size, dim_hidden]
        """
        # Embedding
        x = self.embedding(x)  # [batch_size, dim_hidden]
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 1
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.linear3(x)
        x = F.relu(x)

        return x


class MCM_MultiMLP(nn.Layer):
    """MCM (Multi-Component Model) with multiple MLP branches.

    This model uses embedding layers to encode solvent and solute IDs,
    then concatenates them with composition information and passes through
    separate MLP branches to predict ln(gamma1) and ln(gamma2).

    Model architecture:
        1. Embedding layers for solvent and solute IDs
        2. Concatenate embeddings with composition (x1, 1-x1)
        3. Two separate MLP branches for gamma1 and gamma2 prediction
        4. Optional Gibbs-Duhem constraint loss computation

    Args:
        solvent_id_max: Maximum solvent ID (vocabulary size - 1)
        dim_hidden_channels: Hidden dimension for embeddings and MLPs (default: 128)
        dropout_hidden: Dropout rate for hidden layers (default: 0.05)
        dropout_interaction: Dropout rate for interaction layers (default: 0.03)
        mlp_activation: Activation function for MLP layers (default: "relu")
        mlp_num_hid_layers: Number of hidden layers in MLP (default: 1)
        pinn_lambda: Weight for Gibbs-Duhem constraint loss (default: 1.0)
    """

    def __init__(
        self,
        solvent_id_max: int,
        dim_hidden_channels: int = 128,
        dropout_hidden: float = 0.05,
        dropout_interaction: float = 0.03,
        mlp_activation: Optional[str] = None,
        mlp_num_hid_layers: int = 1,
        pinn_lambda: float = 1.0,
        **kwargs
    ):
        super().__init__()

        self.mlp_activation = get_activation(mlp_activation, get_nn=True)
        self.dropout_p1 = dropout_hidden
        self.dropout_p2 = dropout_interaction
        self.dim_hidden_channels = dim_hidden_channels

        # Embedding module for solvent and solute
        self.solvent_emb = MLPModule(
            dim_in=solvent_id_max + 1,
            dim_hidden=self.dim_hidden_channels,
            dropout=self.dropout_p1,
        )

        # Mid embedding dimension (concatenated solvent + solute)
        mid_emb = 2 * self.dim_hidden_channels

        # Build MLP layers for gamma1 prediction
        list_layers_end_1 = [nn.Linear(mid_emb + 2, mid_emb), self.mlp_activation()]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_1.append(nn.Linear(mid_emb, mid_emb))
                list_layers_end_1.append(self.mlp_activation())
        list_layers_end_1.append(nn.Linear(mid_emb, 1))

        # Build MLP layers for gamma2 prediction
        list_layers_end_2 = [nn.Linear(mid_emb + 2, mid_emb), self.mlp_activation()]
        if mlp_num_hid_layers > 1:
            for _ in range(mlp_num_hid_layers - 1):
                list_layers_end_2.append(nn.Linear(mid_emb, mid_emb))
                list_layers_end_2.append(self.mlp_activation())
        list_layers_end_2.append(nn.Linear(mid_emb, 1))

        # Create two separate MLP branches
        self.layers_end = nn.LayerList(
            [nn.Sequential(*list_layers_end_1), nn.Sequential(*list_layers_end_2)]
        )

        # Gibbs-Duhem loss function
        self.gd_loss_fn = GibbsDuhemLoss(
            lambda_gd=pinn_lambda, loss_type="mse", create_graph=False
        )

    def forward(self, batch_data: Dict) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass of MCM model.

        Args:
            batch_data: Dictionary containing:
                - solv1_id: Solvent 1 IDs [batch_size]
                - solv2_id: Solvent 2 IDs [batch_size]
                - x1: Composition of solvent 1 [batch_size]
                - gamma1: Target ln(gamma1) [batch_size, 1]
                - gamma2: Target ln(gamma2) [batch_size, 1]

        Returns:
            Dictionary containing:
                - loss_dict: Dictionary of losses
                    - loss: Total loss for training (MANDATORY)
                    - pred_loss: Prediction loss (MSE) for logging
                    - gd_loss: Gibbs-Duhem constraint loss for logging
                - pred_dict: Dictionary of predictions
                    - gamma1: Predicted gamma1
                    - gamma2: Predicted gamma2
                    - ln_gamma1: Predicted ln(gamma1)
                    - ln_gamma2: Predicted ln(gamma2)
        """
        # Get composition
        solv1_x = batch_data["x1"]
        solv1_x.stop_gradient = False

        # Get solvent and solute IDs
        solv1_id = batch_data["solv1_id"].cast("int64")
        solv2_id = batch_data["solv2_id"].cast("int64")

        # Embedding
        x_solvent = self.solvent_emb(solv1_id)  # [batch_size, dim_hidden]
        x_solute = self.solvent_emb(solv2_id)  # [batch_size, dim_hidden]

        # Concatenate embeddings with composition
        h = paddle.concat(
            [x_solvent, solv1_x.unsqueeze(-1), x_solute, (1 - solv1_x).unsqueeze(-1)],
            axis=1,
        ).cast(
            "float32"
        )  # [batch_size, 2*dim_hidden + 2]

        # Predict ln(gamma1) and ln(gamma2) using separate MLP branches
        output_y1 = self.layers_end[0](h)  # [batch_size, 1]
        output_y2 = self.layers_end[1](h)  # [batch_size, 1]

        # Concatenate outputs
        output = paddle.concat([output_y1, output_y2], axis=1)  # [batch_size, 2]

        # Split into ln_gamma1 and ln_gamma2
        ln_gamma1_pred = output[:, 0:1]  # [batch_size, 1]
        ln_gamma2_pred = output[:, 1:2]  # [batch_size, 1]

        # Convert to gamma
        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)

        # Compute prediction loss
        gamma1_label = batch_data["gamma1"]
        gamma2_label = batch_data["gamma2"]

        pred_loss = 0.5 * F.mse_loss(
            ln_gamma1_pred.squeeze(-1), gamma1_label.squeeze(-1)
        ) + 0.5 * F.mse_loss(ln_gamma2_pred.squeeze(-1), gamma2_label.squeeze(-1))

        # Compute Gibbs-Duhem constraint loss
        gd_loss = self.gd_loss_fn(ln_gamma1_pred, ln_gamma2_pred, solv1_x)

        total_loss = pred_loss + gd_loss

        loss_dict = {"loss": total_loss, "pred_loss": pred_loss, "gd_loss": gd_loss}

        pred_dict = {
            "gamma1": gamma1_pred,
            "gamma2": gamma2_pred,
            "ln_gamma1": ln_gamma1_pred,
            "ln_gamma2": ln_gamma2_pred,
        }

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(
        self, solv1_id: paddle.Tensor, solv2_id: paddle.Tensor, x1: paddle.Tensor
    ) -> Dict[str, paddle.Tensor]:
        """Predict activity coefficients for a binary mixture.

        This method is for inference only and does not compute losses.

        Args:
            solv1_id: Solvent 1 IDs [batch_size]
            solv2_id: Solvent 2 IDs [batch_size]
            x1: Composition of solvent 1 [batch_size]

        Returns:
            Dictionary containing:
                - gamma1: Predicted activity coefficient for solvent 1
                - gamma2: Predicted activity coefficient for solvent 2
        """
        batch_data = {
            "solv1_id": solv1_id,
            "solv2_id": solv2_id,
            "x1": x1,
            "gamma1": paddle.zeros_like(x1),
            "gamma2": paddle.zeros_like(x1),
        }

        output = self.forward(batch_data)
        return output["pred_dict"]
