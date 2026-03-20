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
SolvGNN model for predicting binary activity coefficients.

This module implements the SolvGNN model which uses graph neural networks
to predict activity coefficients for binary solvent mixtures, incorporating
Gibbs-Duhem thermodynamic constraints.
"""

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.losses.gibbs_duhem_loss import GibbsDuhemLoss
from ppmat.models.gdinn.utils.graph_utils import mean_nodes
from ppmat.models.gdinn.utils.layers import GraphConv
from ppmat.models.gdinn.utils.layers import MPNNConv
from ppmat.models.gdinn.utils.layers import get_activation


class SolvGNN(nn.Layer):
    """SolvGNN model for predicting binary activity coefficients.

    This model takes two molecular graphs (for the two solvents in a binary mixture)
    and predicts their activity coefficients (gamma1, gamma2). It enforces the
    Gibbs-Duhem thermodynamic constraint:
        x1*d(ln(gamma1))/dx1 + x2*d(ln(gamma2))/dx1 = 0.

    Model architecture:
        1. Two separate graph convolutional branches for each solvent
        2. Graph-level pooling to get molecular embeddings
        3. Global interaction layer between the two embeddings
        4. MLP classifier to predict gamma1 and gamma2
        5. Gibbs-Duhem constraint loss computation

    Args:
        in_dim: Input node feature dimension (default: 74 for atom features)
        hidden_dim: Hidden dimension for graph layers (default: 256)
        n_classes: Number of output classes (default: 1 for gamma)
        mlp_dropout_rate: Dropout rate for MLP layers (default: 0.0)
        mlp_activation: Activation function for MLP (default: None)
        mpnn_activation: Activation function for MPNN layers (default: None)
        num_step_message_passing: Number of message passing steps (default: 1)
        pinn_lambda: Weight for Gibbs-Duhem constraint loss (default: 1.0)
    """

    def __init__(
        self,
        in_dim: int = 74,
        hidden_dim: int = 256,
        n_classes: int = 1,
        mlp_dropout_rate: float = 0.0,
        mlp_activation: Optional[str] = None,
        mpnn_activation: Optional[str] = None,
        num_step_message_passing: int = 1,
        pinn_lambda: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Graph convolutional layers (shared between two solvents)
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # Global MPNN convolution layer for interaction
        # Input dimension is hidden_dim + 1 (for composition information)
        self.global_conv = MPNNConv(
            node_in_feats=hidden_dim + 1,
            edge_in_feats=1,
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=num_step_message_passing,
            activation=mpnn_activation,
        )

        # MLP classifier (shared for both solvents)
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

        # Gibbs-Duhem loss function
        self.gd_loss_fn = GibbsDuhemLoss(
            lambda_gd=pinn_lambda, loss_type="mse", create_graph=True
        )

    def forward(self, batch_data: Dict) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass of SolvGNN model.

        Args:
            batch_data: Dictionary containing:
                - g1: First molecular graph (solvent 1)
                - g2: Second molecular graph (solvent 2)
                - x1: Composition of solvent 1 (mole fraction) aka `solv1_x` [batch_size, 1]
                - gamma1: Target activity coefficient for solvent 1 [batch_size, 1]
                - gamma2: Target activity coefficient for solvent 2 [batch_size, 1]
                - intra_hb1: Intra-molecular hydrogen bonds in solvent 1 [batch_size, 1]
                - intra_hb2: Intra-molecular hydrogen bonds in solvent 2 [batch_size, 1]
                - inter_hb: Inter-molecular hydrogen bonds [batch_size, 1]
                - empty_solvsys: Empty solvent system graph for global interaction.
                    Must be provided by BinaryActivityCollator.

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
        g1 = batch_data["g1"]
        g2 = batch_data["g2"]

        h1 = g1.node_feat["h"].cast("float32")
        h2 = g2.node_feat["h"].cast("float32")

        solv1_x = batch_data["x1"]
        solv1_x.stop_gradient = False

        h1 = F.relu(self.conv1(g1, h1))
        h1 = F.relu(self.conv2(g1, h1))
        h2 = F.relu(self.conv1(g2, h2))
        h2 = F.relu(self.conv2(g2, h2))
        g1.node_feat["h"] = h1
        g2.node_feat["h"] = h2

        hg1 = mean_nodes(g1, "h")  # [batch_size, hidden_dim]
        hg2 = mean_nodes(g2, "h")  # [batch_size, hidden_dim]

        hg1 = paddle.concat(
            [hg1, solv1_x.unsqueeze(-1)], axis=1
        )  # [batch_size, hidden_dim + 1]
        hg2 = paddle.concat(
            [hg2, (1 - solv1_x).unsqueeze(-1)], axis=1
        )  # [batch_size, hidden_dim + 1]

        empty_solvsys = batch_data["empty_solvsys"]

        # Create hydrogen bond edge features
        # All hb tensors are 1D [batch_size] in original
        inter_hb = batch_data["inter_hb"].cast("float32")  # [batch_size]
        intra_hb1 = batch_data["intra_hb1"].cast("float32")  # [batch_size]
        intra_hb2 = batch_data["intra_hb2"].cast("float32")  # [batch_size]
        # repeat(2) on 1D tensor in PyTorch doubles it: [batch] -> [2*batch]
        hb_features = paddle.concat(
            [paddle.tile(inter_hb, [2]), intra_hb1, intra_hb2]
        ).unsqueeze(
            1
        )  # [4 * batch_size, 1]

        # Concatenate both molecule embeddings for global convolution
        hg_concat = paddle.concat(
            [hg1, hg2], axis=0
        )  # [2 * batch_size, hidden_dim + 1]

        # Apply global MPNN convolution for molecular interaction
        hg = self.global_conv(
            empty_solvsys, hg_concat, hb_features
        )  # [2 * batch_size, hidden_dim]

        # Predict ln_gamma using shared classifier
        output = self.mlp_activation(self.classify1(hg))
        output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)  # [2 * batch_size, n_classes]

        # Split predictions back into two molecules
        half = output.shape[0] // 2
        output = paddle.concat(
            [output[:half, :], output[half:, :]], axis=1
        )  # [batch_size, 2 * n_classes]

        # Split into ln_gamma1 and ln_gamma2
        ln_gamma1_pred = output[:, : self.n_classes]  # [batch_size, 1]
        ln_gamma2_pred = output[:, self.n_classes :]  # [batch_size, 1]

        # Convert to gamma (gamma = exp(ln(gamma)))
        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)

        # Compute prediction loss
        gamma1_label = batch_data["gamma1"]
        gamma2_label = batch_data["gamma2"]

        # Labels (gamma1_label, gamma2_label) are already ln(gamma) values from the dataset
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

    def predict(self, g1, g2, x1: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """Predict activity coefficients for a binary mixture.

        This method is for inference only and does not compute losses.

        Args:
            g1: First molecular graph (solvent 1)
            g2: Second molecular graph (solvent 2)
            x1: Composition of solvent 1 [batch_size, 1]

        Returns:
            Dictionary containing:
                - gamma1: Predicted activity coefficient for solvent 1
                - gamma2: Predicted activity coefficient for solvent 2
        """
        batch_data = {
            "g1": g1,
            "g2": g2,
            "x1": x1,
            "gamma1": paddle.zeros_like(x1),  # Dummy label
            "gamma2": paddle.zeros_like(x1),  # Dummy label
        }

        output = self.forward(batch_data)
        return output["pred_dict"]


class SolvGNNxMLP(nn.Layer):
    """SolvGNN with MLP that includes composition in MLP input.

    This variant differs from the base SolvGNN in how composition is incorporated:
    - Base SolvGNN: Composition is concatenated with molecular embeddings BEFORE global_conv
    - SolvGNNxMLP: Composition is concatenated with global_conv output BEFORE MLP

    This allows the MLP to directly learn composition-dependent transformations.

    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden dimension for graph layers
        n_classes: Number of output classes (default: 1 for gamma)
        mlp_dropout_rate: Dropout rate for MLP layers
        mlp_activation: Activation function for MLP
        mpnn_activation: Activation function for MPNN layers
        num_step_message_passing: Number of message passing steps
        mlp_num_hid_layers: Number of hidden layers in MLP (1 or 2)
        pinn_lambda: Weight for Gibbs-Duhem constraint loss
    """

    def __init__(
        self,
        in_dim: int = 74,
        hidden_dim: int = 256,
        n_classes: int = 1,
        mlp_dropout_rate: float = 0.0,
        mlp_activation: Optional[str] = None,
        mpnn_activation: Optional[str] = None,
        num_step_message_passing: int = 1,
        mlp_num_hid_layers: int = 2,
        pinn_lambda: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.mlp_num_hid_layers = mlp_num_hid_layers

        # Graph convolutional layers (shared between two solvents)
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # Global MPNN convolution layer for interaction
        # Note: Input dimension is hidden_dim (NOT +1 like base SolvGNN)
        self.global_conv = MPNNConv(
            node_in_feats=hidden_dim,
            edge_in_feats=1,
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=num_step_message_passing,
            activation=mpnn_activation,
        )

        # MLP classifier
        # Input dimension is hidden_dim + 1 (composition added AFTER global_conv)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = nn.Linear(hidden_dim + 1, hidden_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError("mlp_num_hid_layers must be 1 or 2")
        self.classify3 = nn.Linear(hidden_dim, n_classes)

        # Gibbs-Duhem loss function
        self.gd_loss_fn = GibbsDuhemLoss(
            lambda_gd=pinn_lambda, loss_type="mse", create_graph=True
        )

    def forward(self, batch_data: Dict) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass of SolvGNNxMLP model."""
        g1 = batch_data["g1"]
        g2 = batch_data["g2"]

        # Get composition
        solv1_x = batch_data["x1"]
        solv1_x.stop_gradient = False

        h1 = g1.node_feat["h"].cast("float32")
        h2 = g2.node_feat["h"].cast("float32")

        h1 = F.relu(self.conv1(g1, h1))
        h1 = F.relu(self.conv2(g1, h1))
        h2 = F.relu(self.conv1(g2, h2))
        h2 = F.relu(self.conv2(g2, h2))
        g1.node_feat["h"] = h1
        g2.node_feat["h"] = h2

        hg1 = mean_nodes(g1, "h")
        hg2 = mean_nodes(g2, "h")

        empty_solvsys = batch_data["empty_solvsys"]

        # Create hydrogen bond edge features
        inter_hb = batch_data["inter_hb"].cast("float32")
        intra_hb1 = batch_data["intra_hb1"].cast("float32")
        intra_hb2 = batch_data["intra_hb2"].cast("float32")
        hb_features = paddle.concat(
            [paddle.tile(inter_hb, [2]), intra_hb1, intra_hb2]
        ).unsqueeze(1)

        # Concatenate both molecule embeddings for global convolution
        # Note: NO composition concatenation here (unlike base SolvGNN)
        hg_concat = paddle.concat([hg1, hg2], axis=0)

        # Apply global MPNN convolution
        hg = self.global_conv(empty_solvsys, hg_concat, hb_features)

        # Concatenate composition AFTER global_conv (key difference from base SolvGNN)
        hg = paddle.concat(
            [hg, paddle.concat([solv1_x, 1 - solv1_x]).unsqueeze(-1)], axis=1
        )

        # MLP classifier
        output = self.mlp_dropout(hg)
        output = self.mlp_activation(self.classify1(output))
        if self.mlp_num_hid_layers == 2:
            output = self.mlp_dropout(output)
            output = self.mlp_activation(self.classify2(output))
        output = self.classify3(output)

        # Split predictions
        half = output.shape[0] // 2
        output = paddle.concat([output[:half, :], output[half:, :]], axis=1)

        ln_gamma1_pred = output[:, : self.n_classes]
        ln_gamma2_pred = output[:, self.n_classes :]

        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)

        # Compute losses
        gamma1_label = batch_data["gamma1"]
        gamma2_label = batch_data["gamma2"]

        pred_loss = 0.5 * F.mse_loss(
            ln_gamma1_pred.squeeze(-1), gamma1_label.squeeze(-1)
        ) + 0.5 * F.mse_loss(ln_gamma2_pred.squeeze(-1), gamma2_label.squeeze(-1))

        gd_loss = self.gd_loss_fn(ln_gamma1_pred, ln_gamma2_pred, solv1_x)

        total_loss = pred_loss + gd_loss

        loss_dict = {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "gd_loss": gd_loss,
        }

        pred_dict = {
            "gamma1": gamma1_pred,
            "gamma2": gamma2_pred,
            "ln_gamma1": ln_gamma1_pred,
            "ln_gamma2": ln_gamma2_pred,
        }

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}


class GEGNN(nn.Layer):
    """GEGNN: Gibbs Excess Energy Graph Neural Network.

    This model predicts a shared Gibbs Excess Energy (G^E) and derives activity
    coefficients from it using thermodynamic relationships:
        gamma_1 = G^E + (1-x1) * d(G^E)/dx1
        gamma_2 = G^E - x1 * d(G^E)/dx1

    This formulation automatically satisfies the Gibbs-Duhem constraint by construction.

    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden dimension for graph layers
        n_classes: Number of output classes (default: 1 for G^E)
        mlp_dropout_rate: Dropout rate for MLP layers
        mlp_activation: Activation function for MLP
        mpnn_activation: Activation function for MPNN layers
        num_step_message_passing: Number of message passing steps
        pinn_lambda: Weight for Gibbs-Duhem constraint loss
    """

    def __init__(
        self,
        in_dim: int = 74,
        hidden_dim: int = 256,
        n_classes: int = 1,
        mlp_dropout_rate: float = 0.0,
        mlp_activation: Optional[str] = None,
        mpnn_activation: Optional[str] = None,
        num_step_message_passing: int = 1,
        pinn_lambda: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Graph convolutional layers
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # Global MPNN convolution layer
        self.global_conv = MPNNConv(
            node_in_feats=hidden_dim,
            edge_in_feats=1,
            node_out_feats=hidden_dim,
            edge_hidden_feats=32,
            num_step_message_passing=num_step_message_passing,
            activation=mpnn_activation,
        )

        # SLP (Solvation Layer Perceptron) for transforming embeddings with composition
        self.mlp_activation = get_activation(mlp_activation)
        self.mfp_trans = nn.Linear(hidden_dim + 1, hidden_dim + 1)

        # MLP classifier for G^E prediction
        self.classify1 = nn.Linear(hidden_dim + 1, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

        # Gibbs-Duhem loss function
        self.gd_loss_fn = GibbsDuhemLoss(
            lambda_gd=pinn_lambda, loss_type="mse", create_graph=True
        )

    def forward(self, batch_data: Dict) -> Dict[str, Dict[str, paddle.Tensor]]:
        """Forward pass of GEGNN model."""
        g1 = batch_data["g1"]
        g2 = batch_data["g2"]

        # Get composition
        solv1_x = batch_data["x1"]
        solv1_x.stop_gradient = False

        h1 = g1.node_feat["h"].cast("float32")
        h2 = g2.node_feat["h"].cast("float32")

        h1 = F.relu(self.conv1(g1, h1))
        h1 = F.relu(self.conv2(g1, h1))
        h2 = F.relu(self.conv1(g2, h2))
        h2 = F.relu(self.conv2(g2, h2))
        g1.node_feat["h"] = h1
        g2.node_feat["h"] = h2

        hg1 = mean_nodes(g1, "h")
        hg2 = mean_nodes(g2, "h")

        empty_solvsys = batch_data["empty_solvsys"]

        # Create hydrogen bond edge features
        inter_hb = batch_data["inter_hb"].cast("float32").flatten()
        intra_hb1 = batch_data["intra_hb1"].cast("float32").flatten()
        intra_hb2 = batch_data["intra_hb2"].cast("float32").flatten()
        hb_features = paddle.concat(
            [paddle.tile(inter_hb, [2]), intra_hb1, intra_hb2]
        ).unsqueeze(1)

        # Concatenate both molecule embeddings for global convolution
        hg_concat = paddle.concat([hg1, hg2], axis=0)

        # Apply global MPNN convolution for molecular interaction
        hg = self.global_conv(empty_solvsys, hg_concat, hb_features)

        # Split back into two molecules
        half = hg.shape[0] // 2
        hg1 = hg[:half, :]
        hg2 = hg[half:, :]

        # SLP: Transform embeddings with composition
        hg1_temp = self.mlp_activation(
            self.mfp_trans(paddle.concat([hg1, solv1_x.unsqueeze(-1)], axis=1))
        )
        hg2_temp = self.mlp_activation(
            self.mfp_trans(paddle.concat([hg2, (1 - solv1_x).unsqueeze(-1)], axis=1))
        )

        # Pooling: Average the two transformed embeddings
        hg_temp = (hg1_temp + hg2_temp) / 2

        # MLP to predict G^E (Gibbs Excess Energy)
        output = self.mlp_activation(self.classify1(hg_temp))
        output = self.mlp_activation(self.classify2(output))
        G_E = self.classify3(output)  # [batch_size, 1]

        # Derive activity coefficients from G^E using thermodynamic relationships
        G_dx1 = paddle.grad(
            outputs=G_E.sum(),
            inputs=solv1_x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if G_dx1 is None:
            G_dx1 = paddle.zeros_like(solv1_x)

        # gamma_1 = G^E + (1-x1) * d(G^E)/dx1
        # gamma_2 = G^E - x1 * d(G^E)/dx1
        # Note: Original code computes ln(gamma), not gamma directly
        ln_gamma1_pred = G_E.squeeze(-1) + (1 - solv1_x) * G_dx1
        ln_gamma2_pred = G_E.squeeze(-1) - solv1_x * G_dx1

        # Reshape to [batch_size, 1]
        ln_gamma1_pred = ln_gamma1_pred.unsqueeze(-1)
        ln_gamma2_pred = ln_gamma2_pred.unsqueeze(-1)

        gamma1_pred = paddle.exp(ln_gamma1_pred)
        gamma2_pred = paddle.exp(ln_gamma2_pred)

        # Compute prediction loss
        gamma1_label = batch_data["gamma1"]
        gamma2_label = batch_data["gamma2"]

        pred_loss = 0.5 * F.mse_loss(
            ln_gamma1_pred.squeeze(-1), gamma1_label.squeeze(-1)
        ) + 0.5 * F.mse_loss(ln_gamma2_pred.squeeze(-1), gamma2_label.squeeze(-1))

        # Compute Gibbs-Duhem constraint loss (should be ~0 by construction)
        gd_loss = self.gd_loss_fn(ln_gamma1_pred, ln_gamma2_pred, solv1_x)

        total_loss = pred_loss + gd_loss

        loss_dict = {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "gd_loss": gd_loss,
        }

        pred_dict = {
            "gamma1": gamma1_pred,
            "gamma2": gamma2_pred,
            "ln_gamma1": ln_gamma1_pred,
            "ln_gamma2": ln_gamma2_pred,
            "G_E": G_E,
        }

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(self, g1, g2, x1: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """Predict activity coefficients for a binary mixture."""
        batch_data = {
            "g1": g1,
            "g2": g2,
            "x1": x1,
            "gamma1": paddle.zeros_like(x1),
            "gamma2": paddle.zeros_like(x1),
        }
        output = self.forward(batch_data)
        return output["pred_dict"]
