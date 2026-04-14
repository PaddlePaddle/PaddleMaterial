"""
Weighted Directed Message Passing Neural Network (wD-MPNN) for PaddleMaterials.

Ported from: https://github.com/Ramprasad-Group/polymer-chemprop

The model performs directed message passing on molecular graphs with optional
per-atom and per-bond weights (for polymer-aware predictions), followed by a
feed-forward network to produce property predictions.

Architecture:
    1. MPNEncoder – directed message passing with weighted edges
    2. FFN – feed-forward network for final prediction
"""

from typing import Dict, List, Optional, Tuple, Union
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.wd_mpnn.featurization import BatchMolGraph, MolGraph, index_select_ND


class MPNEncoder(nn.Layer):
    """
    Directed message passing encoder for molecular graphs.

    Implements the wD-MPNN message passing formula:
        m(a1→a2) = Σ_{a0∈nei(a1)} m(a0→a1) * w(a0→a1) − m(a2→a1)

    followed by atom-level readout with weighted aggregation.
    """

    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        aggregation: str = "mean",
        aggregation_norm: float = 100.0,
        bias: bool = True,
    ):
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm

        self.dropout_layer = nn.Dropout(p=dropout)
        self.act_func = nn.ReLU()

        # Cached zero vector for empty molecules
        self.register_buffer(
            name="cached_zero_vector",
            tensor=paddle.zeros([hidden_size]),
        )

        # Input projection: bond features → hidden
        self.W_i = nn.Linear(bond_fdim, hidden_size, bias_attr=bias)
        # Message update
        self.W_h = nn.Linear(hidden_size, hidden_size, bias_attr=bias)
        # Output projection: (atom features || hidden) → hidden
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size, bias_attr=True)

    def forward(
        self,
        f_atoms: paddle.Tensor,
        f_bonds: paddle.Tensor,
        w_atoms: paddle.Tensor,
        w_bonds: paddle.Tensor,
        a2b: paddle.Tensor,
        b2a: paddle.Tensor,
        b2revb: paddle.Tensor,
        a_scope: List[Tuple[int, int]],
        degree_of_polym: List[float],
    ) -> paddle.Tensor:
        """
        Encode a batched molecular graph.

        Args:
            f_atoms: (total_atoms, atom_fdim) atom feature matrix.
            f_bonds: (total_bonds, bond_fdim) bond feature matrix.
            w_atoms: (total_atoms,) per-atom weights.
            w_bonds: (total_bonds,) per-bond weights.
            a2b: (total_atoms, max_num_bonds) atom-to-bond adjacency.
            b2a: (total_bonds,) bond-to-source-atom mapping.
            b2revb: (total_bonds,) bond-to-reverse-bond mapping.
            a_scope: List of (start, size) tuples per molecule.
            degree_of_polym: Per-molecule degree of polymerization.

        Returns:
            Tensor of shape (num_molecules, hidden_size).
        """
        # Initial bond message
        inp = self.W_i(f_bonds)  # (n_bonds, hidden)
        message = self.act_func(inp)  # (n_bonds, hidden)

        # Message passing iterations
        for _ in range(self.depth - 1):
            # Gather neighbor messages per atom, weighted by bond weights
            nei_a_message = index_select_ND(message, a2b)  # (n_atoms, max_bonds, hidden)
            nei_a_weight = index_select_ND(w_bonds, a2b)  # (n_atoms, max_bonds)
            nei_a_message = nei_a_message * nei_a_weight.unsqueeze(-1)
            a_message = nei_a_message.sum(axis=1)  # (n_atoms, hidden)

            # Subtract reverse message
            rev_message = paddle.index_select(message, b2revb, axis=0)  # (n_bonds, hidden)
            message = paddle.index_select(a_message, b2a, axis=0) - rev_message

            message = self.W_h(message)
            message = self.act_func(inp + message)  # residual
            message = self.dropout_layer(message)

        # Final aggregation: atom hidden states
        nei_a_message = index_select_ND(message, a2b)
        nei_a_weight = index_select_ND(w_bonds, a2b)
        nei_a_message = nei_a_message * nei_a_weight.unsqueeze(-1)
        a_message = nei_a_message.sum(axis=1)  # (n_atoms, hidden)

        a_input = paddle.concat([f_atoms, a_message], axis=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Per-molecule readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = paddle.slice(
                    atom_hiddens, axes=[0], starts=[a_start], ends=[a_start + a_size]
                )
                w_atom_vec = paddle.slice(
                    w_atoms, axes=[0], starts=[a_start], ends=[a_start + a_size]
                )
                # Weight atom representations
                mol_vec = w_atom_vec.unsqueeze(-1) * cur_hiddens

                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(axis=0) / w_atom_vec.sum(axis=0)
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(axis=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(axis=0) / self.aggregation_norm

                # Scale by degree of polymerization (log-scaled per RFC)
                xn = degree_of_polym[i]
                mol_vec = (1.0 + math.log(max(xn, 1.0))) * mol_vec
                mol_vecs.append(mol_vec)

        mol_vecs = paddle.stack(mol_vecs, axis=0)  # (n_mols, hidden)
        return mol_vecs


class WDMPNN(nn.Layer):
    """
    Weighted Directed Message Passing Neural Network.

    Combines an MPNEncoder for molecular graph encoding with a feed-forward
    network for property prediction. Follows PaddleMaterials model conventions:
    ``forward()`` returns ``{"loss_dict": {...}, "pred_dict": {...}}``.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        ffn_hidden_size: int = 300,
        ffn_num_layers: int = 2,
        aggregation: str = "mean",
        aggregation_norm: float = 100.0,
        property_names: Union[str, List[str]] = "property",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "mse_loss",
        atom_fdim: int = 133,
        bond_fdim: int = 14,
        bias: bool = True,
        output_size: int = 1,
    ):
        """
        Args:
            hidden_size: Hidden dimension for message passing.
            depth: Number of message passing iterations.
            dropout: Dropout probability.
            ffn_hidden_size: Hidden dimension for FFN layers.
            ffn_num_layers: Number of FFN layers (including output layer).
            aggregation: Readout aggregation type ('mean', 'sum', 'norm').
            aggregation_norm: Normalization constant for 'norm' aggregation.
            property_names: Name(s) of the target property.
            data_mean: Mean for output normalization.
            data_std: Std for output normalization.
            loss_type: Loss function ('mse_loss' or 'l1_loss').
            atom_fdim: Atom feature dimension.
            bond_fdim: Bond feature dimension.
            bias: Whether to use bias in linear layers.
            output_size: Number of output targets.
        """
        super().__init__()

        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            self.property_names = property_names

        self.hidden_size = hidden_size
        self.output_size = output_size

        # Normalization buffers
        self.register_buffer(
            name="data_mean", tensor=paddle.to_tensor(data_mean, dtype="float32")
        )
        self.register_buffer(
            name="data_std", tensor=paddle.to_tensor(data_std, dtype="float32")
        )

        # Loss function
        if loss_type == "mse_loss":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Encoder
        self.encoder = MPNEncoder(
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim,
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            bias=bias,
        )

        # Feed-forward network
        self.ffn = self._build_ffn(
            first_linear_dim=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
            output_size=output_size,
            dropout=dropout,
        )

    @staticmethod
    def _build_ffn(
        first_linear_dim: int,
        ffn_hidden_size: int,
        ffn_num_layers: int,
        output_size: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build the feed-forward network."""
        dropout_layer = nn.Dropout(p=dropout)
        activation = nn.ReLU()

        if ffn_num_layers == 1:
            layers = [dropout_layer, nn.Linear(first_linear_dim, output_size)]
        else:
            layers = [dropout_layer, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                layers.extend([activation, dropout_layer, nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            layers.extend([activation, dropout_layer, nn.Linear(ffn_hidden_size, output_size)])

        return nn.Sequential(*layers)

    def normalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return tensor * self.data_std + self.data_mean

    def _forward(self, data: Dict) -> paddle.Tensor:
        """
        Core forward computation.

        Args:
            data: Dict containing either a ``BatchMolGraph`` under key ``"mol_graph"``
                  or pre-computed graph components (``f_atoms``, ``f_bonds``, etc.).

        Returns:
            Raw predictions of shape (batch_size, output_size).
        """
        if "mol_graph" in data:
            mol_graph: BatchMolGraph = data["mol_graph"]
            (
                f_atoms, f_bonds, w_atoms, w_bonds,
                a2b, b2a, b2revb,
                a_scope, _b_scope, degree_of_polym,
            ) = mol_graph.get_components()
        else:
            f_atoms = data["f_atoms"]
            f_bonds = data["f_bonds"]
            w_atoms = data["w_atoms"]
            w_bonds = data["w_bonds"]
            a2b = data["a2b"]
            b2a = data["b2a"]
            b2revb = data["b2revb"]
            a_scope = data["a_scope"]
            degree_of_polym = data.get("degree_of_polym", [1.0] * len(a_scope))

        encoding = self.encoder(
            f_atoms, f_bonds, w_atoms, w_bonds,
            a2b, b2a, b2revb, a_scope, degree_of_polym,
        )
        output = self.ffn(encoding)
        return output

    def forward(
        self,
        data: Dict,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> Dict:
        """
        Full forward pass with optional loss and prediction.

        Args:
            data: Input data dict with graph components and optionally labels.
            return_loss: Whether to compute and return the loss.
            return_prediction: Whether to return unnormalized predictions.

        Returns:
            Dict with ``"loss_dict"`` and ``"pred_dict"`` entries.
        """
        assert return_loss or return_prediction, (
            "At least one of return_loss or return_prediction must be True."
        )
        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(input=pred, label=label)
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            pred = self.unnormalize(pred)
            prediction[self.property_names] = pred

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, data: Dict) -> Dict:
        """
        Run inference and return unnormalized predictions.

        Args:
            data: Input data dict with graph components.

        Returns:
            Dict mapping property name to predicted value.
        """
        pred = self._forward(data)
        pred = self.unnormalize(pred)
        return {self.property_names: pred}
