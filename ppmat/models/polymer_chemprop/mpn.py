from functools import reduce
from typing import List

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.polymer_chemprop.featurization import BatchMolGraph
from ppmat.models.polymer_chemprop.featurization import Featurization_parameters
from ppmat.models.polymer_chemprop.featurization import get_atom_fdim
from ppmat.models.polymer_chemprop.featurization import get_bond_fdim
from ppmat.models.polymer_chemprop.nn_utils import get_activation_function
from ppmat.models.polymer_chemprop.nn_utils import index_select_ND


class MPNEncoder(nn.Layer):
    """An MPNEncoder is a message passing neural network for encoding a molecule."""

    def __init__(
        self,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        atom_messages: bool = False,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        activation: str = "ReLU",
        atom_descriptors: str = None,
        atom_descriptors_size: int = 0,
    ):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.layers_per_message = 1
        self.undirected = undirected
        self.aggregation = aggregation
        self.aggregation_norm = aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(activation)

        # Cached zeros
        self.register_buffer("cached_zero_vector", paddle.zeros([self.hidden_size]))

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias_attr=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias_attr=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if atom_descriptors == "descriptor":
            self.atom_descriptors_size = atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(
                self.hidden_size + self.atom_descriptors_size,
                self.hidden_size + self.atom_descriptors_size,
            )

        self.atom_descriptors = atom_descriptors

    def forward(
        self, mol_graph: BatchMolGraph, atom_descriptors_batch: List[np.ndarray] = None
    ) -> paddle.Tensor:
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [
                np.zeros([1, atom_descriptors_batch[0].shape[1]])
            ] + atom_descriptors_batch
            atom_descriptors_batch = paddle.to_tensor(
                np.concatenate(atom_descriptors_batch, axis=0), dtype="float32"
            )

        (
            f_atoms,
            f_bonds,
            w_atoms,
            w_bonds,
            a2b,
            b2a,
            b2revb,
            a_scope,
            b_scope,
            degree_of_polym,
        ) = mol_graph.get_components(atom_messages=self.atom_messages)

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = paddle.concat((nei_a_message, nei_f_bonds), axis=2)
                message = nei_message.sum(axis=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                nei_a_weight = index_select_ND(w_bonds, a2b)
                nei_a_message = nei_a_message * nei_a_weight[..., None]
                a_message = nei_a_message.sum(axis=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        nei_a_weight = index_select_ND(w_bonds, a2x)
        nei_a_message = nei_a_message * nei_a_weight[..., None]
        a_message = nei_a_message.sum(axis=1)
        a_input = paddle.concat([f_atoms, a_message], axis=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Concatenate atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(
                    "The number of atoms is different from the length of the extra atom features"
                )
            atom_hiddens = paddle.concat([atom_hiddens, atom_descriptors_batch], axis=1)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens[a_start : a_start + a_size]
                mol_vec = cur_hiddens
                w_atom_vec = w_atoms[a_start : a_start + a_size]
                mol_vec = w_atom_vec[..., None] * mol_vec
                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(axis=0) / w_atom_vec.sum(axis=0)
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(axis=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(axis=0) / self.aggregation_norm

                mol_vec = degree_of_polym[i] * mol_vec
                mol_vecs.append(mol_vec)

        mol_vecs = paddle.stack(mol_vecs, axis=0)
        return mol_vecs


class MPN(nn.Layer):
    """An MPN is a wrapper around MPNEncoder which featurizes input as needed."""

    def __init__(
        self,
        atom_fdim: int = None,
        bond_fdim: int = None,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        atom_messages: bool = False,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        activation: str = "ReLU",
        features_only: bool = False,
        use_input_features: bool = False,
        atom_descriptors: str = None,
        atom_descriptors_size: int = 0,
        number_of_molecules: int = 1,
        mpn_shared: bool = False,
        featurization_config: Featurization_parameters = None,
    ):
        super(MPN, self).__init__()

        if featurization_config is None:
            featurization_config = Featurization_parameters()
        self.featurization_config = featurization_config

        self.atom_fdim = atom_fdim or get_atom_fdim(config=featurization_config)
        self.bond_fdim = bond_fdim or get_bond_fdim(
            config=featurization_config, atom_messages=atom_messages
        )

        self.features_only = features_only
        self.use_input_features = use_input_features
        self.atom_descriptors = atom_descriptors

        if self.features_only:
            return

        encoder_kwargs = dict(
            atom_fdim=self.atom_fdim,
            bond_fdim=self.bond_fdim,
            hidden_size=hidden_size,
            bias=bias,
            depth=depth,
            dropout=dropout,
            undirected=undirected,
            atom_messages=atom_messages,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            activation=activation,
            atom_descriptors=atom_descriptors,
            atom_descriptors_size=atom_descriptors_size,
        )

        if mpn_shared:
            shared_encoder = MPNEncoder(**encoder_kwargs)
            self.encoder = nn.LayerList([shared_encoder] * number_of_molecules)
        else:
            self.encoder = nn.LayerList(
                [MPNEncoder(**encoder_kwargs) for _ in range(number_of_molecules)]
            )

    def forward(
        self,
        batch: List[BatchMolGraph],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
    ) -> paddle.Tensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of BatchMolGraph (one per molecule component).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: A paddle tensor of shape (num_molecules, hidden_size) containing the encoding.
        """
        if self.use_input_features:
            features_batch = paddle.to_tensor(np.stack(features_batch), dtype="float32")

            if self.features_only:
                return features_batch

        if self.atom_descriptors == "descriptor":
            if len(batch) > 1:
                raise NotImplementedError(
                    "Atom descriptors are currently only supported with one molecule "
                    "per input (i.e., number_of_molecules = 1)."
                )
            encodings = [
                enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)
            ]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: paddle.concat((x, y), axis=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.reshape([1, -1])
            output = paddle.concat([output, features_batch], axis=1)

        return output
