from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
from models import initializer
from models.layers import MLP
from models.layers import ActivationFunction
from models.layers import EdgeSet2Set
from models.layers import EmbeddingBlock
from models.layers import MEGNetBlock
from models.layers import Set2Set
from utils.default_elements import DEFAULT_ELEMENTS


class MEGNetPlus(paddle.nn.Layer):
    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: (int | None) = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: (BondExpansion | None) = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        pretrained=None,
        num_predictions: int = 1,
        **kwargs,
    ):
        """Useful defaults for all arguments have been specified based on MEGNet formation energy model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_state_embedding: Dimension of state embedding.
            ntypes_state: Number of state types.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_type: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_state_embedding: Architecture of embedding layer for state attributes
            include_state: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution. Defaults to 0, i.e., no dropout.
            element_types: Elements included in the training set
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()
        # self.save_args(locals(), kwargs)
        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion
        self.pretrained = pretrained
        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]
        try:
            activation: paddle.nn.Layer = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )
        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)
        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)
            for _ in range(nblocks - 1)
        ]
        self.blocks = paddle.nn.LayerList(sublayers=blocks)
        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)
        self.output_proj = MLP(
            dims=[
                2 * 2 * dim_blocks_out + dim_blocks_out,
                *hidden_layer_sizes_output,
                num_predictions,
            ],
            activation=activation,
            activate_last=False,
        )
        self.dropout = paddle.nn.Dropout(p=dropout) if dropout else None
        self.is_classification = is_classification
        self.include_state_embedding = include_state

        if self.pretrained:
            self.set_dict(paddle.load(self.pretrained))
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(
        self, g: dgl.DGLGraph, state_attr: (paddle.Tensor | None) = None, **kwargs
    ):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            g (dgl.DGLGraph): DGL graphs
            state_attr (paddle.Tensor): State attributes
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Prediction
        """
        node_attr = g.node_feat["node_type"]
        edge_attr = self.bond_expansion(g.edge_feat["bond_dist"])
        node_feat, edge_feat, state_feat = self.embedding(
            node_attr, edge_attr, state_attr
        )
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)
        for block in self.blocks:
            output = block(g, edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output
        node_vec = self.node_s2s(g, node_feat)
        edge_vec = self.edge_s2s(g, edge_feat)

        vec = paddle.concat([node_vec, edge_vec, state_feat], axis=1)
        if self.dropout:
            vec = self.dropout(vec)
        output = self.output_proj(vec)
        if self.is_classification:
            output = paddle.nn.functional.sigmoid(x=output)
        return paddle.squeeze(x=output)
