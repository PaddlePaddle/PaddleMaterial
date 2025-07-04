from __future__ import annotations

from typing import Literal

import paddle
import paddle.nn as nn

from ppmat.models.common import initializer
from ppmat.models.megnet.bond import BondExpansion
from ppmat.models.megnet.layers import MLP
from ppmat.models.megnet.layers import ActivationFunction
from ppmat.models.megnet.layers import EdgeSet2Set
from ppmat.models.megnet.layers import EmbeddingBlock
from ppmat.models.megnet.layers import MEGNetBlock
from ppmat.models.megnet.layers import Set2Set
from ppmat.utils.default_elements import DEFAULT_ELEMENTS


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
        include_state: bool = True,
        dropout: float = 0.0,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion_cfg=None,
        cutoff: float = 4.0,
        property_names: Literal[
            "band_gap", "formation_energy_per_atom"
        ] = "formation_energy_per_atom",
    ):
        super().__init__()
        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = BondExpansion(**bond_expansion_cfg)

        self.property_names = (
            [property_names] if isinstance(property_names, str) else property_names
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]
        try:
            activation: paddle.nn.Layer = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                "Invalid activation type, please try using one of "
                f"{[af.name for af in ActivationFunction]}"
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

        self.heads = {}
        if "band_gap" in self.property_names:
            self.heads["band_gap"] = MLP(
                dims=[
                    2 * 2 * dim_blocks_out + dim_blocks_out,
                    *hidden_layer_sizes_output,
                    1,
                ],
                activation=activation,
                activate_last=False,
            )
        if "formation_energy_per_atom" in self.property_names:
            self.heads["formation_energy_per_atom"] = MLP(
                dims=[
                    2 * 2 * dim_blocks_out + dim_blocks_out,
                    *hidden_layer_sizes_output,
                    1,
                ],
                activation=activation,
                activate_last=False,
            )

        self.dropout = paddle.nn.Dropout(p=dropout) if dropout else None
        self.include_state_embedding = include_state
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch, **kwargs):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            g (pgl.Graph): PGL graphs
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Prediction
        """

        g = batch["graph"]
        batch_size = g.num_graph
        state_attr = paddle.zeros([batch_size, 2])
        node_attr = g.node_feat["atom_types"]
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
        results = {}

        for key in self.property_names:
            results[key] = self.heads[key](vec)

        return results
