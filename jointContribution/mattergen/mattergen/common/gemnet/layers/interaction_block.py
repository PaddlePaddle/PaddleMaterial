import paddle

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/interaction_block.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
import math

from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock
from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.common.gemnet.layers.base_layers import ResidualLayer
from mattergen.common.gemnet.layers.efficient import EfficientInteractionBilinear
from mattergen.common.gemnet.layers.embedding_block import EdgeEmbedding
from mattergen.common.gemnet.layers.scaling import ScalingFactor


class InteractionBlockTripletsOnly(paddle.nn.Layer):
    """
    Interaction block for GemNet-T/dT.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_bil_trip,
        num_before_skip,
        num_after_skip,
        num_concat,
        num_atom,
        activation=None,
        scale_file=None,
        name="Interaction",
    ):
        super().__init__()
        self.name = name
        self.skip_connection_factor = 2.0**-0.5
        block_nr = name.split("_")[-1]
        self.dense_ca = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False)
        self.trip_interaction = TripletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            scale_file=scale_file,
            name=f"TripInteraction_{block_nr}",
        )
        self.layers_before_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation) for i in range(num_before_skip)
            ]
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation) for i in range(num_after_skip)
            ]
        )
        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
            scale_file=scale_file,
            name=f"AtomUpdate_{block_nr}",
        )
        self.concat_layer = EdgeEmbedding(
            emb_size_atom, emb_size_edge, emb_size_edge, activation=activation
        )
        self.residual_m = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation) for _ in range(num_concat)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h,
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        rbf_h,
        idx_s,
        idx_t,
    ):
        """
        Returns
        -------
            h: paddle.Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: paddle.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ca_skip = self.dense_ca(m)
        x3 = self.trip_interaction(m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca)
        x = x_ca_skip + x3
        x = x * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, idx_t)
        h = h + h2
        h = h * self.skip_connection_factor
        m2 = self.concat_layer(h, m, idx_s, idx_t)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class TripletInteraction(paddle.nn.Layer):
    """
    Triplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).

        activation: str
            Name of the activation function to use in the dense layers except for the final dense layer.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_trip,
        emb_size_bilinear,
        emb_size_rbf,
        emb_size_cbf,
        activation=None,
        scale_file=None,
        name="TripletInteraction",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.dense_ba = Dense(emb_size_edge, emb_size_edge, activation=activation, bias=False)
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")
        self.mlp_cbf = EfficientInteractionBilinear(emb_size_trip, emb_size_cbf, emb_size_bilinear)
        self.scale_cbf_sum = ScalingFactor(scale_file=scale_file, name=name + "_sum_cbf")
        self.down_projection = Dense(
            emb_size_edge, emb_size_trip, activation=activation, bias=False
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear, emb_size_edge, activation=activation, bias=False
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear, emb_size_edge, activation=activation, bias=False
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca):
        """
        Returns
        -------
            m: paddle.Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ba = self.dense_ba(m)
        rbf_emb = self.mlp_rbf(rbf3)
        x_ba2 = x_ba * rbf_emb
        x_ba = self.scale_rbf(x_ba, x_ba2)
        x_ba = self.down_projection(x_ba)
        x_ba = x_ba[id3_ba]
        x = self.mlp_cbf(cbf3, x_ba, id3_ca, id3_ragged_idx)
        x = self.scale_cbf_sum(x_ba, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3
