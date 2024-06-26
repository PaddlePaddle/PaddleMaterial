import paddle

from .atom_update_block import AtomUpdateBlock
from .base_layers import Dense
from .base_layers import ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding
from .scaling import ScalingFactor


class InteractionBlock(paddle.nn.Layer):
    """
    Interaction block for GemNet-Q/dQ.

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_quad,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
        emb_size_bil_trip,
        emb_size_bil_quad,
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
        block_nr = name.split("_")[-1]
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_ca",
        )
        self.quad_interaction = QuadrupletInteraction(
            emb_size_edge=emb_size_edge,
            emb_size_quad=emb_size_quad,
            emb_size_bilinear=emb_size_bil_quad,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            emb_size_sbf=emb_size_sbf,
            activation=activation,
            scale_file=scale_file,
            name=f"QuadInteraction_{block_nr}",
        )
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
                ResidualLayer(
                    emb_size_edge, activation=activation, name=f"res_bef_skip_{i}"
                )
                for i in range(num_before_skip)
            ]
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(
                    emb_size_edge, activation=activation, name=f"res_aft_skip_{i}"
                )
                for i in range(num_after_skip)
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
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            name="concat",
        )
        self.residual_m = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation, name=f"res_m_{i}")
                for i in range(num_concat)
            ]
        )
        self.inv_sqrt_2 = 1 / 2.0**0.5
        self.inv_sqrt_3 = 1 / 3.0**0.5

    def forward(
        self,
        h,
        m,
        rbf4,
        cbf4,
        sbf4,
        Kidx4,
        rbf3,
        cbf3,
        Kidx3,
        id_swap,
        id3_expand_ba,
        id3_reduce_ca,
        id4_reduce_ca,
        id4_expand_intm_db,
        id4_expand_abd,
        rbf_h,
        id_c,
        id_a,
    ):
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ca_skip = self.dense_ca(m)
        x4 = self.quad_interaction(
            m,
            rbf4,
            cbf4,
            sbf4,
            Kidx4,
            id_swap,
            id4_reduce_ca,
            id4_expand_intm_db,
            id4_expand_abd,
        )
        x3 = self.trip_interaction(
            m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca
        )
        x = x_ca_skip + x3 + x4
        x = x * self.inv_sqrt_3
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, id_a)
        h = h + h2
        h = h * self.inv_sqrt_2
        m2 = self.concat_layer(h, m, id_c, id_a)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


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
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip,
        emb_size_quad,
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
        **kwargs,
    ):
        super().__init__()
        self.name = name
        block_nr = name.split("_")[-1]
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_ca",
        )
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
                ResidualLayer(
                    emb_size_edge, activation=activation, name=f"res_bef_skip_{i}"
                )
                for i in range(num_before_skip)
            ]
        )
        self.layers_after_skip = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(
                    emb_size_edge, activation=activation, name=f"res_aft_skip_{i}"
                )
                for i in range(num_after_skip)
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
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            name="concat",
        )
        self.residual_m = paddle.nn.LayerList(
            sublayers=[
                ResidualLayer(emb_size_edge, activation=activation, name=f"res_m_{i}")
                for i in range(num_concat)
            ]
        )
        self.inv_sqrt_2 = 1 / 2.0**0.5

    def forward(
        self,
        h,
        m,
        rbf3,
        cbf3,
        Kidx3,
        id_swap,
        id3_expand_ba,
        id3_reduce_ca,
        rbf_h,
        id_c,
        id_a,
        **kwargs,
    ):
        """
        Returns
        -------
            h: Tensor, shape=(nEdges, emb_size_atom)
                Atom embeddings.
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ca_skip = self.dense_ca(m)
        x3 = self.trip_interaction(
            m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca
        )
        x = x_ca_skip + x3
        x = x * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        m = m + x
        m = m * self.inv_sqrt_2
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        h2 = self.atom_update(h, m, rbf_h, id_a)
        h = h + h2
        h = h * self.inv_sqrt_2
        m2 = self.concat_layer(h, m, id_c, id_a)
        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)
        m = m + m2
        m = m * self.inv_sqrt_2
        return h, m


class QuadrupletInteraction(paddle.nn.Layer):
    """
    Quadruplet-based message passing block.

    Parameters
    ----------
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_quad: int
            (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear: int
            Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        activation: str
            Name of the activation function to use in the dense layers (except for the final dense layer).
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_quad,
        emb_size_bilinear,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
        activation=None,
        scale_file=None,
        name="QuadrupletInteraction",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.dense_db = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_db",
        )
        self.mlp_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, name="MLP_rbf4_2", bias=False
        )
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")
        self.mlp_cbf = Dense(
            emb_size_cbf, emb_size_quad, activation=None, name="MLP_cbf4_2", bias=False
        )
        self.scale_cbf = ScalingFactor(scale_file=scale_file, name=name + "_had_cbf")
        self.mlp_sbf = EfficientInteractionBilinear(
            emb_size_quad, emb_size_sbf, emb_size_bilinear, name="MLP_sbf4_2"
        )
        self.scale_sbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_sbf"
        )
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_quad,
            activation=activation,
            bias=False,
            name="dense_down",
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ca",
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ac",
        )
        self.inv_sqrt_2 = 1 / 2.0**0.5

    def forward(
        self,
        m,
        rbf,
        cbf,
        sbf,
        Kidx4,
        id_swap,
        id4_reduce_ca,
        id4_expand_intm_db,
        id4_expand_abd,
    ):
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_db = self.dense_db(m)
        x_db2 = x_db * self.mlp_rbf(rbf)
        x_db = self.scale_rbf(x_db, x_db2)
        x_db = self.down_projection(x_db)
        x_db = x_db[id4_expand_intm_db]
        x_db2 = x_db * self.mlp_cbf(cbf)
        x_db = self.scale_cbf(x_db, x_db2)
        x_db = x_db[id4_expand_abd]
        x = self.mlp_sbf(sbf, x_db, id4_reduce_ca, Kidx4)
        x = self.scale_sbf_sum(x_db, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x4 = x_ca + x_ac
        x4 = x4 * self.inv_sqrt_2
        return x4


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
            Name of the activation function to use in the dense layers (except for the final dense layer).
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
        self.dense_ba = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_ba",
        )
        self.mlp_rbf = Dense(
            emb_size_rbf, emb_size_edge, activation=None, name="MLP_rbf3_2", bias=False
        )
        self.scale_rbf = ScalingFactor(scale_file=scale_file, name=name + "_had_rbf")
        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear, name="MLP_cbf3_2"
        )
        self.scale_cbf_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum_cbf"
        )
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
            name="dense_down",
        )
        self.up_projection_ca = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ca",
        )
        self.up_projection_ac = Dense(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
            name="dense_up_ac",
        )
        self.inv_sqrt_2 = 1 / 2.0**0.5

    def forward(self, m, rbf3, cbf3, Kidx3, id_swap, id3_expand_ba, id3_reduce_ca):
        """
        Returns
        -------
            m: Tensor, shape=(nEdges, emb_size_edge)
                Edge embeddings (c->a).
        """
        x_ba = self.dense_ba(m)
        mlp_rbf = self.mlp_rbf(rbf3)
        x_ba2 = x_ba * mlp_rbf
        x_ba = self.scale_rbf(x_ba, x_ba2)
        x_ba = self.down_projection(x_ba)
        x_ba = x_ba[id3_expand_ba]
        x = self.mlp_cbf(cbf3, x_ba, id3_reduce_ca, Kidx3)
        x = self.scale_cbf_sum(x_ba, x)
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)
        x_ac = x_ac[id_swap]
        x3 = x_ca + x_ac
        x3 = x3 * self.inv_sqrt_2
        return x3
