import os
import sys

import paddle

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.basis_layers import BesselBasisLayer
from .layers.basis_layers import SphericalBasisLayer
from .layers.basis_layers import TensorBasisLayer
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding
from .layers.embedding_block import EdgeEmbedding
from .layers.interaction_block import InteractionBlock
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.scaling import AutomaticFit
from .utils import scatter

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
except ImportError:
    tf = None

sys.path.append("/home/chenxiaoxu02/workspaces/gemnet_paddle/utils")


class GemNet(paddle.nn.Layer):
    """
    Parameters
    ----------
        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_quad: int
            (Down-projected) Embedding size in the quadruplet message
            passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_sbf: int
            Embedding size of the spherical basis transformation (two angles).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message
            passing block after the bilinear layer.
        emb_size_bil_quad: int
            Embedding size of the edge embeddings in the quadruplet-based
            message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy
            potential.
        triplets_only: bool
            If True use GemNet-T or GemNet-dT.No quadruplet based message passing.
        num_targets: int
            Number of prediction targets.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        int_cutoff: float
            Interaction cutoff for interactomic directions in Angstrom.
            No effect for GemNet-(d)T
        envelope_exponent: int
            Exponent of the envelope function. Determines the shape of the
             smooth cutoff.
        extensive: bool
            Whether the output should be extensive (proportional to the
            number of atoms)
        forces_coupled: bool
            No effect if direct_forces is False. If True enforce
            that |F_ac| = |F_ca|
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_quad: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_size_bil_quad: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        triplets_only: bool,
        num_targets: int = 1,
        direct_forces: bool = False,
        cutoff: float = 5.0,
        int_cutoff: float = 10.0,
        envelope_exponent: int = 5,
        extensive=True,
        forces_coupled: bool = False,
        output_init="HeOrthogonal",
        activation: str = "swish",
        scale_file=None,
        name="gemnet",
        **kwargs,
    ):
        super().__init__()
        assert num_blocks > 0
        self.num_targets = num_targets
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.forces_coupled = forces_coupled
        AutomaticFit.reset()
        self.direct_forces = direct_forces
        self.triplets_only = triplets_only
        self.rbf_basis = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )
        if not self.triplets_only:
            self.cbf_basis = SphericalBasisLayer(
                num_spherical,
                num_radial,
                cutoff=int_cutoff,
                envelope_exponent=envelope_exponent,
                efficient=False,
            )
            self.sbf_basis = TensorBasisLayer(
                num_spherical,
                num_radial,
                cutoff=cutoff,
                envelope_exponent=envelope_exponent,
                efficient=True,
            )
        self.cbf_basis3 = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            efficient=True,
        )
        if not self.triplets_only:
            self.mlp_rbf4 = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                name="MLP_rbf4_shared",
                bias=False,
            )
            self.mlp_cbf4 = Dense(
                num_radial * num_spherical,
                emb_size_cbf,
                activation=None,
                name="MLP_cbf4_shared",
                bias=False,
            )
            self.mlp_sbf4 = EfficientInteractionDownProjection(
                num_spherical**2, num_radial, emb_size_sbf, name="MLP_sbf4_shared"
            )
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbf3_shared",
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf, name="MLP_cbf3_shared"
        )
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbfh_shared",
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            name="MLP_rbfout_shared",
            bias=False,
        )
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )
        out_blocks = []
        int_blocks = []
        interaction_block = (
            InteractionBlockTripletsOnly if self.triplets_only else InteractionBlock
        )
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_quad=emb_size_quad,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    emb_size_bil_quad=emb_size_bil_quad,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i + 1}",
                )
            )
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=direct_forces,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )
        self.out_blocks = paddle.nn.LayerList(sublayers=out_blocks)
        self.int_blocks = paddle.nn.LayerList(sublayers=int_blocks)

    @staticmethod
    def calculate_interatomic_vectors(R, id_s, id_t):
        """
        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_s: Tensor, shape = (nEdges,)
                Indices of the source atom of the edges.
            id_t: Tensor, shape = (nEdges,)
                Indices of the target atom of the edges.

        Returns
        -------
            (D_st, V_st): tuple
                D_st: Tensor, shape = (nEdges,)
                    Distance from atom t to s.
                V_st: Tensor, shape = (nEdges,)
                    Unit direction from atom t to s.
        """
        Rt = R[id_t]
        Rs = R[id_s]
        V_st = Rt - Rs
        D_st = paddle.sqrt(x=paddle.sum(x=V_st**2, axis=1))
        V_st = V_st / D_st[..., None]
        return D_st, V_st

    @staticmethod
    def calculate_neighbor_angles(R_ac, R_ab):
        """Calculate angles between atoms c <- a -> b.

        Parameters
        ----------
            R_ac: Tensor, shape = (N,3)
                Vector from atom a to c.
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.

        Returns
        -------
            angle_cab: Tensor, shape = (N,)
                Angle between atoms c <- a -> b.
        """
        x = paddle.sum(x=R_ac * R_ab, axis=1)
        y = paddle.cross(x=R_ac, y=R_ab).norm(axis=-1)
        y = paddle.maximum(y, paddle.to_tensor(data=1e-09))
        angle = paddle.atan2(x=y, y=x)
        return angle

    @staticmethod
    def vector_rejection(R_ab, P_n):
        """
        Project the vector R_ab onto a plane with normal vector P_n.

        Parameters
        ----------
            R_ab: Tensor, shape = (N,3)
                Vector from atom a to b.
            P_n: Tensor, shape = (N,3)
                Normal vector of a plane onto which to project R_ab.

        Returns
        -------
            R_ab_proj: Tensor, shape = (N,3)
                Projected vector (orthogonal to P_n).
        """
        a_x_b = paddle.sum(x=R_ab * P_n, axis=-1)
        b_x_b = paddle.sum(x=P_n * P_n, axis=-1)
        return R_ab - (a_x_b / b_x_b)[:, None] * P_n

    @staticmethod
    def calculate_angles(
        R,
        id_c,
        id_a,
        id4_int_b,
        id4_int_a,
        id4_expand_abd,
        id4_reduce_cab,
        id4_expand_intm_db,
        id4_reduce_intm_ca,
        id4_expand_intm_ab,
        id4_reduce_intm_ab,
    ):
        """Calculate angles for quadruplet-based message passing.

        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_c: Tensor, shape = (nEdges,)
                Indices of atom c (source atom of edge).
            id_a: Tensor, shape = (nEdges,)
                Indices of atom a (target atom of edge).
            id4_int_b: torch.Tensor, shape (nInterEdges,)
                Indices of the atom b of the interaction edge.
            id4_int_a: torch.Tensor, shape (nInterEdges,)
                Indices of the atom a of the interaction edge.
            id4_expand_abd: torch.Tensor, shape (nQuadruplets,)
                Indices to map from intermediate d->b to quadruplet d->b.
            id4_reduce_cab: torch.Tensor, shape (nQuadruplets,)
                Indices to map from intermediate c->a to quadruplet c->a.
            id4_expand_intm_db: torch.Tensor, shape (intmTriplets,)
                Indices to map d->b to intermediate d->b.
            id4_reduce_intm_ca: torch.Tensor, shape (intmTriplets,)
                Indices to map c->a to intermediate c->a.
            id4_expand_intm_ab: torch.Tensor, shape (intmTriplets,)
                Indices to map b-a to intermediate b-a of the quadruplet's
                part a-b<-d.
            id4_reduce_intm_ab: torch.Tensor, shape (intmTriplets,)
                Indices to map b-a to intermediate b-a of the quadruplet's
                part c->a-b.

        Returns
        -------
            angle_cab: Tensor, shape = (nQuadruplets,)
                Angle between atoms c <- a -> b.
            angle_abd: Tensor, shape = (intmTriplets,)
                Angle between atoms a <- b -> d.
            angle_cabd: Tensor, shape = (nQuadruplets,)
                Angle between atoms c <- a-b -> d.
        """
        Ra = R[id4_int_a[id4_expand_intm_ab]]
        Rb = R[id4_int_b[id4_expand_intm_ab]]
        Rd = R[id_c[id4_expand_intm_db]]
        R_ba = Ra - Rb
        R_bd = Rd - Rb
        angle_abd = GemNet.calculate_neighbor_angles(R_ba, R_bd)
        R_bd_proj = GemNet.vector_rejection(R_bd, R_ba)
        R_bd_proj = R_bd_proj[id4_expand_abd]
        Rc = R[id_c[id4_reduce_intm_ca]]
        Ra = R[id_a[id4_reduce_intm_ca]]
        Rb = R[id4_int_b[id4_reduce_intm_ab]]
        R_ac = Rc - Ra
        R_ab = Rb - Ra
        angle_cab = GemNet.calculate_neighbor_angles(R_ab, R_ac)
        angle_cab = angle_cab[id4_reduce_cab]
        R_ac_proj = GemNet.vector_rejection(R_ac, R_ab)
        R_ac_proj = R_ac_proj[id4_reduce_cab]
        angle_cabd = GemNet.calculate_neighbor_angles(R_ac_proj, R_bd_proj)
        return angle_cab, angle_abd, angle_cabd

    @staticmethod
    def calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba):
        """Calculate angles for triplet-based message passing.

        Parameters
        ----------
            R: Tensor, shape = (nAtoms,3)
                Atom positions.
            id_c: Tensor, shape = (nEdges,)
                Indices of atom c (source atom of edge).
            id_a: Tensor, shape = (nEdges,)
                Indices of atom a (target atom of edge).
            id3_reduce_ca: Tensor, shape = (nTriplets,)
                Edge indices of edge c -> a of the triplets.
            id3_expand_ba: Tensor, shape = (nTriplets,)
                Edge indices of edge b -> a of the triplets.

        Returns
        -------
            angle_cab: Tensor, shape = (nTriplets,)
                Angle between atoms c <- a -> b.
        """
        Rc = R[id_c[id3_reduce_ca]]
        Ra = R[id_a[id3_reduce_ca]]
        Rb = R[id_c[id3_expand_ba]]
        # print(R.shape, id_c.shape, id_a.shape, id3_reduce_ca.shape, id3_expand_ba.shape, Ra.shape, Rb.shape, Rc.shape)
        # print(id_c, id3_reduce_ca)
        R_ac = Rc - Ra
        R_ab = Rb - Ra
        return GemNet.calculate_neighbor_angles(R_ac, R_ab)

    def forward(self, inputs):
        Z, R = inputs["Z"], inputs["R"]
        id_a, id_c, id_undir, id_swap = (
            inputs["id_a"],
            inputs["id_c"],
            inputs["id_undir"],
            inputs["id_swap"],
        )
        id3_expand_ba, id3_reduce_ca = inputs["id3_expand_ba"], inputs["id3_reduce_ca"]
        if not self.triplets_only:
            batch_seg, Kidx4, Kidx3 = (
                inputs["batch_seg"],
                inputs["Kidx4"],
                inputs["Kidx3"],
            )
            id4_int_b, id4_int_a = inputs["id4_int_b"], inputs["id4_int_a"]
            # id4_reduce_ca, id4_expand_db = (
            #     inputs["id4_reduce_ca"],
            #     inputs["id4_expand_db"],
            # )
            id4_reduce_ca, _ = (
                inputs["id4_reduce_ca"],
                inputs["id4_expand_db"],
            )
            id4_reduce_cab, id4_expand_abd = (
                inputs["id4_reduce_cab"],
                inputs["id4_expand_abd"],
            )
            id4_reduce_intm_ca, id4_expand_intm_db = (
                inputs["id4_reduce_intm_ca"],
                inputs["id4_expand_intm_db"],
            )
            id4_reduce_intm_ab, id4_expand_intm_ab = (
                inputs["id4_reduce_intm_ab"],
                inputs["id4_expand_intm_ab"],
            )
        else:
            batch_seg, Kidx4, Kidx3 = inputs["batch_seg"], None, inputs["Kidx3"]
            id4_int_b, id4_int_a = None, None
            # id4_reduce_ca, id4_expand_db = None, None
            id4_reduce_ca = None
            id4_reduce_cab, id4_expand_abd = None, None
            id4_reduce_intm_ca, id4_expand_intm_db = None, None
            id4_reduce_intm_ab, id4_expand_intm_ab = None, None
        if not self.direct_forces:
            inputs["R"].stop_gradient = not True
        D_ca, V_ca = self.calculate_interatomic_vectors(R, id_c, id_a)
        if not self.triplets_only:
            D_ab, _ = self.calculate_interatomic_vectors(R, id4_int_b, id4_int_a)
            Phi_cab, Phi_abd, Theta_cabd = self.calculate_angles(
                R,
                id_c,
                id_a,
                id4_int_b,
                id4_int_a,
                id4_expand_abd,
                id4_reduce_cab,
                id4_expand_intm_db,
                id4_reduce_intm_ca,
                id4_expand_intm_ab,
                id4_reduce_intm_ab,
            )
            cbf4 = self.cbf_basis(D_ab, Phi_abd, id4_expand_intm_ab, None)
            sbf4 = self.sbf_basis(D_ca, Phi_cab, Theta_cabd, id4_reduce_ca, Kidx4)
        rbf = self.rbf_basis(D_ca)
        Angles3_cab = self.calculate_angles3(
            R, id_c, id_a, id3_reduce_ca, id3_expand_ba
        )
        cbf3 = self.cbf_basis3(D_ca, Angles3_cab, id3_reduce_ca, Kidx3)
        h = self.atom_emb(Z)
        m = self.edge_emb(h, rbf, id_c, id_a)
        if not self.triplets_only:
            rbf4 = self.mlp_rbf4(rbf)
            cbf4 = self.mlp_cbf4(cbf4)
            sbf4 = self.mlp_sbf4(sbf4)
        else:
            rbf4 = None
            cbf4 = None
            sbf4 = None
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(cbf3)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_a, F_ca = self.out_blocks[0](h, m, rbf_out, id_a)
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf4=rbf4,
                cbf4=cbf4,
                sbf4=sbf4,
                Kidx4=Kidx4,
                rbf3=rbf3,
                cbf3=cbf3,
                Kidx3=Kidx3,
                id_swap=id_swap,
                id3_expand_ba=id3_expand_ba,
                id3_reduce_ca=id3_reduce_ca,
                id4_reduce_ca=id4_reduce_ca,
                id4_expand_intm_db=id4_expand_intm_db,
                id4_expand_abd=id4_expand_abd,
                rbf_h=rbf_h,
                id_c=id_c,
                id_a=id_a,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, id_a)
            F_ca += F
            E_a += E
        nMolecules = paddle.max(x=batch_seg) + 1
        if self.extensive:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="add")
        else:
            E_a = scatter(E_a, batch_seg, dim=0, dim_size=nMolecules, reduce="mean")
        if self.direct_forces:
            nAtoms = tuple(Z.shape)[0]
            if self.forces_coupled:
                nEdges = tuple(id_c.shape)[0]
                F_ca = scatter(
                    F_ca, id_undir, dim=0, dim_size=int(nEdges / 2), reduce="mean"
                )
                F_ca = F_ca[id_undir]
            F_ji = F_ca[:, :, None] * V_ca[:, None, :]
            F_j = scatter(F_ji, id_a, dim=0, dim_size=nAtoms, reduce="add")
        else:
            if self.num_targets > 1:
                forces = []
                for i in range(self.num_targets):
                    forces += [
                        -paddle.grad(
                            outputs=E_a[:, i].sum(),
                            inputs=inputs["R"],
                            create_graph=True,
                        )[0]
                    ]
                F_j = paddle.stack(x=forces, axis=1)
            else:
                F_j = -paddle.grad(
                    outputs=E_a.sum(), inputs=inputs["R"], create_graph=True
                )[0]
            inputs["R"].stop_gradient = not False
        return E_a, F_j

    def load_tfmodel(self, path):
        reader = tf.train.load_checkpoint(path)

        def copy_(src, name):
            W = reader.get_tensor(f"{name}/.ATTRIBUTES/VARIABLE_VALUE")
            if name[-12:] == "scale_factor":
                W = paddle.to_tensor(data=W)
            else:
                W = paddle.to_tensor(data=W)
            if name[-6:] == "kernel":
                if len(tuple(W.shape)) == 2:
                    W = W.t()
            src.data.copy_(W)

        copy_(self.rbf_basis.frequencies, "rbf_basis/frequencies")
        copy_(self.atom_emb.embeddings.weight, "atom_emb/embeddings")
        copy_(self.edge_emb.dense.weight, "edge_emb/dense/kernel")
        shared_mlps = ["mlp_cbf3", "mlp_rbf3", "mlp_rbf_h", "mlp_rbf_out"]
        if not self.triplets_only:
            shared_mlps += ["mlp_rbf4", "mlp_cbf4", "mlp_sbf4"]
        for layer in shared_mlps:
            copy_(getattr(self, layer).weight, f"{layer}/kernel")
        for i, block in enumerate(self.int_blocks):
            if not self.triplets_only:
                for layer in [
                    "dense_db",
                    "mlp_rbf",
                    "mlp_cbf",
                    "mlp_sbf",
                    "down_projection",
                    "up_projection_ca",
                    "up_projection_ac",
                ]:
                    copy_(
                        getattr(block.quad_interaction, layer).weight,
                        f"int_blocks/{i}/quad_interaction/{layer}/kernel",
                    )
                for layer in ["rbf", "cbf", "sbf_sum"]:
                    copy_(
                        getattr(block.quad_interaction, f"scale_{layer}").scale_factor,
                        f"""int_blocks/{i}/quad_interaction
                        /scale_{layer}/scale_factor""",
                    )
            for layer in [
                "dense_ba",
                "mlp_rbf",
                "mlp_cbf",
                "down_projection",
                "up_projection_ac",
                "up_projection_ca",
            ]:
                copy_(
                    getattr(block.trip_interaction, layer).weight,
                    f"int_blocks/{i}/trip_interaction/{layer}/kernel",
                )
            for layer in ["rbf", "cbf_sum"]:
                copy_(
                    getattr(block.trip_interaction, f"scale_{layer}").scale_factor,
                    f"int_blocks/{i}/trip_interaction/scale_{layer}/scale_factor",
                )
            copy_(
                block.atom_update.dense_rbf.weight,
                f"int_blocks/{i}/atom_update/dense_rbf/kernel",
            )
            copy_(
                block.atom_update.scale_sum.scale_factor,
                f"int_blocks/{i}/atom_update/scale_sum/scale_factor",
            )
            copy_(
                block.atom_update.layers[0].weight,
                f"int_blocks/{i}/atom_update/layers/0/kernel",
            )
            for j, res_layer in enumerate(block.atom_update.layers[1:]):
                j = j + 1
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(
                        layer.weight,
                        f"int_blocks/{i}/atom_update/layers/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                    )
            copy_(
                block.concat_layer.dense.weight,
                f"int_blocks/{i}/concat_layer/dense/kernel",
            )
            copy_(block.dense_ca.weight, f"int_blocks/{i}/dense_ca/kernel")
            for j, res_layer in enumerate(block.layers_after_skip):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(
                        layer.weight,
                        f"int_blocks/{i}/layers_after_skip/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                    )
            for j, res_layer in enumerate(block.layers_before_skip):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(
                        layer.weight,
                        f"int_blocks/{i}/layers_before_skip/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                    )
            for j, res_layer in enumerate(block.residual_m):
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(
                        layer.weight,
                        f"int_blocks/{i}/residual_m/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                    )
        for i, block in enumerate(self.out_blocks):
            copy_(block.dense_rbf.weight, f"out_blocks/{i}/dense_rbf/kernel")
            copy_(block.layers[0].weight, f"out_blocks/{i}/layers/0/kernel")
            for j, res_layer in enumerate(block.layers[1:]):
                j = j + 1
                for k, layer in enumerate(res_layer.dense_mlp):
                    copy_(
                        layer.weight,
                        f"out_blocks/{i}/layers/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                    )
            copy_(block.out_energy.weight, f"out_blocks/{i}/out_energy/kernel")
            copy_(
                block.scale_sum.scale_factor, f"out_blocks/{i}/scale_sum/scale_factor"
            )
            if self.direct_forces:
                copy_(block.out_forces.weight, f"out_blocks/{i}/out_forces/kernel")
                copy_(block.out_forces.bias, f"out_blocks/{i}/out_forces/bias")
                copy_(block.seq_forces[0].weight, f"out_blocks/{i}/seq_forces/0/kernel")
                copy_(
                    block.scale_rbf.scale_factor,
                    f"out_blocks/{i}/scale_rbf/scale_factor",
                )
                for j, res_layer in enumerate(block.seq_forces[1:]):
                    j = j + 1
                    for k, layer in enumerate(res_layer.dense_mlp):
                        copy_(
                            layer.weight,
                            f"out_blocks/{i}/seq_forces/{j}/dense_mlp/layer_with_weights-{k}/kernel",
                        )

    def predict(self, inputs):
        E, F = self(inputs)
        E = E.detach().cpu()
        F = F.detach().cpu()
        return E, F

    def load_weights(self, path):
        self.set_state_dict(state_dict=paddle.load(path=path))

    def save_weights(self, path):
        paddle.save(obj=self.state_dict(), path=path)
