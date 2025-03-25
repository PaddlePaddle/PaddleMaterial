import sys

import paddle

from paddle_utils import *

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/gemnet.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
from typing import Optional
from paddle_scatter import scatter

from mattergen.common.gemnet.layers.radial_basis import RadialBasis
from mattergen.common.gemnet.utils import inner_product_normalized
from mattergen.common.utils.data_utils import frac_to_cart_coords_with_lattice
from mattergen.common.utils.data_utils import lattice_params_to_matrix_paddle


from mattergen.common.gemnet.gemnet import GemNetT, ModelOutput

class GemNetT_MD(GemNetT):
    """
    GemNet-T_MD, triplets-only variant of GemNet, with mean distance embedding. 

    Parameters
    ----------
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        atom_embedding: paddle.nn.Layer
            a module that embeds atomic numbers into vectors of size emb_dim_atomic_number.
        emb_size_atom: int
            Embedding size of the atoms. This can be different from emb_dim_atomic_number.
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
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
        encoder_mode: bool
            if <True>, use the encoder mode of the model, i.e. only get the atom/edge embedddings.
        use_md: bool
            if <True>, use the mean distance embedding. 
    """

    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
        atom_embedding: paddle.nn.Layer,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_stress: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_cell_images_per_dim: int = 5,
        encoder_mode: bool = False,
        use_md: bool=True,
        **kwargs,
    ):
        super().__init__(
            num_targets=num_targets,
            latent_dim=latent_dim,
            atom_embedding=atom_embedding,
            num_spherical=num_spherical,
            num_radial=num_radial,
            num_blocks=num_blocks,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            emb_size_bil_trip=emb_size_bil_trip,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_concat=num_concat,
            num_atom=num_atom,
            regress_stress=regress_stress,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            rbf=rbf,
            envelope=envelope,
            cbf=cbf,
            otf_graph=otf_graph,
            output_init=output_init,
            activation=activation,
            max_cell_images_per_dim=max_cell_images_per_dim,
            encoder_mode=encoder_mode,
        )

        self.use_md = use_md
        # radial basis function for mean distance
        if use_md:
            self.radial_basis_md = RadialBasis(
                num_radial=num_radial, cutoff=0.5, rbf=rbf, envelope=envelope
            )
            self.radial_basis_md_linear = paddle.nn.Linear(
                in_features=num_radial,
                out_features=emb_size_atom,
                bias_attr=False,
            )

    def forward(
        self,
        z: paddle.Tensor,
        frac_coords: paddle.Tensor,
        atom_types: paddle.Tensor,
        num_atoms: paddle.Tensor,
        batch: paddle.Tensor,
        lengths: Optional[paddle.Tensor] = None,
        angles: Optional[paddle.Tensor] = None,
        edge_index: Optional[paddle.Tensor] = None,
        to_jimages: Optional[paddle.Tensor] = None,
        num_bonds: Optional[paddle.Tensor] = None,
        lattice: Optional[paddle.Tensor] = None,
    ) -> ModelOutput:
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ) with D3PM need to use atomic number
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            angles: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            edge_index: (2, N_edge) (optional, only needed if self.otf_graph is False)
            to_jimages: (N_edge, 3) (optional, only needed if self.otf_graph is False)
            num_bonds: (N_cryst,) (optional, only needed if self.otf_graph is False)
            lattice: (N_cryst, 3, 3) (optional, either lengths and angles or lattice must be passed)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        if self.otf_graph:
            assert all(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is active but received input graph information."
        else:
            assert not any(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is off but received no input graph information."
        assert (angles is None and lengths is None) != (
            lattice is None
        ), "Either lattice or lengths and angles must be provided, not both or none."
        if angles is not None and lengths is not None:
            lattice = lattice_params_to_matrix_paddle(lengths, angles)
        assert lattice is not None
        distorted_lattice = lattice
        pos = frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice=distorted_lattice)

        atomic_numbers = atom_types.cast(dtype="int64")
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = self.generate_interaction_graph(
            pos, distorted_lattice, num_atoms, edge_index, to_jimages, num_bonds
        )

        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z[batch]
            h = paddle.concat(x=[h, z_per_atom], axis=1)
            h = self.atom_latent_emb(h)

        if self.use_md:
            mean_distance = (frac_coords[:, 2] - 0.5).abs()
            mean_distance = scatter(mean_distance, batch, reduce='mean')
            mean_distance_rbf = self.radial_basis_md(mean_distance)
            md_rbf_out = self.radial_basis_md_linear(mean_distance_rbf)
            md_rbf_out = md_rbf_out[batch]
            h = h + md_rbf_out

        m = self.edge_emb(h, rbf, idx_s, idx_t)
        batch_edge = batch[edge_index[0]]
        cosines = paddle.nn.functional.cosine_similarity(
            x1=V_st[:, None], x2=distorted_lattice[batch_edge], axis=-1
        )
        m = paddle.concat(x=[m, cosines], axis=-1)
        m = self.angle_edge_emb(m)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        distance_vec = V_st * D_st[:, None]
        lattice_update = None
        rbf_lattice = self.mlp_rbf_lattice(rbf)
        lattice_update = self.lattice_out_blocks[0](
            edge_emb=m,
            edge_index=edge_index,
            distance_vec=distance_vec,
            lattice=distorted_lattice,
            batch=batch,
            rbf=rbf_lattice,
            normalize_score=True,
        )
        F_fully_connected = paddle.to_tensor(data=0.0, place=distorted_lattice.place)
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            F_st += F
            E_t += E
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update += self.lattice_out_blocks[i + 1](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=distorted_lattice,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )
        nMolecules = paddle.max(x=batch) + 1
        if self.encoder_mode:
            return E_t
        E_t = scatter(E_t, batch, dim=0, dim_size=nMolecules, reduce="sum")
        output = dict(energy=E_t, node_embeddings=h)
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)
        output["forces"] = F_t + F_fully_connected
        if self.regress_stress:
            output["stress"] = lattice_update
        return ModelOutput(**output)

