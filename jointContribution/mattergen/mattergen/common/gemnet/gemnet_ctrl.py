import paddle

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/gemnet.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
from typing import Dict
from typing import List
from typing import Optional

from paddle_scatter import scatter

from mattergen.common.data.types import PropertySourceId
from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.gemnet import ModelOutput
from mattergen.common.gemnet.utils import inner_product_normalized
from mattergen.common.utils.data_utils import frac_to_cart_coords_with_lattice
from mattergen.common.utils.data_utils import lattice_params_to_matrix_paddle


class GemNetTCtrl(GemNetT):
    """
    GemNet-T, triplets-only variant of GemNet

    This variation allows for layerwise conditional control for the purpose of
    conditional finetuning. It adds the following on top of GemNetT:

    for each condition in <condition_on_adapt>:

    1. a series of adapt layers that take the concatenation of the node embedding
       and the condition embedding, process it with an MLP. There is one adapt layer
       for each GemNetT message passing block.
    2. a series of mixin layers that take the output of the adapt layer and mix it in
       to the atom embedding. There is one mixin layer for each GemNetT message passing block.
       The mixin layers are initialized to zeros so at the beginning of training, the model
       outputs exactly the same scores as the base GemNetT model.

    """

    def __init__(self, condition_on_adapt: List[PropertySourceId], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_on_adapt = condition_on_adapt
        self.cond_adapt_layers = paddle.nn.LayerDict()
        self.cond_mixin_layers = paddle.nn.LayerDict()
        self.emb_size_atom = kwargs["emb_size_atom"] if "emb_size_atom" in kwargs else 512
        for cond in condition_on_adapt:
            adapt_layers = []
            mixin_layers = []
            for _ in range(self.num_blocks):
                adapt_layers.append(
                    paddle.nn.Sequential(
                        paddle.nn.Linear(
                            in_features=self.emb_size_atom * 2,
                            out_features=self.emb_size_atom,
                        ),
                        paddle.nn.ReLU(),
                        paddle.nn.Linear(
                            in_features=self.emb_size_atom,
                            out_features=self.emb_size_atom,
                        ),
                    )
                )
                mixin_layers.append(
                    paddle.nn.Linear(
                        in_features=self.emb_size_atom,
                        out_features=self.emb_size_atom,
                        bias_attr=False,
                    )
                )
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(mixin_layers[-1].weight)
            self.cond_adapt_layers[cond] = paddle.nn.LayerList(sublayers=adapt_layers)
            self.cond_mixin_layers[cond] = paddle.nn.LayerList(sublayers=mixin_layers)

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
        charges: Optional[paddle.Tensor] = None,
        cond_adapt: Optional[Dict[PropertySourceId, paddle.Tensor]] = None,
        cond_adapt_mask: Optional[Dict[PropertySourceId, paddle.Tensor]] = None,
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
            cond_adapt: (N_cryst, num_cond, dim_cond) (optional, conditional signal for score prediction)
            cond_adapt_mask: (N_cryst, num_cond) (optional, mask for which data points receive conditional signal)
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
        if cond_adapt is not None and cond_adapt_mask is not None:
            cond_adapt_per_atom = {}
            cond_adapt_mask_per_atom = {}
            for cond in self.condition_on_adapt:
                cond_adapt_per_atom[cond] = cond_adapt[cond][batch]
                cond_adapt_mask_per_atom[cond] = 1.0 - cond_adapt_mask[cond][batch].astype(
                    dtype="float32"
                )
        for i in range(self.num_blocks):
            h_adapt = paddle.zeros_like(x=h)
            for cond in self.condition_on_adapt:
                h_adapt_cond = self.cond_adapt_layers[cond][i](
                    paddle.concat(x=[h, cond_adapt_per_atom[cond]], axis=-1)
                )
                h_adapt_cond = self.cond_mixin_layers[cond][i](h_adapt_cond)
                h_adapt += cond_adapt_mask_per_atom[cond] * h_adapt_cond
            h = h + h_adapt
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
        E_t = scatter(E_t, batch, dim=0, dim_size=nMolecules, reduce="sum")
        output = dict(energy=E_t, node_embeddings=h)
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)
        output["forces"] = F_t
        if self.regress_stress:
            output["stress"] = lattice_update
        return ModelOutput(**output)

    @property
    def num_params(self):
        return sum(p.size for p in self.parameters())
