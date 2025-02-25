import sys


from typing import Callable

import paddle
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId
from mattergen.denoiser import (GemNetTDenoiser,
                                get_chemgraph_from_denoiser_output)
from mattergen.property_embeddings import (ZerosEmbedding,
                                           get_property_embeddings,
                                           get_use_unconditional_embedding)
from paddle_utils import *

BatchTransform = Callable[[ChemGraph], ChemGraph]


class GemNetTAdapter(GemNetTDenoiser):
    """
    Denoiser layerwise adapter with GemNetT. On top of a mattergen.denoiser.GemNetTDenoiser,
    additionally inputs <property_embeddings_adapt> that specifies extra conditions to be conditioned on.
    """

    def __init__(self, property_embeddings_adapt: paddle.nn.LayerDict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.property_embeddings_adapt = paddle.nn.LayerDict(
            sublayers=property_embeddings_adapt
        )
        assert all(
            [
                (k not in self.property_embeddings.keys())
                for k in self.property_embeddings_adapt.keys()
            ]
        ), f"One of adapter conditions {self.property_embeddings_adapt.keys()} already exists in base model {self.property_embeddings.keys()}, please remove."
        for property_embedding in self.property_embeddings_adapt.values():
            property_embedding.unconditional_embedding_module = ZerosEmbedding(
                hidden_dim=property_embedding.unconditional_embedding_module.hidden_dim
            )

    def forward(self, x: ChemGraph, t: paddle.Tensor) -> ChemGraph:
        """
        augment <z_per_crystal> with <self.condition_embs_adapt>.
        """
        frac_coords, lattice, atom_types, num_atoms, batch = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        t_enc = self.noise_level_encoding(t).to(lattice.place)
        z_per_crystal = t_enc
        conditions_base_model: paddle.Tensor = get_property_embeddings(
            property_embeddings=self.property_embeddings, batch=x
        )
        if len(conditions_base_model) > 0:
            z_per_crystal = paddle.concat(
                x=[z_per_crystal, conditions_base_model], axis=-1
            )
        conditions_adapt_dict = {}
        conditions_adapt_mask_dict = {}
        for cond_field, property_embedding in self.property_embeddings_adapt.items():
            conditions_adapt_dict[cond_field] = property_embedding.forward(batch=x)
            try:
                conditions_adapt_mask_dict[
                    cond_field
                ] = get_use_unconditional_embedding(batch=x, cond_field=cond_field)
            except KeyError:
                conditions_adapt_mask_dict[cond_field] = paddle.ones_like(
                    x=x["num_atoms"], dtype="bool"
                ).reshape(-1, 1)
        output = self.gemnet(
            z=z_per_crystal,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch,
            lengths=None,
            angles=None,
            lattice=lattice,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            cond_adapt=conditions_adapt_dict,
            cond_adapt_mask=conditions_adapt_mask_dict,
        )
        pred_atom_types = self.fc_atom(output.node_embeddings)
        return get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=output.stress,
            pred_cart_pos_eps=output.forces,
            training=self.training,
            element_mask_func=self.element_mask_func,
            x_input=x,
        )

    @property
    def cond_fields_model_was_trained_on(self) -> list[PropertySourceId]:
        """
        We adopt the convention that all property embeddings are stored in torch.nn.ModuleDicts of
        name property_embeddings or property_embeddings_adapt in the case of a fine tuned model.

        This function returns the list of all field names that a given score model was trained to
        condition on.
        """
        return list(self.property_embeddings) + list(self.property_embeddings_adapt)
