import sys


from typing import Callable

import paddle
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId
from mattergen.common.utils.globals import (MAX_ATOMIC_NUM,
                                            SELECTED_ATOMIC_NUMBERS)
from mattergen.diffusion.model_utils import NoiseLevelEncoding
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.property_embeddings import (ChemicalSystemMultiHotEmbedding,
                                           get_property_embeddings,
                                           get_use_unconditional_embedding)
from paddle_utils import *

BatchTransform = Callable[[ChemGraph], ChemGraph]


def atomic_numbers_to_mask(
    atomic_numbers: paddle.Tensor, max_atomic_num: int
) -> paddle.Tensor:
    """Convert atomic numbers to a mask.

    Args:
        atomic_numbers (torch.LongTensor): One-based atomic numbers of shape (batch_size, )

    Returns:
        torch.Tensor: Mask of shape (batch_size, num_classes)
    """
    k_hot_mask = paddle.eye(num_rows=max_atomic_num)[atomic_numbers - 1]
    return k_hot_mask


def mask_logits(logits: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """Mask logits by setting the logits for masked items to -inf.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, num_classes)
        mask (torch.Tensor): Mask of shape (batch_size, num_classes). Values with zero are masked.

    Returns:
        torch.Tensor: Masked logits
    """
    return logits + (1 - mask) * -10000000000.0


def mask_disallowed_elements(
    logits: paddle.Tensor,
    x: (ChemGraph | None) = None,
    batch_idx=None, #: (paddle.int64 | None) = None, todo: fix this
    predictions_are_zero_based: bool = True,
):
    """
    Mask out atom types that are disallowed in general,
    as well as potentially all elements not in the chemical system we condition on.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, num_classes)
        x (ChemGraph)
        batch_idx (torch.LongTensor, optional): Batch indices. Defaults to None. Must be provided if condition is not None.
        predictions_are_zero_based (bool, optional): Whether the logits are zero-based. Defaults to True. Basically, if we're using D3PM,
            the logits are zero-based (model predicts atomic number index)
    """
    selected_atomic_numbers = paddle.to_tensor(
        data=SELECTED_ATOMIC_NUMBERS, place=logits.place
    )
    predictions_are_one_based = not predictions_are_zero_based
    one_hot_selected_elements = atomic_numbers_to_mask(
        atomic_numbers=selected_atomic_numbers + int(predictions_are_one_based),
        max_atomic_num=tuple(logits.shape)[1],
    )
    k_hot_mask = one_hot_selected_elements.sum(axis=0)[None]
    logits = mask_logits(logits=logits, mask=k_hot_mask)
    if x is not None and "chemical_system" in x and x["chemical_system"] is not None:
        try:
            do_not_mask_atom_logits = get_use_unconditional_embedding(
                batch=x, cond_field="chemical_system"
            )
        except KeyError:
            do_not_mask_atom_logits = paddle.ones(
                shape=(len(x["chemical_system"]), 1), dtype="bool"
            )
        assert (
            batch_idx is not None
        ), "batch_idx must be provided if condition is not None"
        keep_all_logits = paddle.ones(shape=(len(x["chemical_system"]), 1))
        multi_hot_chemical_system = (
            ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(
                x=ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(
                    x=x["chemical_system"]
                ),
                device=x["num_atoms"].place,
            )
        )
        keep_logits = paddle.where(
            condition=do_not_mask_atom_logits,
            x=keep_all_logits,
            y=multi_hot_chemical_system,
        )
        if predictions_are_zero_based:
            keep_logits = keep_logits[:, 1:]
            if tuple(keep_logits.shape)[1] == tuple(logits.shape)[1] - 1:
                keep_logits = paddle.concat(
                    x=[keep_logits, paddle.zeros_like(x=keep_logits[:, :1])], axis=-1
                )
        logits = mask_logits(logits, keep_logits[batch_idx])
    return logits


def get_chemgraph_from_denoiser_output(
    pred_atom_types: paddle.Tensor,
    pred_lattice_eps: paddle.Tensor,
    pred_cart_pos_eps: paddle.Tensor,
    training: bool,
    element_mask_func: (Callable | None),
    x_input: ChemGraph,
) -> ChemGraph:
    """
    Convert raw denoiser output to ChemGraph and optionally apply masking to element logits.

    Keyword arguments
    -----------------
    pred_atom_atoms: predicted logits for atom types
    pred_lattice_eps: predicted lattice noise
    pred_cart_pos_eps: predicted cartesian position noise
    training: whether or not the model is in training mode - logit masking is only applied when sampling
    element_mask_func: when not training, a function can be applied to mask logits for certain atom types
    x_input: the nosiy state input to the score model, contains the lattice to convert cartesisan to fractional noise.
    """
    if not training and element_mask_func:
        pred_atom_types = element_mask_func(
            logits=pred_atom_types, x=x_input, batch_idx=x_input.get_batch_idx("pos")
        )
    replace_dict = dict(
        pos=(
            x_input["cell"]
            .inverse()
            .transpose(perm=dim2perm(x_input["cell"].inverse().ndim, 1, 2))[
                x_input.get_batch_idx("pos")
            ]
            @ pred_cart_pos_eps.unsqueeze(axis=-1)
        ).squeeze(axis=-1),
        cell=pred_lattice_eps,
        atomic_numbers=pred_atom_types,
    )
    return x_input.replace(**replace_dict)


class GemNetTDenoiser(ScoreModel):
    """Denoiser"""

    def __init__(
        self,
        gemnet: paddle.nn.Layer,
        hidden_dim: int = 512,
        denoise_atom_types: bool = True,
        atom_type_diffusion: str = ["mask", "uniform"][0],
        property_embeddings: (paddle.nn.LayerDict | None) = None,
        property_embeddings_adapt: (paddle.nn.LayerDict | None) = None,
        element_mask_func: (Callable | None) = None,
        **kwargs
    ):
        """Construct a GemNetTDenoiser object.

        Args:
            gemnet: a GNN module
            hidden_dim (int, optional): Number of hidden dimensions in the GemNet. Defaults to 128.
            denoise_atom_types (bool, optional): Whether to denoise the atom  types. Defaults to False.
            atom_type_diffusion (str, optional): Which type of atom type diffusion to use. Defaults to "mask".
            condition_on (Optional[List[str]], optional): Which aspects of the data to condition on. Strings must be in ["property", "chemical_system"]. If None (default), condition on ["chemical_system"].
        """
        super(GemNetTDenoiser, self).__init__()
        self.gemnet = gemnet
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        self.hidden_dim = hidden_dim
        self.denoise_atom_types = denoise_atom_types
        self.atom_type_diffusion = atom_type_diffusion
        self.property_embeddings = paddle.nn.LayerDict(
            sublayers=property_embeddings or {}
        )
        with_mask_type = self.denoise_atom_types and "mask" in self.atom_type_diffusion
        self.fc_atom = paddle.nn.Linear(
            in_features=hidden_dim, out_features=MAX_ATOMIC_NUM + int(with_mask_type)
        )
        self.element_mask_func = element_mask_func

    def forward(self, x: ChemGraph, t: paddle.Tensor) -> ChemGraph:
        """
        args:
            x: tuple containing:
                frac_coords: (N_atoms, 3)
                lattice: (N_cryst, 3, 3)
                atom_types: (N_atoms, ), need to use atomic number e.g. H = 1 or ion state
                num_atoms: (N_cryst,)
                batch: (N_atoms,)
            t: (N_cryst,): timestep per crystal
        returns:
            tuple of:
                predicted epsilon: (N_atoms, 3)
                lattice update: (N_crystals, 3, 3)
                predicted atom types: (N_atoms, MAX_ATOMIC_NUM)
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
        property_embedding_values = get_property_embeddings(
            batch=x, property_embeddings=self.property_embeddings
        )
        if len(property_embedding_values) > 0:
            z_per_crystal = paddle.concat(
                x=[z_per_crystal, property_embedding_values], axis=-1
            )
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
        return list(self.property_embeddings)
