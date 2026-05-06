"""
Crystal Sampler: combines SpaceGroupSampler, LatticeSampler, WyckoffElementTransformer,
and EquivariantDiffusionModel for full crystal generation.

"""
import dataclasses
from typing import List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical

from ppmat.models.sgequidiff.crystal_classes import ASUCrystal
from ppmat.models.sgequidiff.data_utils import lattice_params_to_matrix_paddle
from ppmat.models.sgequidiff.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
from ppmat.models.sgequidiff.lattice_sampler import (
    TelescopingDiscreteLatticeSampler,
    LatticeSamplerConfig,
)
from ppmat.models.sgequidiff.wyckoff_transformer import (
    WyckoffElementTransformer,
    WyckoffElementTransformerConfig,
)


class SpaceGroupSampler(nn.Layer):
    def __init__(self):
        super().__init__()
        self.marginal_space_group_logits = paddle.create_parameter(
            shape=[230],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def sample_and_log_prob(self, batch_size: int = 1, temperature: float = 1.0):
        log_probs = F.log_softmax(self.marginal_space_group_logits, axis=-1)
        log_probs = log_probs.unsqueeze(0).expand([batch_size, -1])

        dist = Categorical(logits=log_probs / temperature)
        sample = dist.sample([1]).squeeze(0)

        sample_log_probs = paddle.take_along_axis(
            log_probs, sample.unsqueeze(-1), axis=-1
        ).squeeze(-1)
        return sample, sample_log_probs

    def log_prob(self, space_group_indices):
        normed_logits = F.log_softmax(self.marginal_space_group_logits, axis=-1)
        return normed_logits[space_group_indices]

@dataclasses.dataclass
class CrystalSamplerConfig:
    diffusion_model_config: EquivariantDiffusionModelConfig
    lattice_model_config: LatticeSamplerConfig
    transformer_config: WyckoffElementTransformerConfig
    lattice_length_noise: Optional[float] = 0.0
    lattice_angle_noise: Optional[float] = 0.0
    space_group_grad_weight: Optional[float] = 1.0
    lattice_grad_weight: Optional[float] = 1.0
    wyckoff_element_grad_weight: Optional[float] = 1.0
    frac_coord_grad_weight: Optional[float] = 1.0

class CrystalSampler(nn.Layer):
    """
    Full crystal sampler combining all submodules.
    """
    def __init__(self, config: CrystalSamplerConfig):
        super().__init__()
        self.config = config
        self.atom_coord_diffusion_model = EquivariantDiffusionModel(
            config.diffusion_model_config,
        )
        self.space_group_sampler = SpaceGroupSampler()
        self.lattice_sampler = TelescopingDiscreteLatticeSampler(
            config.lattice_model_config,
        )
        self.wyckoff_and_element_sampler = WyckoffElementTransformer(
            config.transformer_config,
        )

    @paddle.no_grad()
    def sample_crystal(
        self,
        batch_size: int,
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        space_group_numbers=None,
        lattice_parameters=None,
        wyckoff_element_data=None,
    ) -> List[ASUCrystal]:
        """
        Full crystal sampling pipeline.
        """

        if space_group_numbers is None:
            space_group_indices, space_group_log_prob = (
                self.sample_and_log_prob_space_group(batch_size, temperature)
            )
        else:
            assert space_group_numbers.shape[0] == batch_size
            space_group_indices = space_group_numbers - 1
            space_group_log_prob = paddle.zeros_like(space_group_indices)


        if lattice_parameters is None:
            lattice_lengths, lattice_angles, lattice_log_prob = (
                self.sample_and_log_prob_lattice_parameters(space_group_indices)
            )
        else:
            lattice_lengths = lattice_parameters[:, :3]
            lattice_angles = lattice_parameters[:, 3:]
            lattice_log_prob = paddle.zeros([lattice_lengths.shape[0]])

        lattice_matrices = lattice_params_to_matrix_paddle(lattice_lengths, lattice_angles)


        if wyckoff_element_data is None:
            (
                element_indices,
                wyckoff_indices,
                n_asu_atoms_per_xtal,
                elements_log_prob,
                wyckoffs_log_prob,
                termination_log_prob,
            ) = self.sample_and_log_prob_elements_and_wyckoffs(
                lattice_lengths, lattice_angles, space_group_indices, temperature
            )
        else:
            element_indices = wyckoff_element_data["element_indices"]
            wyckoff_indices = wyckoff_element_data["wyckoff_indices"]
            n_asu_atoms_per_xtal = wyckoff_element_data["n_asu_atoms_per_xtal"]


        frac_coords = self.sample_frac_coords(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr=diffusion_snr,
            max_step_size=1e6,
        )


        asu_crystals = []
        atom_offsets = paddle.concat(
            [
                paddle.cumsum(n_asu_atoms_per_xtal, axis=0) - n_asu_atoms_per_xtal,
                n_asu_atoms_per_xtal.sum().unsqueeze(0),
            ],
            axis=0,
        )
        for i in range(n_asu_atoms_per_xtal.shape[0]):
            asu_crystals.append(
                ASUCrystal(
                    space_group_number=1 + space_group_indices[i],
                    conventional_lattice_lengths=lattice_lengths[i],
                    conventional_lattice_angles=lattice_angles[i],
                    element_indices=element_indices[atom_offsets[i]:atom_offsets[i + 1]],
                    wyckoff_indices=wyckoff_indices[atom_offsets[i]:atom_offsets[i + 1]],
                    conventional_frac_coords=frac_coords[atom_offsets[i]:atom_offsets[i + 1]],
                )
            )
        return asu_crystals

    def sample_and_log_prob_space_group(self, batch_size, temperature=1.0):
        return self.space_group_sampler.sample_and_log_prob(batch_size, temperature)

    def sample_and_log_prob_lattice_parameters(self, space_group_indices):
        lengths, angles, log_probs, regularizer = self.lattice_sampler(space_group_indices)
        return lengths, angles, log_probs

    def sample_and_log_prob_elements_and_wyckoffs(
        self, lattice_lengths, lattice_angles, space_group_indices, temperature=1.0
    ):
        return self.wyckoff_and_element_sampler.sample_and_log_prob(
            lattice_lengths, lattice_angles, space_group_indices, temperature
        )

    def sample_frac_coords(
        self,
        wyckoff_indices,
        element_indices,
        space_group_indices,
        n_asu_atoms_per_xtal,
        lattice_matrices,
        lattice_lengths,
        lattice_angles,
        snr=0.4,
        max_step_size=1e6,
    ):
        trajectory = self.atom_coord_diffusion_model.sample(
            wyckoff_indices,
            element_indices,
            space_group_indices,
            n_asu_atoms_per_xtal,
            lattice_matrices,
            lattice_lengths,
            lattice_angles,
            snr,
            max_step_size,
        )
        return trajectory[0]
