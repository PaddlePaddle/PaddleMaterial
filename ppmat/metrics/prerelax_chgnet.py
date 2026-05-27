# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-relaxation module using ppmat's built-in CHGNet for energy prediction.

Provides structure relaxation and energy calculation needed for the
Stability metric in S.U.N. evaluation. Uses ppmat's CHGNet model
(Paddle implementation) -- no external PyTorch dependencies required.
"""

import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid import cost at module level
_chgnet_model = None
_graph_converter = None


def _load_chgnet(model_name: str = "chgnet_mptrj"):
    """Lazy-load CHGNet model and graph converter."""
    global _chgnet_model, _graph_converter
    if _chgnet_model is not None:
        return _chgnet_model, _graph_converter

    from ppmat.models import build_model_from_name
    from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter

    _graph_converter = CHGNetGraphConverter(
        atom_graph_cutoff=6.0, bond_graph_cutoff=3.0
    )
    model, _ = build_model_from_name(model_name)
    model.eval()
    _chgnet_model = model
    return _chgnet_model, _graph_converter


def predict_energy(
    structures: List[Optional[Structure]],
    model_name: str = "chgnet_mptrj",
    batch_size: int = 1,
) -> List[Optional[float]]:
    """Predict energy per atom for each structure using CHGNet.

    Args:
        structures: List of pymatgen Structures (None for invalid).
        model_name: CHGNet model name in ppmat registry.
        batch_size: Prediction batch size.

    Returns:
        List of energy per atom (eV/atom), None for invalid structures.
    """
    model, converter = _load_chgnet(model_name)
    energies = []

    for struct in tqdm(structures, desc="Predicting energies (CHGNet)"):
        if struct is None:
            energies.append(None)
            continue
        try:
            graph = converter(struct)
            pred = model.predict(graph)
            e_per_atom = float(np.array(pred["energy_per_atom"]).flatten()[0])
            energies.append(e_per_atom)
        except Exception as e:
            logger.debug(f"Energy prediction failed: {e}")
            energies.append(None)

    return energies


def relax_structure(
    structure: Structure,
    model_name: str = "chgnet_mptrj",
    steps: int = 200,
    fmax: float = 0.05,
    step_size: float = 0.02,
) -> Tuple[Optional[Structure], Optional[float], int]:
    """Relax a crystal structure using gradient-based optimization with CHGNet.

    Performs iterative structure relaxation by computing forces from CHGNet
    energy predictions and updating atomic positions accordingly.

    Args:
        structure: pymatgen Structure to relax.
        model_name: CHGNet model name in ppmat registry.
        steps: Maximum number of relaxation steps.
        fmax: Force convergence threshold (eV/A).
        step_size: Position update step size.

    Returns:
        Tuple of (relaxed_structure, energy_per_atom, n_steps).
        None values if relaxation fails.
    """
    model, converter = _load_chgnet(model_name)

    try:
        import paddle

        current_struct = structure.copy()
        converged_step = steps

        for step in range(steps):
            graph = converter(current_struct)
            pred = model.predict(graph)
            e_per_atom = float(np.array(pred["energy_per_atom"]).flatten()[0])

            forces = np.array(pred["force"])
            max_force = np.max(np.abs(forces))

            if max_force < fmax:
                converged_step = step + 1
                break

            # Simple gradient descent on positions
            frac_coords = current_struct.frac_coords
            lattice_matrix = current_struct.lattice.matrix

            # Force in Cartesian -> displacement in fractional
            cart_displacements = -step_size * forces / (max_force + 1e-8)
            frac_displacements = cart_displacements @ np.linalg.inv(lattice_matrix)

            new_frac = frac_coords + frac_displacements
            # Wrap to [0, 1)
            new_frac = new_frac % 1.0

            current_struct = Structure(
                current_struct.lattice,
                current_struct.species,
                new_frac,
                coords_are_cartesian=False,
            )

        # Final energy
        graph = converter(current_struct)
        pred = model.predict(graph)
        final_energy = float(np.array(pred["energy_per_atom"]).flatten()[0])

        return current_struct, final_energy, converged_step

    except Exception as e:
        logger.debug(f"Relaxation failed: {e}")
        return None, None, 0


def predict_energies_with_relaxation(
    structures: List[Optional[Structure]],
    model_name: str = "chgnet_mptrj",
    steps: int = 200,
    fmax: float = 0.05,
    relax: bool = True,
) -> Dict[str, List]:
    """Predict energies for structures, optionally with pre-relaxation.

    This is the main entry point for S.U.N. Stability computation.

    Args:
        structures: List of pymatgen Structures (None for invalid).
        model_name: CHGNet model name in ppmat registry.
        steps: Maximum relaxation steps.
        fmax: Force convergence threshold (eV/A).
        relax: Whether to relax structures before energy evaluation.

    Returns:
        Dict with:
            - 'energy_gen': energy per atom before relaxation
            - 'energy_relaxed': energy per atom after relaxation (or same as gen)
            - 'relaxed_structures': relaxed Structures (or originals)
            - 'num_sites': number of atoms per structure
    """
    energy_gen = []
    energy_relaxed = []
    relaxed_structures = []
    num_sites = []

    for struct in tqdm(
        structures,
        desc=f"{'Relaxing' if relax else 'Evaluating'} structures (CHGNet)",
    ):
        if struct is None:
            energy_gen.append(None)
            energy_relaxed.append(None)
            relaxed_structures.append(None)
            num_sites.append(None)
            continue

        try:
            n_sites = struct.num_sites
            model, converter = _load_chgnet(model_name)

            # Initial energy
            graph = converter(struct)
            pred = model.predict(graph)
            e_gen = float(np.array(pred["energy_per_atom"]).flatten()[0]) * n_sites

            if relax:
                relaxed, e_relax, n_steps = relax_structure(
                    struct, model_name=model_name, steps=steps, fmax=fmax
                )
                if relaxed is not None:
                    energy_gen.append(e_gen / n_sites)
                    energy_relaxed.append(e_relax)
                    relaxed_structures.append(relaxed)
                    num_sites.append(n_sites)
                else:
                    energy_gen.append(e_gen / n_sites)
                    energy_relaxed.append(e_gen / n_sites)
                    relaxed_structures.append(struct)
                    num_sites.append(n_sites)
            else:
                energy_gen.append(e_gen / n_sites)
                energy_relaxed.append(e_gen / n_sites)
                relaxed_structures.append(struct)
                num_sites.append(n_sites)

        except Exception as e:
            logger.debug(f"Structure evaluation failed: {e}")
            energy_gen.append(None)
            energy_relaxed.append(None)
            relaxed_structures.append(None)
            num_sites.append(None)

    return {
        "energy_gen": energy_gen,
        "energy_relaxed": energy_relaxed,
        "relaxed_structures": relaxed_structures,
        "num_sites": num_sites,
    }
