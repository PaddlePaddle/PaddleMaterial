import sys


from copy import deepcopy
from itertools import chain, permutations
from typing import List, Tuple

import paddle
from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.common.tests.testutils import get_mp_20_debug_batch
from mattergen.common.utils.data_utils import (
    cart_to_frac_coords_with_lattice, frac_to_cart_coords_with_lattice,
    lattice_matrix_to_params_paddle, lattice_params_to_matrix_paddle)
from mattergen.common.utils.eval_utils import make_structure
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from paddle_utils import *
from pymatgen.core.structure import Structure
from scipy.spatial.transform import Rotation
from paddle_geometric.data import Batch, Data


def get_model(**kwargs) -> GemNetT:
    return GemNetT(
        atom_embedding=AtomEmbedding(emb_size=4),
        num_targets=1,
        latent_dim=4,
        num_radial=4,
        num_blocks=1,
        emb_size_atom=4,
        emb_size_edge=4,
        emb_size_trip=4,
        emb_size_bil_trip=4,
        otf_graph=True,
        scale_file=f"{MODELS_PROJECT_ROOT}/common/gemnet/gemnet-dT.json",
        **kwargs,
    )


def structures_list_to_batch(structures: List[Structure]) -> Batch:
    return Batch.from_data_list(
        [
            Data(
                angles=paddle.to_tensor(data=s.lattice.angles, dtype="float32")[None],
                lengths=paddle.to_tensor(data=s.lattice.lengths, dtype="float32")[None],
                frac_coords=paddle.to_tensor(data=s.frac_coords).astype(
                    dtype="float32"
                ),
                atom_types=paddle.to_tensor(data=s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in structures
        ]
    )


def reformat_batch(
    batch: Batch,
) -> Tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
]:
    return (
        None,
        batch.frac_coords,
        batch.atom_types,
        batch.num_atoms,
        batch.batch,
        batch.lengths,
        batch.angles,
    )


def get_cubic_data(supercell: Tuple[int, int, int]) -> Tuple[Tuple, Tuple]:
    normal_structures = [
        Structure(
            lattice=[[2, 0, 0], [0, 3.1, 0], [0, 0, 2.9]],
            coords=[[0, 0, 0]],
            species="C",
        ),
        Structure(
            lattice=[[3.1, 0, 0], [0, 2, 0], [0, 0, 4]],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            species=["C", "C"],
        ),
    ]
    normal_structures = list(
        chain.from_iterable([deepcopy(normal_structures) for _ in range(32)])
    )
    supercell_structures = deepcopy(normal_structures)
    for s in supercell_structures:
        s.make_supercell(supercell)
    normal_batch = structures_list_to_batch(structures=normal_structures)
    supercell_batch = structures_list_to_batch(structures=supercell_structures)
    return reformat_batch(batch=normal_batch), reformat_batch(batch=supercell_batch)


def test_lattice_score_scale_invariance():
    cutoff = 5.0
    max_neighbors = 1000
    paddle.seed(seed=495606849)
    model = get_model(
        max_neighbors=max_neighbors,
        cutoff=cutoff,
        regress_stress=True,
        max_cell_images_per_dim=20,
    )
    model.eval()
    batch = get_mp_20_debug_batch()
    batch = Batch.from_data_list(batch.to_data_list()[:10])
    supercell_structures = [
        make_structure(
            d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords
        )
        for d in batch.to_data_list()
    ]
    for s in supercell_structures:
        s.make_supercell((2, 2, 2))
    supercell_batch = Batch.from_data_list(
        [
            Data(
                angles=paddle.to_tensor(data=s.lattice.angles, dtype="float32")[None],
                lengths=paddle.to_tensor(data=s.lattice.lengths, dtype="float32")[None],
                frac_coords=paddle.to_tensor(data=s.frac_coords).astype(
                    dtype="float32"
                ),
                atom_types=paddle.to_tensor(data=s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in supercell_structures
        ]
    )
    with paddle.no_grad():
        out_normal_cells = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )
        out_supercells = model.forward(
            None,
            supercell_batch.frac_coords,
            supercell_batch.atom_types,
            supercell_batch.num_atoms,
            supercell_batch.batch,
            supercell_batch.lengths,
            supercell_batch.angles,
        )
    assert out_normal_cells.stress is not None
    assert out_supercells.stress is not None
    all_close = paddle.allclose(
        x=out_normal_cells.stress, y=out_supercells.stress, atol=1e-05
    ).item()
    assert all_close, (out_normal_cells.stress - out_supercells.stress).abs().max()


def test_nonconservative_lattice_score_translation_invariance():
    model = get_model(
        max_neighbors=200, cutoff=5.0, regress_stress=True, max_cell_images_per_dim=10
    )
    model.eval()
    batch = get_mp_20_debug_batch()
    structures = [
        make_structure(
            d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords
        )
        for d in batch.to_data_list()
    ]
    translated_batch = Batch.from_data_list(
        [
            Data(
                angles=paddle.to_tensor(data=s.lattice.angles, dtype="float32")[None],
                lengths=paddle.to_tensor(data=s.lattice.lengths, dtype="float32")[None],
                frac_coords=(
                    paddle.to_tensor(data=s.frac_coords).astype(dtype="float32")
                    + paddle.rand(shape=[1, 3])
                )
                % 1.0,
                atom_types=paddle.to_tensor(data=s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in structures
        ]
    )
    with paddle.no_grad():
        out_normal_cells = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )
        out_translated = model.forward(
            None,
            translated_batch.frac_coords,
            translated_batch.atom_types,
            translated_batch.num_atoms,
            translated_batch.batch,
            translated_batch.lengths,
            translated_batch.angles,
        )
    assert paddle.allclose(
        atol=0.0001, rtol=0.0001, x=out_normal_cells.stress, y=out_translated.stress
    ).item(), ""


def test_lattice_parameterization_invariance():
    """
    Tests whether our model's predicted score behaves as expected when choosing a different unit cell.
    """
    cutoff = 5.0
    max_neighbors = 200
    paddle.seed(seed=2)
    model = get_model(
        max_neighbors=max_neighbors,
        cutoff=cutoff,
        regress_stress=True,
        max_cell_images_per_dim=30,
    )
    model.eval()
    batch = get_mp_20_debug_batch()
    structures = [
        make_structure(
            d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords
        )
        for d in batch.to_data_list()
    ]
    lattice_matrices = lattice_params_to_matrix_paddle(batch.lengths, batch.angles)
    lattice_matrix_changed = lattice_matrices.clone()
    combs = paddle.to_tensor(data=list(permutations(range(3), 2)))
    lattice_vector_combine_ixs = paddle.randint(
        low=0, high=len(combs), shape=(tuple(lattice_matrices.shape)[0],)
    )
    combs_sel = combs[lattice_vector_combine_ixs]
    change_matrix = (
        paddle.eye(num_rows=3)[None].expand_as(y=lattice_matrices).clone().contiguous()
    )
    change_matrix[
        range(tuple(combs_sel.shape)[0]), combs_sel[:, 0], combs_sel[:, 1]
    ] = 3
    lattice_matrix_changed = (
        lattice_matrices.transpose(perm=dim2perm(lattice_matrices.ndim, 1, 2))
        @ change_matrix
    ).transpose(
        perm=dim2perm(
            (
                lattice_matrices.transpose(perm=dim2perm(lattice_matrices.ndim, 1, 2))
                @ change_matrix
            ).ndim,
            1,
            2,
        )
    )
    new_frac_coords = cart_to_frac_coords_with_lattice(
        frac_to_cart_coords_with_lattice(
            batch.frac_coords, batch.num_atoms, lattice_matrices
        ),
        batch.num_atoms,
        lattice_matrix_changed,
    )
    updated_batch = batch.clone()
    new_lengths, new_angles = lattice_matrix_to_params_paddle(lattice_matrix_changed)
    updated_batch.frac_coords = new_frac_coords
    updated_batch.lengths = new_lengths
    updated_batch.angles = new_angles
    structures_perm = [
        make_structure(
            d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords
        )
        for d in updated_batch.to_data_list()
    ]
    close = [
        paddle.allclose(
            x=paddle.to_tensor(data=structures_perm[ix].distance_matrix),
            y=paddle.to_tensor(data=structures[ix].distance_matrix),
            atol=0.001,
        ).item()
        for ix in range(len(structures))
    ]
    assert all(close)
    with paddle.no_grad():
        out_normal_cells = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=lattice_matrices,
        )
        out_updated_batch = model.forward(
            None,
            updated_batch.frac_coords,
            updated_batch.atom_types,
            updated_batch.num_atoms,
            updated_batch.batch,
            lattice=lattice_matrix_changed,
        )
    assert not paddle.allclose(
        x=change_matrix.inverse() @ out_normal_cells.stress,
        y=out_updated_batch.stress,
        atol=0.001,
    ).item()
    assert not paddle.allclose(
        x=out_normal_cells.stress, y=out_updated_batch.stress, atol=0.001
    ).item()


def test_symmetric_lattice_score():
    model = get_model(
        max_neighbors=20, cutoff=7.0, regress_stress=True, max_cell_images_per_dim=20
    )
    model.eval()
    batch = get_mp_20_debug_batch()
    with paddle.no_grad():
        model_out = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )
    assert model_out.stress is not None
    assert paddle.allclose(
        x=model_out.stress, y=model_out.stress.transpose(1, 2), atol=1e-05
    ).item()


def test_rotation_invariance():
    model = get_model(
        max_neighbors=1000, cutoff=5.0, regress_stress=True, max_cell_images_per_dim=10
    )
    batch = get_mp_20_debug_batch()
    lattices = lattice_params_to_matrix_paddle(batch.lengths, batch.angles)
    with paddle.no_grad():
        model_out = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=lattices,
        )
    rotation_matrix = paddle.to_tensor(
        data=Rotation.random().as_matrix(), dtype="float32"
    )
    rotated_lattices = lattices @ rotation_matrix
    with paddle.no_grad():
        model_out_rotated = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=rotated_lattices,
        )
    forces = model_out.forces
    forces_rotated = model_out_rotated.forces
    stress = model_out.stress
    stress_rotated = model_out_rotated.stress
    assert paddle.allclose(
        x=forces @ rotation_matrix, y=forces_rotated, atol=0.001
    ).item()
    assert paddle.allclose(
        x=rotation_matrix.T @ stress @ rotation_matrix, y=stress_rotated, atol=0.001
    ).item()
