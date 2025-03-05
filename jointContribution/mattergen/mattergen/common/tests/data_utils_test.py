from collections import Counter
from itertools import product
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import pytest
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import RotationTransformation

from mattergen.common.tests.testutils import get_mp_20_debug_batch
from mattergen.common.utils import data_utils


def test_lattice_params_matrix():
    a, b, c = 4.0, 3.0, 2.0
    alpha, beta, gamma = 120.0, 90.0, 90.0
    matrix = data_utils.lattice_params_to_matrix(a, b, c, alpha, beta, gamma)
    result = data_utils.lattice_matrix_to_params(matrix)
    assert np.allclose([a, b, c, alpha, beta, gamma], result)


def test_lattice_params_matrix2():
    matrix = [
        [3.966866, 0.0, 2.42900487e-16],
        [-2.42900487e-16, 3.966866, 2.42900487e-16],
        [0.0, 0.0, 5.73442],
    ]
    matrix = np.array(matrix)
    params = data_utils.lattice_matrix_to_params(matrix)
    result = data_utils.lattice_params_to_matrix(*params)
    assert np.allclose(matrix, result)


def test_lattice_params_to_matrix_paddle():
    lengths = np.array([[4.0, 3.0, 2.0], [1, 3, 2]])
    angles = np.array([[120.0, 90.0, 90.0], [57.0, 130.0, 85.0]])
    lengths_and_angles = np.concatenate([lengths, angles], axis=-1)
    matrix0 = data_utils.lattice_params_to_matrix(*lengths_and_angles[0].tolist())
    matrix1 = data_utils.lattice_params_to_matrix(*lengths_and_angles[1].tolist())
    true_matrix = np.stack([matrix0, matrix1], axis=0)
    torch_matrix = data_utils.lattice_params_to_matrix_paddle(
        paddle.to_tensor(data=lengths), paddle.to_tensor(data=angles)
    )
    assert np.allclose(true_matrix, torch_matrix.numpy(), atol=1e-05)


def test_lattice_matrix_to_params_paddle():
    lengths = np.array([[4.0, 3.0, 2.0], [1, 3, 2]])
    angles = np.array([[120.0, 90.0, 90.0], [57.0, 130.0, 85.0]])
    torch_matrix = data_utils.lattice_params_to_matrix_paddle(
        paddle.to_tensor(data=lengths), paddle.to_tensor(data=angles)
    )
    torch_lengths, torch_angles = data_utils.lattice_matrix_to_params_paddle(torch_matrix)
    assert np.allclose(lengths, torch_lengths.numpy(), atol=1e-05)
    assert np.allclose(angles, torch_angles.numpy(), atol=1e-05)


def test_frac_cart_conversion():
    num_atoms = paddle.to_tensor(data=[4, 3, 2, 5], dtype="int64")
    lengths = paddle.rand(shape=[num_atoms.shape[0], 3]) * 4
    angles = paddle.rand(shape=[num_atoms.shape[0], 3]) * 60 + 60
    frac_coords = paddle.rand(shape=[num_atoms.sum(), 3])
    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
    inverted_frac_coords = data_utils.cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)
    assert paddle.allclose(x=frac_coords, y=inverted_frac_coords, atol=1e-05, rtol=0.001).item()


def test_get_pbc_distances():
    frac_coords = paddle.to_tensor(
        data=[[0.2, 0.2, 0.0], [0.6, 0.8, 0.8], [0.2, 0.2, 0.0], [0.6, 0.8, 0.8]],
        dtype="float32",
    )
    edge_index = paddle.to_tensor(data=[[1, 0], [0, 0], [2, 3]], dtype="int64").T
    lengths = paddle.to_tensor(data=[[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]], dtype="float32")
    angles = paddle.to_tensor(data=[[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]], dtype="float32")
    to_jimages = paddle.to_tensor(data=[[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype="int64")
    num_nodes = paddle.to_tensor(data=[2, 2], dtype="int64")
    num_edges = paddle.to_tensor(data=[2, 1], dtype="int64")
    lattice = data_utils.lattice_params_to_matrix_paddle(lengths, angles)
    out = data_utils.get_pbc_distances(
        frac_coords, edge_index, lattice, to_jimages, num_nodes, num_edges
    )
    true_distances = paddle.to_tensor(data=[1.7549928774784245, 1.0, 1.2], dtype="float32")
    assert paddle.allclose(x=true_distances, y=out["distances"]).item()


def test_get_pbc_distances_cart():
    frac_coords = paddle.to_tensor(
        data=[[0.2, 0.2, 0.0], [0.6, 0.8, 0.8], [0.2, 0.2, 0.0], [0.6, 0.8, 0.8]],
        dtype="float32",
    )
    edge_index = paddle.to_tensor(data=[[1, 0], [0, 0], [2, 3]], dtype="int64").T
    lengths = paddle.to_tensor(data=[[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]], dtype="float32")
    angles = paddle.to_tensor(data=[[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]], dtype="float32")
    to_jimages = paddle.to_tensor(data=[[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype="int64")
    num_nodes = paddle.to_tensor(data=[2, 2], dtype="int64")
    num_edges = paddle.to_tensor(data=[2, 1], dtype="int64")
    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_nodes)
    lattice = data_utils.lattice_params_to_matrix_paddle(lengths, angles)
    out = data_utils.get_pbc_distances(
        cart_coords,
        edge_index,
        lattice,
        to_jimages,
        num_nodes,
        num_edges,
        coord_is_cart=True,
    )
    true_distances = paddle.to_tensor(data=[1.7549928774784245, 1.0, 1.2], dtype="float32")
    assert paddle.allclose(x=true_distances, y=out["distances"]).item()


@pytest.mark.parametrize(
    "max_radius,max_neighbors",
    [(5.5964, 100), (5.6, 100), (100.0, 100), (7.0, 14), (7.0, 15)],
)
def test_pbc_graph_translation_invariant(max_radius: float, max_neighbors: int):
    lengths = paddle.to_tensor(data=[4.0, 4.0, 4.0])[None, :]
    angles = paddle.to_tensor(data=[90.0, 90.0, 90.0])[None, :]
    frac_coords = paddle.to_tensor(data=[[0.2, 0.0, 0.0], [0.9927, 0.5, 0.5]])
    num_atoms = paddle.to_tensor(data=[2])
    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
    translation = paddle.to_tensor(data=[[0.05, 0.1, -0.04]])
    cart_coords_translated = cart_coords + translation
    frac_coords_translated = data_utils.cart_to_frac_coords(
        cart_coords_translated, lengths, angles, num_atoms
    )
    cart_coords_translated = data_utils.frac_to_cart_coords(
        frac_coords_translated, lengths, angles, num_atoms
    )
    lattice = data_utils.lattice_params_to_matrix_paddle(lengths=lengths, angles=angles)
    coords = {"original": cart_coords, "translated": cart_coords_translated}
    output: Dict[str, Dict[str, Dict[int, paddle.Tensor]]] = {
        coord: {
            output_type: {
                max_cells: {c: paddle.to_tensor(data=[0]) for c in coords.keys()}
                for max_cells in [1, 2]
            }
            for output_type in ["edge_index", "to_jimages", "num_bonds"]
        }
        for coord in coords.keys()
    }
    for coord in coords.keys():
        for max_cells in [2, 3]:
            (
                output[coord]["edge_index"][max_cells],
                output[coord]["to_jimages"][max_cells],
                output[coord]["num_bonds"][max_cells],
            ) = data_utils.radius_graph_pbc(
                cart_coords=coords[coord],
                lattice=lattice,
                num_atoms=num_atoms,
                radius=max_radius,
                max_num_neighbors_threshold=max_neighbors,
                max_cell_images_per_dim=max_cells,
            )
    for max_cell in [2, 3]:
        counter1 = Counter(
            [tuple(x) for x in output["original"]["edge_index"][max_cell].t().tolist()]
        )
        counter2 = Counter(
            [tuple(x) for x in output["translated"]["edge_index"][max_cell].t().tolist()]
        )
        assert counter1 == counter2
        assert paddle.equal_all(
            x=output["original"]["num_bonds"][max_cell],
            y=output["translated"]["num_bonds"][max_cell],
        ).item()


def get_random_rotation(
    n_random: int, n_atom: Optional[int] = None
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    lattice = paddle.normal(mean=0, std=1, shape=(3, 3))
    if n_atom is None:
        number_atoms = paddle.randint(low=1, high=17, shape=(1,))
    else:
        number_atoms = paddle.to_tensor(data=[n_atom])
    frac_coord = paddle.rand(shape=(number_atoms[0], 3))
    structure = Structure(
        species=["C" for _ in range(number_atoms[0])],
        lattice=lattice.numpy(),
        coords=frac_coord.numpy(),
    )
    random_axes = np.random.choice([0, 1], size=(n_random, 3))
    for ii, axis in enumerate(random_axes):
        if np.allclose(axis, [0, 0, 0]):
            random_axes[ii] = [1, 0, 0]
    random_angles = np.random.rand(n_random) * 90
    structures = [
        RotationTransformation(axis=axis, angle=angle).apply_transformation(structure)
        for axis, angle in zip(random_axes, random_angles)
    ]
    lattices = paddle.to_tensor(
        data=np.asarray([s.lattice._matrix for s in structures]), dtype="float32"
    )
    frac_coords = (
        paddle.to_tensor(data=structure.frac_coords, dtype="float32")
        .expand(shape=(n_random, number_atoms[0], 3))
        .flatten(stop_axis=1)
    )
    num_atoms = paddle.to_tensor(data=[structure.frac_coords.shape[0]]).expand(shape=n_random)
    return (
        data_utils.frac_to_cart_coords_with_lattice(
            frac_coords=frac_coords, lattice=lattices, num_atoms=num_atoms
        ),
        lattices,
        num_atoms,
    )


def get_random_translation(n_random: int, n_atom: Optional[int] = None):
    lattice = paddle.normal(mean=0, std=0.5, shape=(3, 3))
    lattice[paddle.eye(num_rows=3).astype(dtype="uint8")] = paddle.normal(mean=0, std=1, shape=(3,))
    if n_atom is None:
        number_atoms = paddle.randint(low=1, high=17, shape=(1,))
    else:
        number_atoms = paddle.to_tensor(data=[n_atom])
    frac_coord = paddle.rand(shape=(number_atoms[0], 3))
    natoms = paddle.to_tensor(data=[tuple(frac_coord.shape)[0]]).expand(shape=n_random)
    multiple_lattices = lattice.expand(shape=[n_random, 3, 3])
    translation = paddle.rand(shape=(n_random, 1, 3)).expand(
        shape=(n_random, tuple(frac_coord.shape)[0], 3)
    )
    new_frac_coord = (
        frac_coord.expand(shape=(n_random, tuple(frac_coord.shape)[0], 3)) + translation
    )
    new_frac_coord = new_frac_coord % 1
    new_frac_coord = new_frac_coord.flatten(stop_axis=1)
    new_cart_coord = data_utils.frac_to_cart_coords_with_lattice(
        frac_coords=new_frac_coord, lattice=multiple_lattices, num_atoms=natoms
    )
    return new_cart_coord, multiple_lattices, natoms


def check_invariance(
    max_radius: float,
    max_cell_images_per_dim: int,
    cart: paddle.Tensor,
    lattice: paddle.Tensor,
    num_atoms: paddle.Tensor,
):
    max_neighbors = 100
    edges, _, num_bonds = data_utils.radius_graph_pbc(
        cart_coords=cart,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=max_radius,
        max_num_neighbors_threshold=max_neighbors,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )
    edges = edges.numpy()
    start_from = np.asarray(np.hstack((np.zeros(1), np.cumsum(num_bonds))), dtype=int)
    counters = []
    for ii in range(len(start_from) - 1):
        bond_subset = edges.T[start_from[ii] : start_from[ii + 1]]
        offset = num_atoms[0] * ii
        bond_subset -= offset.numpy()
        counters.append(Counter([tuple(x) for x in bond_subset]))
    count_counters = Counter([f"{c}" for c in counters])
    assert len(set([len(c) for c in counters])) == 1, set([len(c) for c in counters])
    assert len(count_counters) == 1, count_counters


@pytest.mark.parametrize(
    "max_radius, max_cell_images",
    [(3.0, 1), (7.0, 1), (3.0, 2), (7.0, 2), (3.0, 3), (7.0, 3)],
)
def test_rotation_invariance(max_radius: float, max_cell_images: int):
    cart, lattice, num_atoms = get_random_rotation(n_random=10)
    check_invariance(
        max_radius=max_radius,
        max_cell_images_per_dim=max_cell_images,
        cart=cart,
        lattice=lattice,
        num_atoms=num_atoms,
    )


@pytest.mark.parametrize("max_radius, max_cell_images", [(3.0, 10), (7.0, 20)])
def test_translation_invariance(max_radius: float, max_cell_images: int):
    cart, lattice, num_atoms = get_random_translation(n_random=10)
    check_invariance(
        max_radius=max_radius,
        max_cell_images_per_dim=max_cell_images,
        cart=cart,
        lattice=lattice,
        num_atoms=num_atoms,
    )


def get_distances_pymatgen(structure: Structure, rcut: float) -> np.ndarray:
    neigh = structure.get_all_neighbors(r=rcut, include_image=True)
    dist = sorted(
        np.asarray([n.nn_distance for _atom in neigh for n in _atom if n.nn_distance > 1e-12])
    )
    return np.asarray(dist)


def get_distance_pytorch(structure: Structure, rcut: float) -> np.ndarray:
    cart_coords = paddle.to_tensor(data=structure.cart_coords, dtype="float32")
    lattice = paddle.to_tensor(data=[structure.lattice._matrix], dtype="float32")
    num_atoms = paddle.to_tensor(data=[tuple(cart_coords.shape)[0]], dtype="int32")
    edges, images, num_bonds = data_utils.radius_graph_pbc(
        cart_coords=cart_coords,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=rcut,
        max_num_neighbors_threshold=100000,
        max_cell_images_per_dim=100,
    )
    distances = data_utils.get_pbc_distances(
        coords=cart_coords,
        edge_index=edges,
        lattice=lattice,
        to_jimages=images,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        coord_is_cart=True,
    )
    return np.asarray(sorted(distances["distances"].numpy()))


def get_distances_numpy(structure: Structure, rcut: float, dtype) -> np.ndarray:
    frac_coord = np.asarray(structure.frac_coords, dtype=dtype)
    lattice = np.asarray(structure.lattice._matrix, dtype=dtype)
    natm = tuple(frac_coord.shape)[0]
    cart_coord_0_0_0 = np.asarray(np.einsum("ni, ix->nx", frac_coord, lattice), dtype=dtype)
    max_cell = 100
    images = np.asarray(
        list(
            product(
                range(-max_cell, max_cell + 1),
                range(-max_cell, max_cell + 1),
                range(-max_cell, max_cell + 1),
            )
        ),
        dtype=dtype,
    )
    nimages = tuple(images.shape)[0]
    images = np.tile(np.expand_dims(images, 1), (1, natm, 1))
    periodic_frac_coord = np.tile(frac_coord, (nimages, 1, 1)) + images
    periodic_frac_coord = np.tile(np.expand_dims(periodic_frac_coord, 0), (natm, 1, 1, 1))
    assert periodic_frac_coord.dtype == dtype
    cart_coords_tiled = np.tile(np.expand_dims(cart_coord_0_0_0, (1, 2)), (1, nimages, natm, 1))
    periodic_cart_coord = np.einsum("nimk,kx->nimx", periodic_frac_coord, lattice)
    assert periodic_cart_coord.dtype == dtype
    all_distances = np.linalg.norm(cart_coords_tiled - periodic_cart_coord, axis=-1)
    all_distances = all_distances.flatten()
    all_distances = all_distances[
        np.where(np.logical_and(all_distances <= rcut, all_distances > 1e-12))[0]
    ]
    assert all_distances.dtype == dtype
    return np.asarray(sorted(all_distances))


@pytest.mark.parametrize(
    "natom, rcut", [(1, 1.0), (2, 1.0), (3, 1.0), (1, 2.0), (2, 2.0), (3, 2.0)]
)
def test_rdf(natom: int, rcut: float):
    structure = Structure(
        species=["C" for _ in range(natom)],
        coords=np.random.uniform(size=(natom, 3)),
        lattice=np.random.normal(size=(3, 3)),
    )
    assert np.allclose(
        get_distances_numpy(structure=structure, rcut=rcut, dtype=np.float32),
        get_distance_pytorch(structure=structure, rcut=rcut),
    )


def test_polar_decomposition():
    batch = get_mp_20_debug_batch()
    lattices = data_utils.lattice_params_to_matrix_paddle(batch.lengths, batch.angles)
    polar_decomposition = data_utils.compute_lattice_polar_decomposition(lattices)
    symm_lengths, symm_angles = data_utils.lattice_matrix_to_params_paddle(polar_decomposition)
    assert paddle.allclose(x=symm_lengths, y=batch.lengths, atol=0.001).item()
    assert paddle.allclose(x=symm_angles, y=batch.angles, atol=0.001).item()
    assert paddle.allclose(
        x=paddle.linalg.det(polar_decomposition).abs(),
        y=paddle.linalg.det(lattices).abs(),
        atol=0.001,
    ).item()


def test_paddle_nanstd():
    x = paddle.to_tensor(data=[1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0])
    assert data_utils.paddle_nanstd(x=x, dim=0, unbiased=False).item() == np.nanstd(x.numpy())
