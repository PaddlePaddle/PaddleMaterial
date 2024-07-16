import copy
import faulthandler
import itertools
import os
import sys
from functools import partial

import networkx as nx
import numpy as np
import paddle
import pandas as pd
from p_tqdm import p_umap
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
from utils import paddle_aux

faulthandler.enable()
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]
EPSILON = 1e-05
chemical_symbols = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)


def segment_coo(data):
    return data


def segment_csr(data):
    return data


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt="cif")
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def refine_spacegroup(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_conventional_standard_structure()
    space_group = spga.get_space_group_number()
    crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return crystal, space_group


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.0
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """ """
    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None,
                x_diff_weight=-1,
                porous_adjustment=False,
                search_cutoff=10,
            )
            crystal_graph = StructureGraph.with_local_env_strategy(
                crystal, crystalNN_tmp
            )
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    assert np.allclose(
        crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    )
    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))
    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = tuple(atom_types.shape)[0]
    return (
        frac_coords,
        atom_types,
        lengths,
        angles,
        edge_indices,
        to_jimages,
        num_atoms,
    )


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)
    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_paddle(lengths, angles):
    """Batched paddle version to compute lattice matrix from params.

    lengths: paddle.Tensor of shape (N, 3), unit A
    angles: paddle.Tensor of shape (N, 3), unit degree
    """
    angles_r = paddle.deg2rad(x=angles)
    coses = paddle.cos(x=angles_r)
    sins = paddle.sin(x=angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(x=val, min=-1.0, max=1.0)
    gamma_star = paddle.acos(x=val)
    vector_a = paddle.stack(
        x=[
            lengths[:, 0] * sins[:, 1],
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 0] * coses[:, 1],
        ],
        axis=1,
    )
    vector_b = paddle.stack(
        x=[
            -lengths[:, 1] * sins[:, 0] * paddle.cos(x=gamma_star),
            lengths[:, 1] * sins[:, 0] * paddle.sin(x=gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        axis=1,
    )
    vector_c = paddle.stack(
        x=[
            paddle.zeros(shape=lengths.shape[0]),
            paddle.zeros(shape=lengths.shape[0]),
            lengths[:, 2],
        ],
        axis=1,
    )
    return paddle.stack(x=[vector_a, vector_b, vector_c], axis=1)


def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = paddle.unbind(input=batch_lattice, axis=1)
    return paddle.abs(
        x=paddle.einsum(
            "bi,bi->b", vector_a, paddle.cross(x=vector_b, y=vector_c, axis=1)
        )
    )


def lengths_angles_to_volume(lengths, angles):
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    return compute_volume(lattice)


def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()
    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def lattices_to_params_shape(lattices):
    lengths = paddle.sqrt(x=paddle.sum(x=lattices**2, axis=-1))
    angles = paddle.zeros_like(x=lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = paddle.clip(
            x=paddle.sum(x=lattices[..., j, :] * lattices[..., k, :], axis=-1)
            / (lengths[..., j] * lengths[..., k]),
            min=-1.0,
            max=1.0,
        )
    angles = paddle.acos(x=angles) * 180.0 / np.pi
    return lengths, angles


def frac_to_cart_coords(
    frac_coords, lengths, angles, num_atoms, regularized=True, lattices=None
):
    if regularized:
        frac_coords = frac_coords % 1.0
    if lattices is None:
        lattices = lattice_params_to_matrix_paddle(lengths, angles)
    lattice_nodes = paddle.repeat_interleave(x=lattices, repeats=num_atoms, axis=0)
    pos = paddle.einsum("bi,bij->bj", frac_coords, lattice_nodes)
    return pos


def cart_to_frac_coords(cart_coords, lengths, angles, num_atoms, regularized=True):
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    inv_lattice = paddle.linalg.pinv(x=lattice)
    inv_lattice_nodes = paddle.repeat_interleave(
        x=inv_lattice, repeats=num_atoms, axis=0
    )
    frac_coords = paddle.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    if regularized:
        frac_coords = frac_coords % 1.0
    return frac_coords


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
    lattices=None,
):
    if lattices is None:
        lattices = lattice_params_to_matrix_paddle(lengths, angles)
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = paddle.repeat_interleave(x=lattices, repeats=num_atoms, axis=0)
        pos = paddle.einsum("bi,bij->bj", coords, lattice_nodes)
    j_index, i_index = edge_index
    distance_vectors = pos[j_index] - pos[i_index]
    lattice_edges = paddle.repeat_interleave(x=lattices, repeats=num_bonds, axis=0)
    offsets = paddle.einsum(
        "bi,bij->bj", to_jimages.astype(dtype="float32"), lattice_edges
    )
    distance_vectors += offsets
    distances = distance_vectors.norm(axis=-1)
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out


def radius_graph_pbc_wrapper(data, radius, max_num_neighbors_threshold, device):
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms
    )
    return radius_graph_pbc(
        cart_coords,
        data.lengths,
        data.angles,
        data.num_atoms,
        radius,
        max_num_neighbors_threshold,
        device,
    )


def repeat_blocks(
    sizes, repeats, continuous_indexing=True, start_idx=0, block_inc=0, repeat_inc=0
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)
    sizes_nonzero = sizes > 0
    if not paddle.all(x=sizes_nonzero):
        assert block_inc == 0
        sizes = paddle.masked_select(x=sizes, mask=sizes_nonzero)
        if isinstance(repeats, paddle.Tensor):
            repeats = paddle.masked_select(x=repeats, mask=sizes_nonzero)
        if isinstance(repeat_inc, paddle.Tensor):
            repeat_inc = paddle.masked_select(x=repeat_inc, mask=sizes_nonzero)
    if isinstance(repeats, paddle.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = paddle.ones(shape=[1], dtype=sizes.dtype)
            zero = paddle.zeros(shape=[1], dtype=sizes.dtype)
            sizes = paddle.concat(x=(one, sizes))
            repeats = paddle.concat(x=(one, repeats))
            if isinstance(block_inc, paddle.Tensor):
                block_inc = paddle.concat(x=(zero, block_inc))
            if isinstance(repeat_inc, paddle.Tensor):
                repeat_inc = paddle.concat(x=(zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False
    r1 = paddle.repeat_interleave(x=paddle.arange(end=len(sizes)), repeats=repeats)
    N = (sizes * repeats).sum()
    id_ar = paddle.ones(shape=N, dtype="int64")
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(axis=0)
    insert_val = (1 - sizes)[r1[:-1]]
    if isinstance(repeats, paddle.Tensor) and paddle.any(x=repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = paddle.concat(
            x=(paddle.zeros(shape=[1], dtype=sizes.dtype), diffs.cumsum(axis=0))
        )
        if continuous_indexing:
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")
        if isinstance(block_inc, paddle.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        insert_val[idx] += block_inc
    if isinstance(repeat_inc, paddle.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, paddle.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc
    if isinstance(repeats, paddle.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner
    id_ar[insert_index] = insert_val
    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1
    id_ar[0] += start_idx
    res = id_ar.cumsum(axis=0)
    return res


def radius_graph_pbc(
    pos,
    lengths,
    angles,
    natoms,
    radius,
    max_num_neighbors_threshold,
    device,
    lattices=None,
):
    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_paddle(lengths, angles)
    else:
        cell = lattices
    atom_pos = pos
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).astype(dtype="int64")
    index_offset = paddle.cumsum(x=num_atoms_per_image, axis=0) - num_atoms_per_image
    index_offset_expand = paddle.repeat_interleave(
        x=index_offset, repeats=num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = paddle.repeat_interleave(
        x=num_atoms_per_image, repeats=num_atoms_per_image_sqr
    )
    num_atom_pairs = paddle.sum(x=num_atoms_per_image_sqr)
    index_sqr_offset = (
        paddle.cumsum(x=num_atoms_per_image_sqr, axis=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(
        x=index_sqr_offset, repeats=num_atoms_per_image_sqr
    )
    atom_count_sqr = paddle.arange(end=num_atom_pairs) - index_sqr_offset
    index1 = (
        paddle.floor(
            paddle.divide(
                x=atom_count_sqr, y=paddle.to_tensor(num_atoms_per_image_expand)
            )
        )
        + index_offset_expand
    )
    index2 = atom_count_sqr % num_atoms_per_image_expand + index_offset_expand
    pos1 = paddle.index_select(x=atom_pos, axis=0, index=index1)
    pos2 = paddle.index_select(x=atom_pos, axis=0, index=index2)
    cross_a2a3 = paddle.cross(x=cell[:, 1], y=cell[:, 2], axis=-1)
    cell_vol = paddle.sum(x=cell[:, 0] * cross_a2a3, axis=-1, keepdim=True)
    inv_min_dist_a1 = paddle.linalg.norm(x=cross_a2a3 / cell_vol, p=2, axis=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1, 1)
    cross_a3a1 = paddle.cross(x=cell[:, 2], y=cell[:, 0], axis=-1)
    inv_min_dist_a2 = paddle.linalg.norm(x=cross_a3a1 / cell_vol, p=2, axis=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1, 1)
    cross_a1a2 = paddle.cross(x=cell[:, 0], y=cell[:, 1], axis=-1)
    inv_min_dist_a3 = paddle.linalg.norm(x=cross_a1a2 / cell_vol, p=2, axis=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1, 1)
    max_rep = paddle.ones(shape=[3], dtype="int64")
    min_dist = paddle.concat(x=[min_dist_a1, min_dist_a2, min_dist_a3], axis=-1)
    unit_cell_all = []
    num_cells_all = []
    cells_per_dim = [
        paddle.arange(start=-rep, end=rep + 1, dtype="float32") for rep in max_rep
    ]
    unit_cell = paddle.concat(
        x=[_.reshape(-1, 1) for _ in paddle.meshgrid(cells_per_dim)], axis=-1
    )
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    x = unit_cell
    perm_1 = list(range(x.ndim))
    perm_1[0] = 1
    perm_1[1] = 0
    unit_cell = paddle.transpose(x=x, perm=perm_1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(shape=[batch_size, -1, -1])
    x = cell
    perm_2 = list(range(x.ndim))
    perm_2[1] = 2
    perm_2[2] = 1
    data_cell = paddle.transpose(x=x, perm=perm_2)
    pbc_offsets = paddle.bmm(x=data_cell, y=unit_cell_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=num_atoms_per_image_sqr, axis=0
    )
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_sqr = paddle.sum(x=(pos1 - pos2) ** 2, axis=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)
    radius_real = min_dist.min(dim=-1)[0] + 0.01
    radius_real = paddle.repeat_interleave(
        x=radius_real, repeats=num_atoms_per_image_sqr * num_cells
    )
    mask_within_radius = paddle.less_equal(
        x=atom_distance_sqr, y=paddle.to_tensor(radius_real * radius_real)
    )
    mask_not_same = paddle.greater_than(x=atom_distance_sqr, y=paddle.to_tensor(0.0001))
    mask = paddle.logical_and(x=mask_within_radius, y=mask_not_same)
    index1 = paddle.masked_select(x=index1, mask=mask)
    index2 = paddle.masked_select(x=index2, mask=mask)
    unit_cell = paddle.masked_select(
        x=unit_cell_per_atom.view(-1, 3), mask=mask.view(-1, 1).expand(shape=[-1, 3])
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = paddle.masked_select(x=atom_distance_sqr, mask=mask)
    if max_num_neighbors_threshold is not None:
        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )
        if not paddle.all(x=mask_num_neighbors):
            index1 = paddle.masked_select(x=index1, mask=mask_num_neighbors)
            index2 = paddle.masked_select(x=index2, mask=mask_num_neighbors)
            unit_cell = paddle.masked_select(
                x=unit_cell.view(-1, 3),
                mask=mask_num_neighbors.view(-1, 1).expand(shape=[-1, 3]),
            )
            unit_cell = unit_cell.view(-1, 3)
    else:
        ones = paddle.ones(shape=[1], dtype=index1.dtype).expand_as(y=index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())
        image_indptr = paddle.zeros(shape=tuple(natoms.shape)[0] + 1, dtype="int64")
        image_indptr[1:] = paddle.cumsum(x=natoms, axis=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)
    edge_index = paddle.stack(x=(index2, index1))
    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.place
    num_atoms = natoms.sum()
    ones = paddle.ones(shape=[1], dtype=index.dtype).expand_as(y=index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clip(max=max_num_neighbors_threshold)
    image_indptr = paddle.zeros(shape=tuple(natoms.shape)[0] + 1, dtype="int64")
    image_indptr[1:] = paddle.cumsum(x=natoms, axis=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = paddle.to_tensor(
            data=[True], dtype=bool, place=device
        ).expand_as(y=index)
        return mask_num_neighbors, num_neighbors_image
    distance_sort = paddle.full(
        shape=[num_atoms * max_num_neighbors], fill_value=np.inf
    )
    index_neighbor_offset = paddle.cumsum(x=num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        x=index_neighbor_offset, repeats=num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + paddle.arange(end=len(index))
        - index_neighbor_offset_expand
    )
    distance_sort.scatter_(index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)
    distance_sort, index_sort = paddle.sort(x=distance_sort, axis=1), paddle.argsort(
        x=distance_sort, axis=1
    )
    distance_real_cutoff = (
        distance_sort[:, max_num_neighbors_threshold]
        .reshape(-1, 1)
        .expand(shape=[-1, max_num_neighbors])
        + 0.01
    )
    mask_distance = distance_sort < distance_real_cutoff
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        shape=[-1, max_num_neighbors]
    )
    mask_finite = paddle.isfinite(x=distance_sort)
    index_sort = paddle.masked_select(x=index_sort, mask=mask_finite & mask_distance)
    num_neighbor_per_node = (mask_finite & mask_distance).sum(axis=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)
    mask_num_neighbors = paddle.zeros(shape=len(index), dtype=bool)
    mask_num_neighbors.index_fill_(axis=0, index=index_sort, value=True)
    return mask_num_neighbors, num_neighbors_image


def radius_graph_pbc_(
    cart_coords,
    lengths,
    angles,
    num_atoms,
    radius,
    max_num_neighbors_threshold,
    device,
    topk_per_pair=None,
):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)
    atom_pos = cart_coords
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).astype(dtype="int64")
    index_offset = paddle.cumsum(x=num_atoms_per_image, axis=0) - num_atoms_per_image
    index_offset_expand = paddle.repeat_interleave(
        x=index_offset, repeats=num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = paddle.repeat_interleave(
        x=num_atoms_per_image, repeats=num_atoms_per_image_sqr
    )
    num_atom_pairs = paddle.sum(x=num_atoms_per_image_sqr)
    index_sqr_offset = (
        paddle.cumsum(x=num_atoms_per_image_sqr, axis=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(
        x=index_sqr_offset, repeats=num_atoms_per_image_sqr
    )
    atom_count_sqr = paddle.arange(end=num_atom_pairs) - index_sqr_offset
    index1 = (atom_count_sqr // num_atoms_per_image_expand).astype(
        dtype="int64"
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand).astype(
        dtype="int64"
    ) + index_offset_expand
    pos1 = paddle.index_select(x=atom_pos, axis=0, index=index1)
    pos2 = paddle.index_select(x=atom_pos, axis=0, index=index2)
    unit_cell = paddle.to_tensor(data=OFFSET_LIST, place=device).astype(dtype="float32")
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    x = unit_cell
    perm_3 = list(range(x.ndim))
    perm_3[0] = 1
    perm_3[1] = 0
    unit_cell = paddle.transpose(x=x, perm=perm_3)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(shape=[batch_size, -1, -1])
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    x = lattice
    perm_4 = list(range(x.ndim))
    perm_4[1] = 2
    perm_4[2] = 1
    data_cell = paddle.transpose(x=x, perm=perm_4)
    pbc_offsets = paddle.bmm(x=data_cell, y=unit_cell_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=num_atoms_per_image_sqr, axis=0
    )
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_sqr = paddle.sum(x=(pos1 - pos2) ** 2, axis=1)
    if topk_per_pair is not None:
        assert topk_per_pair.shape[0] == num_atom_pairs
        atom_distance_sqr_sort_index = paddle.argsort(x=atom_distance_sqr, axis=1)
        assert tuple(atom_distance_sqr_sort_index.shape) == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index
            + paddle.arange(end=num_atom_pairs)[:, None] * num_cells
        ).view(-1)
        topk_mask = paddle.arange(end=num_cells)[None, :] < topk_per_pair[:, None]
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(mask=topk_mask)
        topk_mask = paddle.zeros(shape=num_atom_pairs * num_cells)
        topk_mask.put_along_axis_(axis=0, indices=topk_indices, values=1.0)
        topk_mask = topk_mask.astype(dtype="bool")
    atom_distance_sqr = atom_distance_sqr.view(-1)
    mask_within_radius = paddle.less_equal(
        x=atom_distance_sqr, y=paddle.to_tensor(radius * radius)
    )
    mask_not_same = paddle.greater_than(x=atom_distance_sqr, y=paddle.to_tensor(0.0001))
    mask = paddle.logical_and(x=mask_within_radius, y=mask_not_same)
    index1 = paddle.masked_select(x=index1, mask=mask)
    index2 = paddle.masked_select(x=index2, mask=mask)
    unit_cell = paddle.masked_select(
        x=unit_cell_per_atom.view(-1, 3), mask=mask.view(-1, 1).expand(shape=[-1, 3])
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = paddle.masked_select(x=topk_mask, mask=mask)
    num_neighbors = paddle.zeros(shape=len(cart_coords))
    num_neighbors.index_add_(axis=0, index=index1, value=paddle.ones(shape=len(index1)))
    num_neighbors = num_neighbors.astype(dtype="int64")
    max_num_neighbors = paddle.max(x=num_neighbors).astype(dtype="int64")
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = paddle.zeros(shape=len(cart_coords) + 1).astype(dtype="int64")
    _natoms = paddle.zeros(shape=tuple(num_atoms.shape)[0] + 1).astype(dtype="int64")
    _num_neighbors[1:] = paddle.cumsum(x=_max_neighbors, axis=0)
    _natoms[1:] = paddle.cumsum(x=num_atoms, axis=0)
    num_neighbors_image = _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return paddle.stack(x=(index2, index1)), unit_cell, num_neighbors_image
        else:
            return (
                paddle.stack(x=(index2, index1)),
                unit_cell,
                num_neighbors_image,
                topk_mask,
            )
    atom_distance_sqr = paddle.masked_select(x=atom_distance_sqr, mask=mask)
    distance_sort = paddle.zeros(shape=len(cart_coords) * max_num_neighbors).fill_(
        value=radius * radius + 1.0
    )
    index_neighbor_offset = paddle.cumsum(x=num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = paddle.repeat_interleave(
        x=index_neighbor_offset, repeats=num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + paddle.arange(end=len(index1))
        - index_neighbor_offset_expand
    )
    distance_sort.scatter_(index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)
    distance_sort, index_sort = paddle.sort(x=distance_sort, axis=1), paddle.argsort(
        x=distance_sort, axis=1
    )
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        shape=[-1, max_num_neighbors_threshold]
    )
    mask_within_radius = paddle.less_equal(
        x=distance_sort, y=paddle.to_tensor(radius * radius)
    )
    index_sort = paddle.masked_select(x=index_sort, mask=mask_within_radius)
    mask_num_neighbors = paddle.zeros(shape=len(index1)).astype(dtype="bool")
    mask_num_neighbors.index_fill_(axis=0, index=index_sort, value=True)
    index1 = paddle.masked_select(x=index1, mask=mask_num_neighbors)
    index2 = paddle.masked_select(x=index2, mask=mask_num_neighbors)
    unit_cell = paddle.masked_select(
        x=unit_cell.view(-1, 3),
        mask=mask_num_neighbors.view(-1, 1).expand(shape=[-1, 3]),
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = paddle.masked_select(x=topk_mask, mask=mask_num_neighbors)
    edge_index = paddle.stack(x=(index2, index1))
    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def min_distance_sqr_pbc(
    cart_coords1,
    cart_coords2,
    lengths,
    angles,
    num_atoms,
    device,
    return_vector=False,
    return_to_jimages=False,
):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)
    pos1 = cart_coords1
    pos2 = cart_coords2
    unit_cell = paddle.to_tensor(data=OFFSET_LIST, place=device).astype(dtype="float32")
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(cart_coords2), 1, 1)
    x = unit_cell
    perm_5 = list(range(x.ndim))
    perm_5[0] = 1
    perm_5[1] = 0
    unit_cell = paddle.transpose(x=x, perm=perm_5)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(shape=[batch_size, -1, -1])
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    x = lattice
    perm_6 = list(range(x.ndim))
    perm_6[1] = 2
    perm_6[2] = 1
    data_cell = paddle.transpose(x=x, perm=perm_6)
    pbc_offsets = paddle.bmm(x=data_cell, y=unit_cell_batch)
    pbc_offsets_per_atom = paddle.repeat_interleave(
        x=pbc_offsets, repeats=num_atoms, axis=0
    )
    pos1 = pos1.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2.view(-1, 3, 1).expand(shape=[-1, -1, num_cells])
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = paddle.sum(x=atom_distance_vector**2, axis=1)
    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)
    return_list = [min_atom_distance_sqr]
    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])
        min_atom_distance_vector = paddle.take_along_axis(
            arr=atom_distance_vector, axis=2, indices=min_indices
        ).squeeze(axis=-1)
        return_list.append(min_atom_distance_vector)
    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].astype(dtype="int64")
        return_list.append(to_jimages)
    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerPaddle(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = paddle.to_tensor(data=X, dtype="float32")
        self.means = paddle.mean(x=X, axis=0)
        self.stds = paddle.std(x=X, axis=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = paddle.to_tensor(data=X, dtype="float32")
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = paddle.to_tensor(data=X, dtype="float32")
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.place != tensor.place:
            self.means = self.means.to(tensor.place)
            self.stds = self.stds.to(tensor.place)

    def copy(self):
        return StandardScalerPaddle(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(means: {self.means.tolist()}, stds: {self.stds.tolist()})"


def get_scaler_from_data_list(data_list, key):
    targets = paddle.to_tensor(data=[d[key] for d in data_list])
    scaler = StandardScalerPaddle()
    scaler.fit(targets)
    return scaler


def process_one(
    row, niggli, primitive, graph_method, prop_list, use_space_group=False, tol=0.01
):
    crystal_str = row["cif"]
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
    result_dict = {}
    if use_space_group:
        crystal, sym_info = get_symmetry_info(crystal, tol=tol)
        result_dict.update(sym_info)
    else:
        result_dict["spacegroup"] = 1
    graph_arrays = build_crystal_graph(crystal, graph_method)
    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict.update(
        {"mp_id": row["material_id"], "cif": crystal_str, "graph_arrays": graph_arrays}
    )
    result_dict.update(properties)
    return result_dict


def preprocess(
    input_file,
    num_workers,
    niggli,
    primitive,
    graph_method,
    prop_list,
    use_space_group=False,
    tol=0.01,
):
    df = pd.read_csv(input_file)
    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        [use_space_group] * len(df),
        [tol] * len(df),
        num_cpus=num_workers,
    )
    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [
        mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))
    ]
    return ordered_results


def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array["frac_coords"]
        atom_types = crystal_array["atom_types"]
        lengths = crystal_array["lengths"]
        angles = crystal_array["angles"]
        crystal = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {"batch_idx": batch_idx, "graph_arrays": graph_arrays}
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(sorted(unordered_results, key=lambda x: x["batch_idx"]))
    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict["graph_arrays"]
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert tuple(lengths.shape)[0] == tuple(angles.shape)[0] == 3
        assert isinstance(num_atoms, int)
        if lattice_scale_method == "scale_length":
            lengths = lengths / float(num_atoms) ** (1 / 3)
        dict["scaled_lattice"] = np.concatenate([lengths, angles])


def mard(targets, preds):
    """Mean absolute relative difference."""
    assert paddle.all(x=targets > 0.0)
    return paddle.mean(x=paddle.abs(x=targets - preds) / targets)


def batch_accuracy_precision_recall(pred_edge_probs, edge_overlap_mask, num_bonds):
    if pred_edge_probs is None and edge_overlap_mask is None and num_bonds is None:
        return 0.0, 0.0, 0.0
    pred_edges = pred_edge_probs.max(dim=1)[1].astype(dtype="float32")
    target_edges = edge_overlap_mask.astype(dtype="float32")
    start_idx = 0
    accuracies, precisions, recalls = [], [], []
    for num_bond in num_bonds.tolist():
        start_0 = pred_edges.shape[0] + start_idx if start_idx < 0 else start_idx
        pred_edge = (
            paddle.slice(pred_edges, [0], [start_0], [start_0 + num_bond])
            .detach()
            .cpu()
            .numpy()
        )
        start_1 = target_edges.shape[0] + start_idx if start_idx < 0 else start_idx
        target_edge = (
            paddle.slice(target_edges, [0], [start_1], [start_1 + num_bond])
            .detach()
            .cpu()
            .numpy()
        )
        accuracies.append(accuracy_score(target_edge, pred_edge))
        precisions.append(precision_score(target_edge, pred_edge, average="binary"))
        recalls.append(recall_score(target_edge, pred_edge, average="binary"))
        start_idx = start_idx + num_bond
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(
            np.isnan(self.means), np.zeros(tuple(self.means.shape)), self.means
        )
        self.stds = np.where(
            np.isnan(self.stds), np.ones(tuple(self.stds.shape)), self.stds
        )
        self.stds = np.where(self.stds == 0, np.ones(tuple(self.stds.shape)), self.stds)
        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )
        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )
        return transformed_with_none
