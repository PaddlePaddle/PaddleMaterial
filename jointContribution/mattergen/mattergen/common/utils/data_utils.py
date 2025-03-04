from functools import lru_cache

import numpy as np
import paddle
from pymatgen.core import Element

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.ocp_graph_utils import radius_graph_pbc as rgp
from paddle_utils import *  # noqa
from paddle_utils import dim2perm

EPSILON = 1e-05
radius_graph_pbc_ocp = rgp


@lru_cache
def get_atomic_number(symbol: str) -> int:
    return Element(symbol).Z


@lru_cache
def get_element_symbol(Z: int) -> str:
    return str(Element.from_Z(Z=Z))


def abs_cap(val: float, max_abs_val: float = 1.0) -> float:
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trigonometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5
    a8d1079/pymatgen/util/num.py#L15 # noqa
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5
    a8d1079/pymatgen/core/lattice.py#L311 # noqa
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


def lattice_params_to_matrix_paddle(
    lengths: paddle.Tensor, angles: paddle.Tensor, eps: float = 0.0
) -> paddle.Tensor:
    """Batched paddle version to compute lattice matrix from params.

    lengths: paddle.Tensor of shape (N, 3), unit A
    angles: paddle.Tensor of shape (N, 3), unit degree
    """
    coses = paddle.clip(x=paddle.cos(x=paddle.deg2rad(x=angles)), min=-1.0, max=1.0)
    sins = (1 - coses**2).sqrt()
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(x=val, min=-1.0 + eps, max=1.0 - eps)
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
            -lengths[:, 1] * sins[:, 0] * val,
            lengths[:, 1] * sins[:, 0] * (1 - val**2).sqrt(),
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


def lattice_matrix_to_params_paddle(
    matrix: paddle.Tensor, eps: float = 0.0
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Convert a batch of lattice matrices into their corresponding unit cell vector
    lengths and angles.

    Args:
        matrix (paddle.Tensor, [B, 3, 3]): The batch of lattice matrices.

    Returns:
        tuple[paddle.Tensor], ([B, 3], [B, 3]): tuple whose first element is the
        lengths of the unit cell vectors, and the second one gives the angles between
        the vectors.
    """
    assert len(tuple(matrix.shape)) == 3
    lengths = matrix.norm(p=2, axis=-1)
    ix_j = paddle.to_tensor(data=[1, 2, 0], dtype="int64", place=matrix.place)
    ix_k = paddle.to_tensor(data=[2, 0, 1], dtype="int64", place=matrix.place)
    cos_angles = paddle.nn.functional.cosine_similarity(
        x1=matrix[:, ix_j], x2=matrix[:, ix_k], axis=-1
    ).clip(min=-1 + eps, max=1 - eps)
    if len(tuple(matrix.shape)) == 2:
        cos_angles = cos_angles.squeeze(axis=0)
        lengths = lengths.squeeze(axis=0)
    return lengths, paddle.acos(x=cos_angles) * 180.0 / np.pi


def lattice_matrix_to_params(
    matrix: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
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


def frac_to_cart_coords(
    frac_coords: paddle.Tensor,
    lengths: paddle.Tensor,
    angles: paddle.Tensor,
    num_atoms: paddle.Tensor,
) -> paddle.Tensor:
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    return frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice)


def cart_to_frac_coords(
    cart_coords: paddle.Tensor,
    lengths: paddle.Tensor,
    angles: paddle.Tensor,
    num_atoms: paddle.Tensor,
) -> paddle.Tensor:
    lattice = lattice_params_to_matrix_paddle(lengths, angles)
    return cart_to_frac_coords_with_lattice(cart_coords, num_atoms, lattice)


def frac_to_cart_coords_with_lattice(
    frac_coords: paddle.Tensor, num_atoms: paddle.Tensor, lattice: paddle.Tensor
) -> paddle.Tensor:
    lattice_nodes = paddle.repeat_interleave(x=lattice, repeats=num_atoms, axis=0)
    pos = paddle.einsum("bi,bij->bj", frac_coords, lattice_nodes)
    return pos


def cart_to_frac_coords_with_lattice(
    cart_coords: paddle.Tensor, num_atoms: paddle.Tensor, lattice: paddle.Tensor
) -> paddle.Tensor:
    inv_lattice = paddle.linalg.pinv(x=lattice)
    inv_lattice_nodes = paddle.repeat_interleave(x=inv_lattice, repeats=num_atoms, axis=0)  # noqa
    frac_coords = paddle.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def get_pbc_distances(
    coords: paddle.Tensor,
    edge_index: paddle.Tensor,
    lattice: paddle.Tensor,
    to_jimages: paddle.Tensor,
    num_atoms: paddle.Tensor,
    num_bonds: paddle.Tensor,
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> paddle.Tensor:
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = paddle.repeat_interleave(x=lattice, repeats=num_atoms, axis=0)
        pos = paddle.einsum("bi,bij->bj", coords, lattice_nodes)
    j_index, i_index = edge_index
    distance_vectors = pos[j_index] - pos[i_index]
    lattice_edges = paddle.repeat_interleave(x=lattice, repeats=num_bonds, axis=0)
    offsets = paddle.einsum("bi,bij->bj", to_jimages.astype(dtype="float32"), lattice_edges)  # noqa
    distance_vectors += offsets
    distances = distance_vectors.norm(axis=-1)
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out


def radius_graph_pbc(
    cart_coords: paddle.Tensor,
    lattice: paddle.Tensor,
    num_atoms: paddle.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = 10,
    topk_per_pair: (paddle.Tensor | None) = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)

        Keyword arguments
        -----------------
        cart_cords.shape=[Ntotal, 3] -- concatenate all atoms over all crystals
        lattice.shape=[Ncrystal, 3, 3]
        num_atoms.shape=[Ncrystal]
        max_cell_images_per_dim -- constrain the max. number of cell images per
                                dimension in event that infinitesimal angles between
                                lattice vectors are encountered.

    WARNING: It is possible (and has been observed) that for rare cases when periodic
    atom images are on or close to the cut off radius boundary, doing these operations
    in 32 bit floating point can lead to atoms being spuriously considered within or
    outside of the cut off radius. This can lead to invariance of the neighbour list
    under global translation of all atoms in the unit cell. For the rare cases where
    this was observed, switching to 64 bit precision solved the issue. Since all graph
    embeddings should taper messages from neighbours to zero at the cut off radius,
    the effect of these errors in 32-bit should be negligible in practice.
    """
    assert topk_per_pair is None, "non None values of topk_per_pair is not supported"
    edge_index, unit_cell, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=paddle.to_tensor(data=[True, True, True], dtype="float32")
        .to("bool")
        .to(cart_coords.place),
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )
    return edge_index, unit_cell, num_neighbors_image


class StandardScalerTorch(paddle.nn.Layer):
    """Normalizes the targets of a dataset."""

    def __init__(
        self,
        means: (paddle.Tensor | None) = None,
        stds: (paddle.Tensor | None) = None,
        stats_dim: tuple[int] = (1,),
    ):
        super().__init__()
        self.register_buffer(
            name="means",
            tensor=paddle.atleast_1d(means)
            if means is not None
            else paddle.empty(shape=stats_dim),  # noqa
        )
        self.register_buffer(
            name="stds",
            tensor=paddle.atleast_1d(stds)
            if stds is not None
            else paddle.empty(shape=stats_dim),  # noqa
        )

    @property
    def device(self) -> (paddle.CPUPlace, paddle.CUDAPlace, str):
        return self.means.place

    def fit(self, X: paddle.Tensor):
        means: paddle.Tensor = paddle.atleast_1d(
            paddle.nanmean(x=X, axis=0).to(self.device)
        )  # noqa
        stds: paddle.Tensor = paddle.atleast_1d(
            paddle_nanstd(X, dim=0, unbiased=False).to(self.device) + EPSILON
        )
        assert tuple(means.shape) == tuple(
            self.means.shape
        ), f"Mean shape mismatch: {tuple(means.shape)} != {tuple(self.means.shape)}"
        assert tuple(stds.shape) == tuple(
            self.stds.shape
        ), f"Std shape mismatch: {tuple(stds.shape)} != {tuple(self.stds.shape)}"
        self.means = means
        self.stds = stds

    def transform(self, X: paddle.Tensor) -> paddle.Tensor:
        assert self.means is not None and self.stds is not None
        return (X - self.means) / self.stds

    def inverse_transform(self, X: paddle.Tensor) -> paddle.Tensor:
        assert self.means is not None and self.stds is not None
        return X * self.stds + self.means

    def match_device(self, X: paddle.Tensor) -> paddle.Tensor:
        assert self.means.size > 0 and self.stds.size > 0
        if self.means.place != X.place:
            self.means = self.means.to(X.place)
            self.stds = self.stds.to(X.place)

    def copy(self) -> "StandardScalerTorch":
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        return self.transform(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(means: {self.means.tolist() if self.means is not None else None}, stds: {self.stds.tolist() if self.stds is not None else None})"  # noqa


def paddle_nanstd(x: paddle.Tensor, dim: int, unbiased: bool) -> paddle.Tensor:
    data_is_present = paddle.all(
        x=paddle.reshape(
            x=paddle.logical_not(x=paddle.isnan(x=x)), shape=(tuple(x.shape)[0], -1)
        ),  # noqa
        axis=1,
    )
    return paddle.std(x=x[data_is_present], axis=dim, unbiased=unbiased)


def compute_lattice_polar_decomposition(lattice_matrix: paddle.Tensor) -> paddle.Tensor:
    # if lattice_matrix.device.type == "cuda":
    #     try:
    #         W, S, V_transp = paddle.linalg.svd(full_matrices=True, x=lattice_matrix)
    #     except: # torch._C._LinAlgError: todo: fix this
    #         W, S, V_transp = paddle.linalg.svd(
    #             full_matrices=True, x=lattice_matrix.to("cpu")
    #         )
    #         W = W.to(lattice_matrix.device.type)
    #         S = S.to(lattice_matrix.device.type)
    #         V_transp = V_transp.to(lattice_matrix.device.type)
    # else:
    #     W, S, V_transp = paddle.linalg.svd(full_matrices=True, x=lattice_matrix)

    W, S, V_transp = paddle.linalg.svd(full_matrices=True, x=lattice_matrix)
    S_square = paddle.diag_embed(input=S)
    V = V_transp.transpose(perm=dim2perm(V_transp.ndim, 1, 2))
    U = W @ V_transp
    P = V @ S_square @ V_transp
    P_prime = U @ P @ U.transpose(perm=dim2perm(U.ndim, 1, 2))
    symm_lattice_matrix = P_prime
    return symm_lattice_matrix


def create_chem_graph_from_composition(
    target_composition_dict: dict[str, float]
) -> ChemGraph:  # noqa
    atomic_numbers = []
    for element_name, number_of_atoms in target_composition_dict.items():
        atomic_numbers += [Element(element_name).Z] * int(number_of_atoms)
    return ChemGraph(
        atomic_numbers=paddle.to_tensor(data=atomic_numbers, dtype="int64"),
        num_atoms=paddle.to_tensor(data=[len(atomic_numbers)], dtype="int64"),
        cell=paddle.eye(num_rows=3, dtype="float32").reshape(1, 3, 3),
        pos=paddle.zeros(shape=(len(atomic_numbers), 3), dtype="float32"),
    )
