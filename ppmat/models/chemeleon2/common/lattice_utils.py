import paddle
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.crystal import lattices_to_params_shape


def lattice_params_to_matrix(lengths, angles):
    return lattice_params_to_matrix_paddle(lengths, angles)


def matrix_to_lattice_params(lattices):
    return lattices_to_params_shape(lattices)


def frac_to_cart_coords(frac_coords, lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    return paddle.einsum('ij,jk->ik', frac_coords, lattice.squeeze(0))


def cart_to_frac_coords(cart_coords, lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    inv_lattice = paddle.inverse(lattice)
    return paddle.einsum('ij,jk->ik', cart_coords, inv_lattice.squeeze(0))


def get_pbc_distances(
    coords1,
    coords2,
    lattice,
    num_atoms=None,
    return_offsets=False,
):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    
    if coords1.shape != coords2.shape:
        raise ValueError("coords1 and coords2 must have the same shape")
    
    diff = coords2 - coords1
    
    diff_frac = cart_to_frac_coords(diff, lattice)
    
    diff_frac = diff_frac - paddle.round(diff_frac)
    
    diff_cart = frac_to_cart_coords(diff_frac, lattice)
    
    distances = paddle.norm(diff_cart, axis=-1)
    
    if return_offsets:
        offsets = paddle.round(cart_to_frac_coords(coords2 - coords1, lattice))
        return distances, offsets
    
    return distances


def lattice_vector_to_volume(lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    
    a = lattice[:, 0, :]
    b = lattice[:, 1, :]
    c = lattice[:, 2, :]
    
    volume = paddle.abs(paddle.sum(a * paddle.cross(b, c), axis=-1))
    
    return volume
