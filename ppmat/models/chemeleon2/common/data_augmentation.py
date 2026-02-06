import paddle


def apply_augmentation(batch, translate=False, rotate=False):
    if not translate and not rotate:
        return batch
    
    batch_aug = batch.clone()
    
    if translate:
        batch_aug = _augmentation_translate(batch_aug)
    
    if rotate:
        batch_aug = _augmentation_rotate(batch_aug)
    
    return batch_aug


def _augmentation_translate(batch):
    lengths_mean = batch.lengths.mean(axis=0)
    lengths_std = batch.lengths.std(axis=0, unbiased=False)
    
    random_translate = paddle.normal(
        mean=paddle.abs(lengths_mean),
        std=paddle.maximum(paddle.abs(lengths_std), paddle.to_tensor([1e-8]))
    ) / 2
    
    cart_coords_aug = batch.cart_coords + random_translate
    
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    frac_coords_aug = paddle.einsum('bi,bij->bj', cart_coords_aug, cell_per_node_inv)
    frac_coords_aug = frac_coords_aug % 1.0
    
    batch.cart_coords = cart_coords_aug
    batch.frac_coords = frac_coords_aug
    
    return batch


def _augmentation_rotate(batch):
    rot_mat = _random_rotation_matrix(validate=True)
    
    cart_coords_aug = paddle.matmul(batch.cart_coords, rot_mat.T)
    lattices_aug = paddle.matmul(batch.lattices, rot_mat.T)
    
    batch.cart_coords = cart_coords_aug
    batch.lattices = lattices_aug
    
    return batch


def apply_noise(batch, ratio=0.1, corruption_scale=0.1):
    if ratio <= 0:
        return batch
    
    batch_noise = batch.clone()
    
    total_num_atoms = batch_noise.num_nodes
    noise_num_atoms = int(total_num_atoms * ratio)
    
    noise_atom_types = batch_noise.atom_types.clone()
    noise_indices = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_atom_types[noise_indices] = 0
    
    noise_cart_coords = batch_noise.cart_coords.clone()
    noise_indices = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_cart_coords[noise_indices] += paddle.randn([noise_num_atoms, 3]) * corruption_scale
    
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    noise_frac_coords = paddle.einsum('bi,bij->bj', noise_cart_coords, cell_per_node_inv)
    noise_frac_coords = noise_frac_coords % 1.0
    
    batch_noise.atom_types = noise_atom_types
    batch_noise.cart_coords = noise_cart_coords
    batch_noise.frac_coords = noise_frac_coords
    
    return batch_noise


def _random_rotation_matrix(validate=False):
    q = paddle.rand([4])
    q = q / paddle.norm(q)
    
    rot_mat = paddle.to_tensor([
        [
            1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
            2 * q[1] * q[2] - 2 * q[0] * q[3],
            2 * q[1] * q[3] + 2 * q[0] * q[2],
        ],
        [
            2 * q[1] * q[2] + 2 * q[0] * q[3],
            1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
            2 * q[2] * q[3] - 2 * q[0] * q[1],
        ],
        [
            2 * q[1] * q[3] - 2 * q[0] * q[2],
            2 * q[2] * q[3] + 2 * q[0] * q[1],
            1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
        ],
    ], dtype='float32')
    
    if validate:
        identity = paddle.matmul(rot_mat, rot_mat.T)
        eye = paddle.eye(3)
        assert paddle.allclose(identity, eye, atol=1e-5, rtol=1e-5), "Not a rotation matrix."
    
    return rot_mat
