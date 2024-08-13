import paddle


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
