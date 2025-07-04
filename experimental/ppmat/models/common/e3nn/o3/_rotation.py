import math

import paddle

from ppmat.models.common.e3nn.paddle_utils import *


def rand_matrix(*shape, requires_grad=False, dtype=None, device=None):
    """random rotation matrix

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 3, 3)`
    """
    R = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    out_7 = R.detach()
    out_7.stop_gradient = not requires_grad
    return out_7


def identity_angles(*shape, requires_grad=False, dtype=None, device=None):
    """angles of the identity rotation

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    abc = paddle.zeros(shape=[3, *shape], dtype=dtype)
    out_8 = abc[0]
    out_8.stop_gradient = not requires_grad
    out_9 = abc[1]
    out_9.stop_gradient = not requires_grad
    out_10 = abc[2]
    out_10.stop_gradient = not requires_grad
    return out_8, out_9, out_10


def rand_angles(*shape, requires_grad=False, dtype=None, device=None):
    """random rotation angles

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    beta : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`

    gamma : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    if dtype is None:
        dtype = paddle.get_default_dtype()
    alpha, gamma = 2 * math.pi * paddle.rand(shape=[2, *shape], dtype=dtype)
    # beta = paddle.rand(shape=shape, dtype=dtype).mul(2.0).sub(1.0).acos()
    beta = paddle.acos(
        2.0 * paddle.rand(shape=shape, dtype=dtype) - 1.0
    )  # modified in 2025 0407 because of error
    out_11 = alpha.detach()
    out_11.stop_gradient = not requires_grad
    alpha = out_11
    out_12 = beta.detach()
    out_12.stop_gradient = not requires_grad
    beta = out_12
    out_13 = gamma.detach()
    out_13.stop_gradient = not requires_grad
    gamma = out_13
    return alpha, beta, gamma


def compose_angles(a1, b1, c1, a2, b2, c2):
    """compose angles

    Computes :math:`(a, b, c)` such that :math:`R(a, b, c) = R(a_1, b_1, c_1) \\circ R(a_2, b_2, c_2)`

    Parameters
    ----------
    a1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    b1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    c1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    a2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    b2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    c2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    a1, b1, c1, a2, b2, c2 = paddle.broadcast_tensors(input=[a1, b1, c1, a2, b2, c2])
    return matrix_to_angles(angles_to_matrix(a1, b1, c1) @ angles_to_matrix(a2, b2, c2))


def inverse_angles(a, b, c):
    """angles of the inverse rotation

    Parameters
    ----------
    a : `torch.Tensor`
        tensor of shape :math:`(...)`

    b : `torch.Tensor`
        tensor of shape :math:`(...)`

    c : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return -c, -b, -a


def identity_quaternion(*shape, requires_grad=False, dtype=None, device=None):
    """quaternion of identity rotation

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 4)`
    """
    q = paddle.zeros(shape=[*shape, 4], dtype=dtype)
    q[..., 0] = 1
    out_14 = q.detach()
    out_14.stop_gradient = not requires_grad
    q = out_14
    return q


def rand_quaternion(*shape, requires_grad=False, dtype=None, device=None):
    """generate random quaternion

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 4)`
    """
    q = angles_to_quaternion(*rand_angles(*shape, dtype=dtype, device=device))
    out_15 = q.detach()
    out_15.stop_gradient = not requires_grad
    q = out_15
    return q


def compose_quaternion(q1, q2):
    """compose two quaternions: :math:`q_1 \\circ q_2`

    Parameters
    ----------
    q1 : `torch.Tensor`
        tensor of shape :math:`(..., 4)`, (applied second)

    q2 : `torch.Tensor`
        tensor of shape :math:`(..., 4)`, (applied first)

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    q1, q2 = paddle.broadcast_tensors(input=[q1, q2])
    return paddle.stack(
        x=[
            q1[..., 0] * q2[..., 0]
            - q1[..., 1] * q2[..., 1]
            - q1[..., 2] * q2[..., 2]
            - q1[..., 3] * q2[..., 3],
            q1[..., 1] * q2[..., 0]
            + q1[..., 0] * q2[..., 1]
            + q1[..., 2] * q2[..., 3]
            - q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2]
            - q1[..., 1] * q2[..., 3]
            + q1[..., 2] * q2[..., 0]
            + q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3]
            + q1[..., 1] * q2[..., 2]
            - q1[..., 2] * q2[..., 1]
            + q1[..., 3] * q2[..., 0],
        ],
        axis=-1,
    )


def inverse_quaternion(q):
    """inverse of a quaternion

    Works only for unit quaternions.

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    q = q.clone()
    q[..., 1:].neg_()
    return q


def rand_axis_angle(*shape, requires_grad=False, dtype=None, device=None):
    """generate random rotation as axis-angle

    Parameters
    ----------
    *shape : int

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape}, 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(\\mathrm{shape})`
    """
    axis, angle = angles_to_axis_angle(*rand_angles(*shape, dtype=dtype, device=device))
    out_16 = axis.detach()
    out_16.stop_gradient = not requires_grad
    axis = out_16
    out_17 = angle.detach()
    out_17.stop_gradient = not requires_grad
    angle = out_17
    return axis, angle


def compose_axis_angle(axis1, angle1, axis2, angle2):
    """compose :math:`(\\vec x_1, \\alpha_1)` with :math:`(\\vec x_2, \\alpha_2)`

    Parameters
    ----------
    axis1 : `torch.Tensor`
        tensor of shape :math:`(..., 3)`, (applied second)

    angle1 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied second)

    axis2 : `torch.Tensor`
        tensor of shape :math:`(..., 3)`, (applied first)

    angle2 : `torch.Tensor`
        tensor of shape :math:`(...)`, (applied first)

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return quaternion_to_axis_angle(
        compose_quaternion(
            axis_angle_to_quaternion(axis1, angle1),
            axis_angle_to_quaternion(axis2, angle2),
        )
    )


def matrix_x(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around X axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(x=angle)
    z = paddle.zeros_like(x=angle)
    return paddle.stack(
        x=[
            paddle.stack(x=[o, z, z], axis=-1),
            paddle.stack(x=[z, c, -s], axis=-1),
            paddle.stack(x=[z, s, c], axis=-1),
        ],
        axis=-2,
    )


def matrix_y(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around Y axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(x=angle)
    z = paddle.zeros_like(x=angle)
    return paddle.stack(
        x=[
            paddle.stack(x=[c, z, s], axis=-1),
            paddle.stack(x=[z, o, z], axis=-1),
            paddle.stack(x=[-s, z, c], axis=-1),
        ],
        axis=-2,
    )


def matrix_z(angle: paddle.Tensor) -> paddle.Tensor:
    """matrix of rotation around Z axis

    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = paddle.ones_like(x=angle)
    z = paddle.zeros_like(x=angle)
    return paddle.stack(
        x=[
            paddle.stack(x=[c, -s, z], axis=-1),
            paddle.stack(x=[s, c, z], axis=-1),
            paddle.stack(x=[z, z, o], axis=-1),
        ],
        axis=-2,
    )


def angles_to_matrix(alpha, beta, gamma):
    """conversion from angles to matrix

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = paddle.broadcast_tensors(input=[alpha, beta, gamma])
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_to_angles(R):
    """conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert paddle.allclose(
        x=paddle.linalg.det(x=R), y=paddle.ones_like(paddle.linalg.det(x=R))
    ).item()
    x = R @ paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=R.dtype)
    a, b = xyz_to_angles(x)
    R = (
        angles_to_matrix(a, b, paddle.zeros_like(x=a)).transpose(
            perm=dim2perm(angles_to_matrix(a, b, paddle.zeros_like(x=a)).ndim, -1, -2)
        )
        @ R
    )
    c = paddle.atan2(x=R[..., 0, 2], y=R[..., 0, 0])
    return a, b, c


def angles_to_quaternion(alpha, beta, gamma):
    """conversion from angles to quaternion

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 4)`
    """
    alpha, beta, gamma = paddle.broadcast_tensors(input=[alpha, beta, gamma])
    qa = axis_angle_to_quaternion(
        paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=alpha.dtype), alpha
    )
    qb = axis_angle_to_quaternion(
        paddle.to_tensor(data=[1.0, 0.0, 0.0], dtype=beta.dtype), beta
    )
    qc = axis_angle_to_quaternion(
        paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=gamma.dtype), gamma
    )
    return compose_quaternion(qa, compose_quaternion(qb, qc))


def matrix_to_quaternion(R):
    """conversion from matrix :math:`R` to quaternion :math:`q`

    Parameters
    ----------
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    return axis_angle_to_quaternion(*matrix_to_axis_angle(R))


def axis_angle_to_quaternion(xyz, angle):
    """convertion from axis-angle to quaternion

    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    """
    xyz, angle = paddle.broadcast_tensors(input=[xyz, angle[..., None]])
    xyz = paddle.nn.functional.normalize(x=xyz, axis=-1)
    c = paddle.cos(x=angle[..., :1] / 2)
    s = paddle.sin(x=angle / 2)
    return paddle.concat(x=[c, xyz * s], axis=-1)


def quaternion_to_axis_angle(q):
    """convertion from quaternion to axis-angle

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    angle = 2 * paddle.acos(x=q[..., 0].clip(min=-1, max=1))
    axis = paddle.nn.functional.normalize(x=q[..., 1:], axis=-1)
    return axis, angle


def matrix_to_axis_angle(R):
    """conversion from matrix to axis-angle

    Parameters
    ----------
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert paddle.allclose(
        x=paddle.linalg.det(x=R), y=paddle.to_tensor(data=1, dtype=R.dtype)
    ).item()
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = paddle.acos(x=tr.sub(1).div(2).clip(min=-1, max=1))
    axis = paddle.stack(
        x=[
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    axis = paddle.nn.functional.normalize(x=axis, axis=-1)
    return axis, angle


def angles_to_axis_angle(alpha, beta, gamma):
    """conversion from angles to axis-angle

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_axis_angle(angles_to_matrix(alpha, beta, gamma))


def axis_angle_to_matrix(axis, angle):
    """conversion from axis-angle to matrix

    Parameters
    ----------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    axis, angle = paddle.broadcast_tensors(input=[axis, angle[..., None]])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, paddle.zeros_like(x=beta))
    Ry = matrix_y(angle[..., 0])
    return R @ Ry @ R.transpose(perm=dim2perm(R.ndim, -2, -1))


def quaternion_to_matrix(q):
    """convertion from quaternion to matrix

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    return axis_angle_to_matrix(*quaternion_to_axis_angle(q))


def quaternion_to_angles(q):
    """convertion from quaternion to angles

    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_angles(quaternion_to_matrix(q))


def axis_angle_to_angles(axis, angle):
    """convertion from axis-angle to angles

    Parameters
    ----------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    angle : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    return matrix_to_angles(axis_angle_to_matrix(axis, angle))


def angles_to_xyz(alpha, beta):
    """convert :math:`(\\alpha, \\beta)` into a point :math:`(x, y, z)` on the sphere

    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Examples
    --------

    >>> angles_to_xyz(torch.tensor(1.7), torch.tensor(0.0)).abs()
    tensor([0., 1., 0.])
    """
    alpha, beta = paddle.broadcast_tensors(input=[alpha, beta])
    x = paddle.sin(x=beta) * paddle.sin(x=alpha)
    y = paddle.cos(x=beta)
    z = paddle.sin(x=beta) * paddle.cos(x=alpha)
    return paddle.stack(x=[x, y, z], axis=-1)


def xyz_to_angles(xyz):
    """convert a point :math:`\\vec r = (x, y, z)` on the sphere into angles :math:`(\\alpha, \\beta)`

    .. math::

        \\vec r = R(\\alpha, \\beta, 0) \\vec e_z


    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xyz = paddle.nn.functional.normalize(x=xyz, p=2, axis=-1)
    xyz = xyz.clip(min=-1, max=1)
    beta = paddle.acos(x=xyz[..., 1])
    alpha = paddle.atan2(x=xyz[..., 0], y=xyz[..., 2])
    return alpha, beta
