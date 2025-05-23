import paddle

"""Core functions of :math:`SO(3)`
"""
import functools
import math

from ppmat.models.common.e3nn.util import explicit_default_types


def su2_generators(j) -> paddle.Tensor:
    m = paddle.arange(start=-j, end=j, dtype="float32")
    raising = paddle.diag(x=-paddle.sqrt(x=j * (j + 1) - m * (m + 1)), offset=-1)
    m = paddle.arange(start=-j + 1, end=j + 1, dtype="float32")
    lowering = paddle.diag(x=paddle.sqrt(x=j * (j + 1) - m * (m - 1)), offset=1)
    m = paddle.arange(start=-j, end=j + 1, dtype="float32")
    raising = paddle.cast(raising, dtype="complex64")
    lowering = paddle.cast(lowering, dtype="complex64")
    return paddle.stack(
        x=[
            0.5 * (raising + lowering),
            paddle.diag(x=1.0j * m),
            -0.5j * (raising - lowering),
        ],
        axis=0,
    )


def change_basis_real_to_complex(l: int, dtype=None, device=None) -> paddle.Tensor:
    q = paddle.zeros(shape=(2 * l + 1, 2 * l + 1), dtype="complex128")
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1.0j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1.0j * (-1) ** m / 2**0.5
    q = (-1.0j) ** l * q
    dtype, device = explicit_default_types(dtype, device)
    if isinstance(dtype, str):
        if dtype.lower() == "float32":
            dtype = paddle.float32
        elif dtype.lower() == "float64":
            dtype = paddle.float64
    dtype = {paddle.float32: paddle.complex64, paddle.float64: paddle.complex128}[dtype]
    q = q.astype(dtype)
    if isinstance(device, paddle.CUDAPlace):
        q = q.cuda(device.device_id)
    elif isinstance(device, paddle.CPUPlace):
        q = q.cpu()
    q = q.clone()
    return q


def so3_generators(l) -> paddle.Tensor:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = paddle.conj(x=Q.T) @ X @ Q
    assert paddle.all(x=paddle.abs(x=paddle.imag(x=X)) < 1e-05)
    return paddle.real(x=X)


def wigner_D(l, alpha, beta, gamma):
    """Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\\text{identity rotation}) = \\text{identity matrix}`
    * :math:`D(R_1 \\circ R_2) = D(R_1) \\circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\\alpha, \\beta, \\gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = paddle.broadcast_tensors(input=[alpha, beta, gamma])
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = so3_generators(l)
    return (
        paddle.linalg.matrix_exp(alpha * X[1])
        @ paddle.linalg.matrix_exp(beta * X[0])
        @ paddle.linalg.matrix_exp(gamma * X[1])
    )


def wigner_3j(l1, l2, l3, dtype=None, device=None):
    """Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:

        .. math::

            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \\qquad \\forall g \\in SO(3)

        where :math:`D` are given by `wigner_D`.

        .. math::

            C_{ijk} C_{ijk} = 1

    Parameters
    ----------
    l1 : int
        :math:`l_1`

    l2 : int
        :math:`l_2`

    l3 : int
        :math:`l_3`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensor. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensor. If ``None`` then set to the default device of the current context.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)
    C = _so3_clebsch_gordan(l1, l2, l3)
    dtype, device = explicit_default_types(dtype, device)
    return C.to(dtype=dtype, device=device).clone()


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1, l2, l3):
    Q1 = change_basis_real_to_complex(l1, dtype="float64")
    Q2 = change_basis_real_to_complex(l2, dtype="float64")
    Q3 = change_basis_real_to_complex(l3, dtype="float64")
    C = _su2_clebsch_gordan(l1, l2, l3).to(dtype="complex128")
    C = paddle.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, paddle.conj(x=Q3.T), C)
    assert paddle.all(x=paddle.abs(x=paddle.imag(x=C)) < 1e-05)
    C = paddle.real(x=C)
    C = C / paddle.linalg.norm(x=C)
    return C


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j1, j2, j3):
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    assert isinstance(j1, (int, float))
    assert isinstance(j2, (int, float))
    assert isinstance(j3, (int, float))
    mat = paddle.zeros(
        shape=(int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)), dtype="float64"
    )
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[
                        int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)
                    ] = _su2_clebsch_gordan_coeff((j1, m1), (j2, m2), (j3, m1 + m2))
    return mat


def _su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    from fractions import Fraction
    from math import factorial

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return factorial(round(n))

    C = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    C = C * S
    return C
