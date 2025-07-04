import paddle
import paddle.nn as nn

from ppmat.models.common.e3nn.paddle_utils import *

"""Spherical Harmonics as functions of Euler angles
"""
import math
from typing import List
from typing import Tuple

from sympy import Integer
from sympy import Poly
from sympy import diff
from sympy import factorial
from sympy import pi
from sympy import sqrt
from sympy import symbols

from ppmat.models.common.e3nn import o3


class SphericalHarmonicsAlphaBeta(paddle.nn.Layer):
    """JITable module version of :meth:`e3nn.o3.spherical_harmonics_alpha_beta`.

    Parameters are identical to :meth:`e3nn.o3.spherical_harmonics_alpha_beta`.
    """

    normalization: str
    _ls_list: List[int]
    _lmax: int

    def __init__(self, l, normalization="integral"):
        super().__init__()
        if isinstance(l, o3.Irreps):
            ls = [l for mul, (l, p) in l for _ in range(mul)]
        elif isinstance(l, int):
            ls = [l]
        else:
            ls = list(l)
        self._ls_list = ls
        self._lmax = max(ls)
        self.legendre = Legendre(ls)
        self.normalization = normalization

    def forward(self, alpha: paddle.Tensor, beta: paddle.Tensor) -> paddle.Tensor:
        y, z = beta.cos(), beta.sin()
        sha = spherical_harmonics_alpha(self._lmax, alpha.flatten())
        shy = self.legendre(y.flatten(), z.flatten())
        out = _mul_m_lm([(1, l) for l in self._ls_list], sha, shy)
        if self.normalization == "norm":
            out.divide_(
                y=paddle.to_tensor(
                    paddle.concat(
                        x=[
                            (
                                math.sqrt(2 * l + 1)
                                / math.sqrt(4 * math.pi)
                                * paddle.ones(shape=2 * l + 1, dtype=out.dtype)
                            )
                            for l in self._ls_list
                        ]
                    )
                )
            )
        elif self.normalization == "component":
            out.multiply_(y=paddle.to_tensor(math.sqrt(4 * math.pi)))
        return out.reshape(tuple(alpha.shape) + (tuple(shy.shape)[1],))


def spherical_harmonics_alpha_beta(l, alpha, beta, *, normalization="integral"):
    """Spherical harmonics of :math:`\\vec r = R_y(\\alpha) R_x(\\beta) e_y`

    .. math:: Y^l(\\alpha, \\beta) = S^l(\\alpha) P^l(\\cos(\\beta))

    where :math:`P^l` are the `Legendre` polynomials


    Parameters
    ----------
    l : int or list of int
        degree of the spherical harmonics.

    alpha : `torch.Tensor`
        tensor of shape ``(...)``.

    beta : `torch.Tensor`
        tensor of shape ``(...)``.

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``
    """
    sh = SphericalHarmonicsAlphaBeta(l, normalization=normalization)
    return sh(alpha, beta)


def spherical_harmonics_alpha(l: int, alpha: paddle.Tensor) -> paddle.Tensor:
    """:math:`S^l(\\alpha)` of `spherical_harmonics_alpha_beta`

    Parameters
    ----------
    l : int
        degree of the spherical harmonics.

    alpha : `torch.Tensor`
        tensor of shape ``(...)``.

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``
    """
    alpha = alpha.unsqueeze(axis=-1)
    m = paddle.arange(start=1, end=l + 1, dtype=alpha.dtype)
    cos = paddle.cos(x=m * alpha)
    m = paddle.arange(start=l, end=0, step=-1, dtype=alpha.dtype)
    sin = paddle.sin(x=m * alpha)
    out = paddle.concat(
        x=[math.sqrt(2) * sin, paddle.ones_like(x=alpha), math.sqrt(2) * cos],
        axis=alpha.ndim - 1,
    )
    return out


class Legendre(nn.Layer):
    def __init__(self, ls):
        super().__init__()
        self.ls = ls

    def forward(self, z: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
        out_shape = list(z.shape) + [sum(2 * l + 1 for l in self.ls)]
        out = paddle.zeros(out_shape, dtype=z.dtype)

        i = 0
        for l in self.ls:
            leg = []
            for m in range(l + 1):
                p = _poly_legendre(l, m)
                x = paddle.zeros_like(z)

                for (zn, yn), c in p.items():
                    x += float(c) * paddle.pow(z, zn) * paddle.pow(y, yn)

                leg.append(paddle.unsqueeze(x, axis=-1))

            for m in range(-l, l + 1):
                out_slice = paddle.slice(out, axes=[-1], starts=[i], ends=[i + 1])
                paddle.assign(leg[abs(m)], out_slice)
                i += 1

        return out


def _poly_legendre(l, m):
    """
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols("z y", real=True)
    return Poly(_sympy_legendre(l, m), domain="R", gens=(z, y)).as_dict()


def _sympy_legendre(l, m):
    """
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)
    - remove (-1)^l

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols("z y", real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1) ** l, z, l + m)
    ex *= sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


def _mul_m_lm(
    mul_l: List[Tuple[int, int]], x_m: paddle.Tensor, x_lm: paddle.Tensor
) -> paddle.Tensor:
    """
    multiply tensor [..., l * m] by [..., m]
    """
    l_max = tuple(x_m.shape)[-1] // 2
    out = []
    i = 0
    for mul, l in mul_l:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i : i + d]
        x1 = x1.reshape(tuple(x1.shape)[:-1] + (mul, 2 * l + 1))
        x2 = x_m[..., l_max - l : l_max + l + 1]
        x2 = x2.reshape(tuple(x2.shape)[:-1] + (1, 2 * l + 1))
        x = x1 * x2
        x = x.reshape(tuple(x.shape)[:-2] + (d,))
        out.append(x)
        i += d
    return paddle.concat(x=out, axis=-1)
