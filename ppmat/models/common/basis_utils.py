# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/basis_utils.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
from typing import Any
from typing import List

import numpy as np
import sympy as sym
from scipy import special as sp
from scipy.optimize import brentq


def Jn(r: np.array, n: int) -> np.array:
    """
    numerical spherical bessel functions of order n
    """
    return sp.spherical_jn(n, r)


def Jn_zeros(n: int, k: int) -> np.array:
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n: int) -> List[Any]:
    """
    Computes the sympy formulas for the spherical bessel functions up to order n
    (excluded)
    """
    x = sym.symbols("x")
    j = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        j += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return j


def bessel_basis(n: int, k: int) -> List[Any]:
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel
    functions up to order n (excluded) and maximum frequency k (excluded).

    Returns:
        bess_basis: list
            Bessel basis formulas taking in a single argument x.
            Has length n where each element has length k. -> In total n*k many.
    """
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i] * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l_degree: int, m_order: int) -> float:
    """Computes the constant pre-factor for the spherical harmonic of degree l and
    order m.

    Parameters
    ----------
        l_degree: int
            Degree of the spherical harmonic. l >= 0
        m_order: int
            Order of the spherical harmonic. -l <= m <= l

    Returns
    -------
        factor: float

    """
    return (
        (2 * l_degree + 1)
        / (4 * np.pi)
        * np.math.factorial(l_degree - abs(m_order))
        / np.math.factorial(l_degree + abs(m_order))
    ) ** 0.5


def associated_legendre_polynomials(
    L_maxdegree: int, zero_m_only: bool = True, pos_m_only: bool = True
) -> List[List[Any]]:
    """Computes string formulas of the associated legendre polynomials up to degree L
    (excluded).

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the associated legendre polynomials (degree
            L is excluded).
        zero_m_only: bool
            If True only calculate the polynomials for the polynomials where m=0.
        pos_m_only: bool
            If True only calculate the polynomials for the polynomials where m>=0.
            Overwritten by zero_m_only.

    Returns
    -------
        polynomials: list
            Contains the sympy functions of the polynomials (in total L many if
            zero_m_only is True else L^2 many).
    """
    z = sym.symbols("z")
    P_l_m = [
        [sym.Integer(0) for _ in range(2 * l_degree + 1)]
        for l_degree in range(L_maxdegree)
    ]
    P_l_m[0][0] = sym.Integer(1)
    if L_maxdegree > 0:
        if zero_m_only:
            P_l_m[1][0] = z
            for l_degree in range(2, L_maxdegree):
                P_l_m[l_degree][0] = sym.simplify(
                    (
                        (2 * l_degree - 1) * z * P_l_m[l_degree - 1][0]
                        - (l_degree - 1) * P_l_m[l_degree - 2][0]
                    )
                    / l_degree
                )
        else:
            for l_degree in range(1, L_maxdegree):
                P_l_m[l_degree][l_degree] = sym.simplify(
                    (1 - 2 * l_degree)
                    * (1 - z**2) ** 0.5
                    * P_l_m[l_degree - 1][l_degree - 1]
                )
            for m_order in range(0, L_maxdegree - 1):
                P_l_m[m_order + 1][m_order] = sym.simplify(
                    (2 * m_order + 1) * z * P_l_m[m_order][m_order]
                )
            for l_degree in range(2, L_maxdegree):
                for m_order in range(l_degree - 1):
                    P_l_m[l_degree][m_order] = sym.simplify(
                        (
                            (2 * l_degree - 1) * z * P_l_m[l_degree - 1][m_order]
                            - (l_degree + m_order - 1) * P_l_m[l_degree - 2][m_order]
                        )
                        / (l_degree - m_order)
                    )
            if not pos_m_only:
                for l_degree in range(1, L_maxdegree):
                    for m_order in range(1, l_degree + 1):
                        P_l_m[l_degree][-m_order] = sym.simplify(
                            (-1) ** m_order
                            * np.math.factorial(l_degree - m_order)
                            / np.math.factorial(l_degree + m_order)
                            * P_l_m[l_degree][m_order]
                        )
    return P_l_m


def real_sph_harm(
    L_maxdegree: int, use_theta: bool, use_phi: bool = True, zero_m_only: bool = True
) -> List[List[Any]]:
    """
    Computes formula strings of the the real part of the spherical harmonics up to
    degree L (excluded).Variables are either spherical coordinates phi and theta (or
    cartesian coordinates x,y,z) on the UNIT SPHERE.

    Parameters
    ----------
        L_maxdegree: int
            Degree up to which to calculate the spherical harmonics (degree L is
            excluded).
        use_theta: bool
            - True: Expects the input of the formula strings to contain theta.
            - False: Expects the input of the formula strings to contain z.
        use_phi: bool
            - True: Expects the input of the formula strings to contain phi.
            - False: Expects the input of the formula strings to contain x and y.
            Does nothing if zero_m_only is True
        zero_m_only: bool
            If True only calculate the harmonics where m=0.

    Returns
    -------
        Y_lm_real: list
            Computes formula strings of the the real part of the spherical harmonics up
            to degree L (where degree L is not excluded).
            In total L^2 many sph harm exist up to degree L (excluded). However, if
            zero_m_only only is True then the total count is reduced to be only L many.
    """
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L_maxdegree, zero_m_only)
    if zero_m_only:
        Y_l_m = [sym.zeros(1) for l_degree in range(L_maxdegree)]
    else:
        Y_l_m = [(sym.zeros(1) * (2 * l_degree + 1)) for l_degree in range(L_maxdegree)]
    if use_theta:
        theta = sym.symbols("theta")
        for l_degree in range(L_maxdegree):
            for m_order in range(len(P_l_m[l_degree])):
                P_l_m[l_degree][m_order] = P_l_m[l_degree][m_order].subs(
                    z, sym.cos(theta)
                )
    for l_degree in range(L_maxdegree):
        Y_l_m[l_degree][0] = sym.simplify(
            sph_harm_prefactor(l_degree, 0) * P_l_m[l_degree][0]
        )
    if not zero_m_only:
        phi = sym.symbols("phi")
        for l_degree in range(1, L_maxdegree):
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.cos(m_order * phi)
                )
            for m_order in range(1, l_degree + 1):
                Y_l_m[l_degree][-m_order] = sym.simplify(
                    2**0.5
                    * (-1) ** m_order
                    * sph_harm_prefactor(l_degree, -m_order)
                    * P_l_m[l_degree][m_order]
                    * sym.sin(m_order * phi)
                )
        if not use_phi:
            x = sym.symbols("x")
            y = sym.symbols("y")
            for l_degree in range(L_maxdegree):
                for m_order in range(len(Y_l_m[l_degree])):
                    assert isinstance(Y_l_m[l_degree][m_order], int)
                    Y_l_m[l_degree][m_order] = sym.simplify(
                        Y_l_m[l_degree][m_order].subs(phi, sym.atan2(y, x))
                    )
    return Y_l_m
