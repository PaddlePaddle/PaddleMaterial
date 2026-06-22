# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Spherical Fourier-Bessel basis functions for 3D geometric deep learning.

Provides three levels of geometric embeddings:
    - DistEmbedding (RBF): radial basis with smooth envelope
    - AngleEmbedding (SBF): spherical Bessel + Legendre (zero-m) basis
    - TorsionEmbedding (TBF): full 3D spherical Fourier-Bessel (non-zero m) basis

Bessel basis generation is imported from ppmat.models.common.basis_utils.
The real spherical harmonics are implemented locally because the common
variant does not support the non-zero-m indexing convention required here.
"""

import math

import paddle
import sympy as sym

from ppmat.models.common.basis_utils import bessel_basis

# ---------------------------------------------------------------------------
# Real spherical harmonics (SphereNet-compatible implementation)
# Uses the same lexicographic indexing as the DIG/SphereNet reference code.
# ---------------------------------------------------------------------------


def _sph_harm_prefactor(l_degree, m_order):
    return (
        (2 * l_degree + 1)
        * math.factorial(l_degree - abs(m_order))
        / (4 * math.pi * math.factorial(l_degree + abs(m_order)))
    ) ** 0.5


def _associated_legendre_polynomials(k, zero_m_only=True):
    z = sym.symbols("z")
    P_l_m = [[0] * (j + 1) for j in range(k)]
    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z
        for j in range(2, k):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j
            )
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < k:
                    P_l_m[i + 1][i] = sym.simplify((2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    P_l_m[j][i] = sym.simplify(
                        (
                            (2 * j - 1) * z * P_l_m[j - 1][i]
                            - (i + j - 1) * P_l_m[j - 2][i]
                        )
                        / (j - i)
                    )
    return P_l_m


def _real_sph_harm(degree, zero_m_only=False, spherical_coordinates=True):
    """Compute symbolic real spherical harmonics up to degree (excluded).

    This is a copy of the DIG/SphereNet implementation that uses list-of-lists
    indexing (not sympy Matrix indexing), so it works correctly for both
    zero-m-only and full non-zero-m cases.
    """
    if not zero_m_only:
        x = sym.symbols("x")
        y = sym.symbols("y")
        S_m = [x * 0]
        C_m = [1 + 0 * x]
        for i in range(1, degree):
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = _associated_legendre_polynomials(degree, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if not isinstance(P_l_m[i][j], int):
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(degree)]
    for i in range(degree):
        Y_func_l_m[i][0] = sym.simplify(_sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, degree):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * _sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, degree):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * _sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )
    return Y_func_l_m


class Envelope(paddle.nn.Layer):
    """Smooth polynomial envelope function for radial cutoff."""

    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class DistEmbedding(paddle.nn.Layer):
    """Radial basis function (RBF) embedding.

    Uses spherical Bessel functions with a smooth envelope cutoff.
    """

    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = paddle.create_parameter(
            shape=[num_radial],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                paddle.arange(
                    1, num_radial + 1, dtype=paddle.get_default_dtype()
                ).multiply(paddle.to_tensor(3.141592653589793))
            ),
        )

    def reset_parameters(self):
        with paddle.no_grad():
            pi_t = paddle.to_tensor(3.141592653589793)
            self.freq.set_value(
                paddle.arange(
                    1, self.freq.shape[0] + 1, dtype=paddle.get_default_dtype()
                ).multiply(pi_t)
            )

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * paddle.sin(self.freq * dist)


class AngleEmbedding(paddle.nn.Layer):
    """Spherical Bessel + Legendre (zero-m) embedding for bond angles.

    Combines radial Bessel functions with m=0 real spherical harmonics
    (Legendre polynomials) to encode pairwise distances and angles.
    """

    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = _real_sph_harm(num_spherical, zero_m_only=True)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": paddle.sin, "cos": paddle.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(
                    lambda x_val: paddle.zeros_like(x_val) + float(sph1)
                )
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = paddle.stack([f(dist) for f in self.bessel_funcs], axis=1)
        cbf = paddle.stack([f(angle) for f in self.sph_funcs], axis=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].reshape([-1, n, k]) * cbf.reshape([-1, n, 1])).reshape(
            [-1, n * k]
        )
        return out


class TorsionEmbedding(paddle.nn.Layer):
    """Full 3D spherical Fourier-Bessel embedding for torsion angles.

    Uses non-zero m spherical harmonics to encode the full 3D geometric
    configuration (radial distance + polar angle + azimuthal angle).
    """

    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = _real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": paddle.sin, "cos": paddle.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(
                    lambda theta_val, phi_val: (
                        paddle.zeros_like(theta_val)
                        + paddle.zeros_like(phi_val)
                        + float(sph1(0, 0))
                    )
                )
            else:
                for k_order in range(-i, i + 1):
                    sph = sym.lambdify(
                        [theta, phi], sph_harm_forms[i][k_order + i], modules
                    )
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = paddle.stack([f(dist) for f in self.bessel_funcs], axis=1)
        cbf = paddle.stack([f(angle, phi) for f in self.sph_funcs], axis=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].reshape([-1, 1, n, k]) * cbf.reshape([-1, n, n, 1])).reshape(
            [-1, n * n * k]
        )
        return out
