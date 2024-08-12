import sys

import numpy as np
import paddle
import sympy as sym

from .basis_utils import bessel_basis
from .basis_utils import real_sph_harm
from .envelope import Envelope

sys.path.append("/home/chenxiaoxu02/workspaces/gemnet_paddle/utils")


class BesselBasisLayer(paddle.nn.Layer):
    """
    1D Bessel Basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        name="bessel_basis",
    ):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        self.envelope = Envelope(envelope_exponent)
        out_0 = paddle.create_parameter(
            shape=paddle.to_tensor(
                data=np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32),
                dtype="float32",
            ).shape,
            dtype=paddle.to_tensor(
                data=np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32),
                dtype="float32",
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    data=np.pi * np.arange(1, self.num_radial + 1, dtype=np.float32),
                    dtype="float32",
                )
            ),
        )
        out_0.stop_gradient = not True
        self.frequencies = out_0

    def forward(self, d):
        d = d[:, None]
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env * self.norm_const * paddle.sin(x=self.frequencies * d_scaled) / d


class SphericalBasisLayer(paddle.nn.Layer):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient: bool = False,
        name: str = "spherical_basis",
    ):
        super().__init__()
        assert num_radial <= 64
        self.efficient = efficient
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.envelope = Envelope(envelope_exponent)
        self.inv_cutoff = 1 / cutoff
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=True
        )
        self.sph_funcs = []
        self.bessel_funcs = []
        self.norm_const = self.inv_cutoff**1.5
        self.register_buffer(
            name="device_buffer", tensor=paddle.zeros(shape=[0]), persistable=False
        )
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": paddle.sin, "cos": paddle.cos, "sqrt": paddle.sqrt}
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: paddle.zeros_like(x=theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, D_ca, Angle_cab, id3_reduce_ca, Kidx):
        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = paddle.stack(x=rbf, axis=1)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf
        sph = [f(Angle_cab) for f in self.sph_funcs]
        sph = paddle.stack(x=sph, axis=1)
        if not self.efficient:
            rbf_env = rbf_env[id3_reduce_ca]
            # rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = rbf_env.reshape([-1, self.num_spherical, self.num_radial])
            # sph = sph.view(-1, self.num_spherical, 1)
            sph = sph.reshape([-1, self.num_spherical, 1])
            # out = (rbf_env * sph).view(-1, self.num_spherical * self.num_radial)
            out = (rbf_env * sph).reshape([-1, self.num_spherical * self.num_radial])
            return out
        else:
            # rbf_env = rbf_env.view(-1, self.num_spherical, self.num_radial)
            rbf_env = rbf_env.reshape((-1, self.num_spherical, self.num_radial))
            x = rbf_env
            perm_0 = list(range(x.ndim))
            perm_0[0] = 1
            perm_0[1] = 0
            rbf_env = paddle.transpose(x=x, perm=perm_0)
            Kmax = (
                0
                if tuple(sph.shape)[0] == 0
                else paddle.maximum(paddle.max(x=Kidx + 1), paddle.to_tensor(data=0))
            )
            nEdges = tuple(d_scaled.shape)[0]
            sph2 = paddle.zeros(
                shape=[nEdges, Kmax, self.num_spherical], dtype=sph.dtype
            )
            sph2[id3_reduce_ca, Kidx] = sph
            return rbf_env, sph2


class TensorBasisLayer(paddle.nn.Layer):
    """
    3D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    envelope_exponent: int = 5
        Exponent of the envelope function.
    efficient: bool
        Whether to use the (memory) efficient implementation or not.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        envelope_exponent: int = 5,
        efficient=False,
        name: str = "tensor_basis",
    ):
        super().__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.efficient = efficient
        self.inv_cutoff = 1 / cutoff
        self.envelope = Envelope(envelope_exponent)
        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, spherical_coordinates=True, zero_m_only=False
        )
        self.sph_funcs = []
        self.bessel_funcs = []
        self.norm_const = self.inv_cutoff**1.5
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": paddle.sin, "cos": paddle.cos, "sqrt": paddle.sqrt}
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if l == 0:
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: paddle.zeros_like(x=theta)
                        + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )
        self.register_buffer(
            name="degreeInOrder",
            tensor=paddle.arange(end=num_spherical) * 2 + 1,
            persistable=False,
        )

    def forward(self, D_ca, Alpha_cab, Theta_cabd, id4_reduce_ca, Kidx):
        d_scaled = D_ca * self.inv_cutoff
        u_d = self.envelope(d_scaled)
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = paddle.stack(x=rbf, axis=1)
        rbf = rbf * self.norm_const
        rbf_env = u_d[:, None] * rbf
        # rbf_env = rbf_env.view((-1, self.num_spherical, self.num_radial))
        rbf_env = rbf_env.reshape((-1, self.num_spherical, self.num_radial))
        rbf_env = paddle.repeat_interleave(
            x=rbf_env, repeats=self.degreeInOrder, axis=1
        )
        if not self.efficient:
            # rbf_env = rbf_env.view((-1, self.num_spherical**2 * self.num_radial))
            rbf_env = rbf_env.reshape((-1, self.num_spherical**2 * self.num_radial))
            rbf_env = rbf_env[id4_reduce_ca]
        sph = [f(Alpha_cab, Theta_cabd) for f in self.sph_funcs]
        sph = paddle.stack(x=sph, axis=1)
        if not self.efficient:
            # >>>>>>            sph = torch.repeat_interleave(sph, self.num_radial, axis=1)
            sph = paddle.repeat_interleave(sph, self.num_radial, axis=1)
            return rbf_env * sph
        else:
            x = rbf_env
            perm_1 = list(range(x.ndim))
            perm_1[0] = 1
            perm_1[1] = 0
            rbf_env = paddle.transpose(x=x, perm=perm_1)
            Kmax = (
                0
                if tuple(sph.shape)[0] == 0
                else paddle.maximum(paddle.max(x=Kidx + 1), paddle.to_tensor(data=0))
            )
            nEdges = tuple(d_scaled.shape)[0]
            sph2 = paddle.zeros(
                shape=[nEdges, Kmax, self.num_spherical**2], dtype=sph.dtype
            )
            sph2[id4_reduce_ca, Kidx] = sph
            return rbf_env, sph2
