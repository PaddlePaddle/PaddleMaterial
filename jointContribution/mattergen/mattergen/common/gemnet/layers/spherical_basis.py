import sys

import paddle

from paddle_utils import *

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/spherical_basis.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
import sympy as sym

from mattergen.common.gemnet.layers.basis_utils import real_sph_harm
from mattergen.common.gemnet.layers.radial_basis import RadialBasis
from paddle_geometric.nn.models.schnet import GaussianSmearing


class CircularBasisLayer(paddle.nn.Layer):
    """
    2D Fourier Bessel Basis

    Parameters
    ----------
    num_spherical: int
        Controls maximum frequency.
    radial_basis: RadialBasis
        Radial basis functions
    cbf: dict
        Name and hyperparameters of the cosine basis function
    efficient: bool
        Whether to use the "efficient" summation order
    """

    def __init__(
        self,
        num_spherical: int,
        radial_basis: RadialBasis,
        cbf: dict,
        efficient: bool = False,
    ):
        super().__init__()
        self.radial_basis = radial_basis
        self.efficient = efficient
        cbf_name = cbf["name"].lower()
        cbf_hparams = cbf.copy()
        del cbf_hparams["name"]
        if cbf_name == "gaussian":
            self.cosφ_basis = GaussianSmearing(
                start=-1, stop=1, num_gaussians=num_spherical, **cbf_hparams
            )
        elif cbf_name == "spherical_harmonics":
            Y_lm = real_sph_harm(num_spherical, use_theta=False, zero_m_only=True)
            sph_funcs = []
            z = sym.symbols("z")
            modules = {"sin": paddle.sin, "cos": paddle.cos, "sqrt": paddle.sqrt}
            m_order = 0
            for l_degree in range(len(Y_lm)):
                if l_degree == 0:
                    first_sph = sym.lambdify([z], Y_lm[l_degree][m_order], modules)
                    sph_funcs.append(lambda z: paddle.zeros_like(x=z) + first_sph(z))
                else:
                    sph_funcs.append(sym.lambdify([z], Y_lm[l_degree][m_order], modules))
            self.cosφ_basis = lambda cosφ: paddle.stack(x=[f(cosφ) for f in sph_funcs], axis=1)
        else:
            raise ValueError(f"Unknown cosine basis function '{cbf_name}'.")

    def forward(self, D_ca, cosφ_cab, id3_ca):
        rbf = self.radial_basis(D_ca)
        cbf = self.cosφ_basis(cosφ_cab)
        if not self.efficient:
            rbf = rbf[id3_ca]
            out = (rbf[:, None, :] * cbf[:, :, None]).view(
                -1, tuple(rbf.shape)[-1] * tuple(cbf.shape)[-1]
            )
            return (out,)
        else:
            return rbf[None, :, :], cbf
