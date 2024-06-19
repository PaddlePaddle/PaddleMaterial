from __future__ import annotations
import paddle
from typing import Literal
from functools import lru_cache
from math import pi, sqrt
import sympy


class GaussianExpansion(paddle.nn.Layer):
    """Gaussian Radial Expansion.

    The bond distance is expanded to a vector of shape [m], where m is the number of Gaussian basis centers.
    """

    def __init__(self, initial: float=0.0, final: float=4.0, num_centers:
        int=20, width: (None | float)=0.5):
        """
        Args:
            initial: Location of initial Gaussian basis center.
            final: Location of final Gaussian basis center
            num_centers: Number of Gaussian Basis functions
            width: Width of Gaussian Basis functions.
        """
        super().__init__()
        out_0 = paddle.create_parameter(shape=paddle.linspace(start=initial,
            stop=final, num=num_centers).shape, dtype=paddle.linspace(start
            =initial, stop=final, num=num_centers).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.
            linspace(start=initial, stop=final, num=num_centers)))
        out_0.stop_gradient = not False
        self.centers = out_0
        if width is None:
            self.width = 1.0 / paddle.diff(x=self.centers).mean()
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        out_1 = paddle.create_parameter(shape=self.centers.shape, dtype=
            self.centers.numpy().dtype, default_initializer=paddle.nn.
            initializer.Assign(self.centers))
        out_1.stop_gradient = not False
        self.centers = out_1

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return paddle.exp(x=-self.width * diff ** 2)


class BondExpansion(paddle.nn.Layer):
    """Expand pair distances into a set of spherical bessel or gaussian functions."""

    def __init__(self, max_l: int=3, max_n: int=3, cutoff: float=5.0,
        rbf_type: Literal['SphericalBessel', 'Gaussian']='SphericalBessel',
        smooth: bool=False, initial: float=0.0, final: float=5.0,
        num_centers: int=100, width: float=0.5) ->None:
        """
        Args:
            max_l (int): order of angular part
            max_n (int): order of radial part
            cutoff (float): cutoff radius
            rbf_type (str): type of radial basis function .i.e. either "SphericalBessel" or 'Gaussian'
            smooth (bool): whether apply the smooth version of spherical bessel functions or not
            initial (float): initial point for gaussian expansion
            final (float): final point for gaussian expansion
            num_centers (int): Number of centers for gaussian expansion.
            width (float): width of gaussian function.
        """
        super().__init__()
        self.max_n = max_n
        self.cutoff = cutoff
        self.max_l = max_l
        self.smooth = smooth
        self.num_centers = num_centers
        self.width = width
        self.initial = initial
        self.final = final
        self.rbf_type = rbf_type
        if rbf_type.lower() == 'sphericalbessel':
            self.rbf = SphericalBesselFunction(max_l, max_n, cutoff, smooth)
        elif rbf_type.lower() == 'gaussian':
            self.rbf = GaussianExpansion(initial, final, num_centers, width)
        else:
            raise ValueError(
                'Undefined rbf_type, please use SphericalBessel or Gaussian instead.'
                )

    def forward(self, bond_dist: paddle.Tensor):
        """Forward.

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis
