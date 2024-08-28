import math

import numpy as np
import paddle
from scipy.special import binom


class GaussianSmearing(paddle.nn.Layer):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return paddle.exp(self.coeff * paddle.pow(dist, 2))


class PolynomialEnvelope(paddle.nn.Layer):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return paddle.where(
            condition=d_scaled < 1, x=env_val, y=paddle.zeros_like(x=d_scaled)
        )


class ExponentialEnvelope(paddle.nn.Layer):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = paddle.exp(x=-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return paddle.where(
            condition=d_scaled < 1, x=env_val, y=paddle.zeros_like(x=d_scaled)
        )


class SphericalBesselBasis(paddle.nn.Layer):
    """
    1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.norm_const = math.sqrt(2 / cutoff**3)
        out_0 = paddle.create_parameter(
            shape=paddle.to_tensor(
                data=np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ).shape,
            dtype=paddle.to_tensor(
                data=np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(
                    data=np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
                )
            ),
        )
        out_0.stop_gradient = not True
        self.frequencies = out_0

    def forward(self, d_scaled):
        return (
            self.norm_const
            / d_scaled[:, None]
            * paddle.sin(x=self.frequencies * d_scaled[:, None])
        )


class BernsteinBasis(paddle.nn.Layer):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            name="prefactor",
            tensor=paddle.to_tensor(data=prefactor, dtype="float32"),
            persistable=False,
        )
        out_1 = paddle.create_parameter(
            shape=paddle.to_tensor(data=pregamma_initial, dtype="float32").shape,
            dtype=paddle.to_tensor(data=pregamma_initial, dtype="float32")
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(data=pregamma_initial, dtype="float32")
            ),
        )
        out_1.stop_gradient = not True
        self.pregamma = out_1
        self.softplus = paddle.nn.Softplus()
        exp1 = paddle.arange(end=num_radial)
        self.register_buffer(name="exp1", tensor=exp1[None, :], persistable=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer(name="exp2", tensor=exp2[None, :], persistable=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)
        exp_d = paddle.exp(x=-gamma * d_scaled)[:, None]
        return self.prefactor * exp_d**self.exp1 * (1 - exp_d) ** self.exp2


class RadialBasis(paddle.nn.Layer):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff
        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]
        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")
        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)
