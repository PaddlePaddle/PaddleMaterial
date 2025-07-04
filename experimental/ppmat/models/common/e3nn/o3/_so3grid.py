import paddle

from ._s2grid import _quadrature_weights
from ._s2grid import s2_grid
from ._wigner import wigner_D


def flat_wigner(lmax, alpha, beta, gamma):
    return paddle.concat(
        x=[
            ((2 * l + 1) ** 0.5 * wigner_D(l, alpha, beta, gamma).flatten(-2))
            for l in range(lmax + 1)
        ],
        axis=-1,
    )


class SO3Grid(paddle.nn.Layer):
    """Apply non linearity on the signal on SO(3)

    Parameters
    ----------
    lmax : int
        irreps representation ``[(2 * l + 1, (l, p_val)) for l in [0, ..., lmax]]``

    resolution : int
        SO(3) grid resolution

    normalization : {'norm', 'component'}

    aspect_ratio : float
        default value (2) should be optimal
    """

    def __init__(self, lmax, resolution, *, normalization="component", aspect_ratio=2):
        super().__init__()
        assert normalization == "component"
        nb = 2 * resolution
        na = round(2 * aspect_ratio * resolution)
        b, a = s2_grid(nb, na)
        self.register_buffer(
            name="D",
            tensor=flat_wigner(
                lmax, a[:, None, None], b[None, :, None], a[None, None, :]
            ),
        )
        qw = _quadrature_weights(nb // 2) * nb**2 / na**2
        self.register_buffer(name="qw", tensor=qw)
        self.register_buffer(name="alpha", tensor=a)
        self.register_buffer(name="beta", tensor=b)
        self.register_buffer(name="gamma", tensor=a)
        self.res_alpha = na
        self.res_beta = nb
        self.res_gamma = na

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.lmax})"

    def to_grid(self, features):
        """evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.irreps.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``
        """
        return (
            paddle.einsum("...i,abci->...abc", features, self.D)
            / tuple(self.D.shape)[-1] ** 0.5
        )

    def from_grid(self, features):
        """evaluate

        Parameters
        ----------

        features : `torch.Tensor`
            tensor of shape ``(..., self.res_alpha, self.res_beta, self.res_gamma)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.irreps.dim)``
        """
        return (
            paddle.einsum("...abc,abci,b->...i", features, self.D, self.qw)
            * tuple(self.D.shape)[-1] ** 0.5
        )
