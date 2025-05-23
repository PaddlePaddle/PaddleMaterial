import math

import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.paddle_utils import *
from ppmat.models.common.e3nn.util import explicit_default_types

"""Transformation between two representations of a signal on the sphere.

.. math:: f: S^2 \\longrightarrow \\mathbb{R}

is a signal on the sphere.

One representation that we like to call "spherical tensor" is

.. math:: f(x) = \\sum_{l=0}^{l_{\\mathit{max}}} F^l \\cdot Y^l(x)

it is made of :math:`(l_{\\mathit{max}} + 1)^2` real numbers represented in the above formula by the familly of vectors
:math:`F^l \\in \\mathbb{R}^{2l+1}`.

Another representation is the discretization around the sphere. For this representation we chose a particular grid of size
:math:`(N, M)`

.. math::

    x_{ij} &= (\\sin(\\beta_i) \\sin(\\alpha_j), \\cos(\\beta_i), \\sin(\\beta_i) \\cos(\\alpha_j))

    \\beta_i &= \\pi (i + 0.5) / N

    \\alpha_j &= 2 \\pi j / M

In the code, :math:`N` is called ``res_beta`` and :math:`M` is ``res_alpha``.

The discrete representation is therefore

.. math:: \\{ h_{ij} = f(x_{ij}) \\}_{ij}
"""


def _quadrature_weights(b, dtype=None, device=None):
    """
    function copied from ``lie_learn.spaces.S3``

    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].
    """
    k = paddle.arange(end=b)
    w = paddle.to_tensor(
        data=[
            (
                2.0
                / b
                * paddle.sin(x=math.pi * (2.0 * j + 1.0) / (4.0 * b))
                * (
                    1.0
                    / (2 * k + 1)
                    * paddle.sin(x=(2 * j + 1) * (2 * k + 1) * math.pi / (4.0 * b))
                ).sum()
            )
            for j in paddle.arange(end=2 * b)
        ],
        dtype=dtype,
        place=device,
    )
    w /= 2.0 * (2 * b) ** 2
    return w


def s2_grid(res_beta, res_alpha, dtype=None, device=None):
    """grid on the sphere

    Parameters
    ----------
    res_beta : int
        :math:`N`

    res_alpha : int
        :math:`M`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensors. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensors. If ``None`` then set to the default device of the current context.

    Returns
    -------
    betas : `torch.Tensor`
        tensor of shape ``(res_beta)``

    alphas : `torch.Tensor`
        tensor of shape ``(res_alpha)``
    """
    dtype, device = explicit_default_types(dtype, device)
    i = paddle.arange(dtype=dtype, end=res_beta)
    betas = (i + 0.5) / res_beta * math.pi
    i = paddle.arange(dtype=dtype, end=res_alpha)
    alphas = i / res_alpha * 2 * math.pi
    return betas, alphas


def spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=None, device=None):
    """spherical harmonics evaluated on the grid on the sphere

    .. math::

        f(x) = \\sum_{l=0}^{l_{\\mathit{max}}} F^l \\cdot Y^l(x)

        f(\\beta, \\alpha) = \\sum_{l=0}^{l_{\\mathit{max}}} F^l \\cdot S^l(\\alpha) P^l(\\cos(\\beta))

    Parameters
    ----------
    lmax : int
        :math:`l_{\\mathit{max}}`

    res_beta : int
        :math:`N`

    res_alpha : int
        :math:`M`

    Returns
    -------
    betas : `torch.Tensor`
        tensor of shape ``(res_beta)``

    alphas : `torch.Tensor`
        tensor of shape ``(res_alpha)``

    shb : `torch.Tensor`
        tensor of shape ``(res_beta, (lmax + 1)**2)``

    sha : `torch.Tensor`
        tensor of shape ``(res_alpha, 2 lmax + 1)``
    """
    betas, alphas = s2_grid(res_beta, res_alpha, dtype=dtype, device=device)
    shb = o3.Legendre(list(range(lmax + 1)))(betas.cos(), betas.sin().abs())
    sha = o3.spherical_harmonics_alpha(lmax, alphas)
    return betas, alphas, shb, sha


def _complete_lmax_res(lmax, res_beta, res_alpha):
    """
    try to use FFT
    i.e. 2 * lmax + 1 == res_alpha
    """
    if res_beta is None:
        if lmax is not None:
            res_beta = 2 * (lmax + 1)
        elif res_alpha is not None:
            res_beta = 2 * ((res_alpha + 1) // 2)
        else:
            raise ValueError("All the entries are None")
    if res_alpha is None:
        if lmax is not None:
            if res_beta is not None:
                res_alpha = max(2 * lmax + 1, res_beta - 1)
            else:
                res_alpha = 2 * lmax + 1
        elif res_beta is not None:
            res_alpha = res_beta - 1
    if lmax is None:
        lmax = min(res_beta // 2 - 1, (res_alpha - 1) // 2)
    assert res_beta % 2 == 0
    assert lmax + 1 <= res_beta // 2
    return lmax, res_beta, res_alpha


def _expand_matrix(ls, like=None, dtype=None, device=None):
    """
    convertion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    :return: tensor [l, m, l * m]
    """
    lmax = max(ls)
    if like is None:
        m = paddle.zeros(
            shape=[len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls)], dtype=dtype
        )
    else:
        m = paddle.zeros(
            shape=(len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls)), dtype=dtype
        )
    i = 0
    for j, l in enumerate(ls):
        m[j, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = paddle.eye(
            num_rows=2 * l + 1, dtype=dtype
        )
        i += 2 * l + 1
    return m


def rfft(x, l):
    """Real fourier transform

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape ``(..., 2 l + 1)``

    res : int
        output resolution, has to be an odd number

    Returns
    -------
    `torch.Tensor`
        tensor of shape ``(..., res)``

    Examples
    --------

    >>> lmax = 8
    >>> res = 101
    >>> _betas, _alphas, _shb, sha = spherical_harmonics_s2_grid(lmax, res, res)
    >>> x = torch.randn(res)
    >>> (rfft(x, lmax) - x @ sha).abs().max().item() < 1e-4
    True
    """
    *size, res = tuple(x.shape)
    x = x.reshape(-1, res)
    x = paddle.fft.rfft(x=x, axis=1)
    x = paddle.concat(
        x=[
            x[:, 1 : l + 1].imag().flip(axis=1).mul(-math.sqrt(2)),
            x[:, :1].real(),
            x[:, 1 : l + 1].real().mul(math.sqrt(2)),
        ],
        axis=1,
    )
    return x.reshape(*size, 2 * l + 1)


def irfft(x, res):
    """Inverse of the real fourier transform

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape ``(..., 2 l + 1)``

    res : int
        output resolution, has to be an odd number

    Returns
    -------
    `torch.Tensor`
        positions on the sphere, tensor of shape ``(..., res, 3)``

    Examples
    --------

    >>> lmax = 8
    >>> res = 101
    >>> _betas, _alphas, _shb, sha = spherical_harmonics_s2_grid(lmax, res, res)
    >>> x = torch.randn(2 * lmax + 1)
    >>> (irfft(x, res) - sha @ x).abs().max().item() < 1e-4
    True
    """
    assert res % 2 == 1
    *size, sm = tuple(x.shape)
    x = x.reshape(-1, sm)
    x = paddle.concat(
        x=[
            paddle.zeros(shape=(tuple(x.shape)[0], (res - sm) // 2), dtype=x.dtype),
            x,
            paddle.zeros(shape=(tuple(x.shape)[0], (res - sm) // 2), dtype=x.dtype),
        ],
        axis=-1,
    )
    assert tuple(x.shape)[1] == res
    l = res // 2
    x = paddle.complex(
        real=paddle.concat(
            x=[x[:, l : l + 1], x[:, l + 1 :].div(math.sqrt(2))], axis=1
        ),
        imag=paddle.concat(
            x=[
                paddle.zeros_like(x=x[:, :1]),
                x[:, :l].flip(axis=-1).div(-math.sqrt(2)),
            ],
            axis=1,
        ),
    )
    x = paddle.fft.irfft(x=x, n=res, axis=1) * res
    return x.reshape(*size, res)


class ToS2Grid(paddle.nn.Layer):
    """Transform spherical tensor into signal on the sphere

    The inverse transformation of `FromS2Grid`

    Parameters
    ----------
    lmax : int
    res : int, tuple of int
        resolution in ``beta`` and in ``alpha``

    normalization : {'norm', 'component', 'integral'}
    dtype : torch.dtype or None, optional
    device : torch.device or None, optional

    Examples
    --------

    >>> m = ToS2Grid(6, (100, 101))
    >>> x = torch.randn(3, 49)
    >>> m(x).shape
    torch.Size([3, 100, 101])


    `ToS2Grid` and `FromS2Grid` are inverse of each other

    >>> m = ToS2Grid(6, (100, 101))
    >>> k = FromS2Grid((100, 101), 6)
    >>> x = torch.randn(3, 49)
    >>> y = k(m(x))
    >>> (x - y).abs().max().item() < 1e-4
    True

    Attributes
    ----------
    grid : `torch.Tensor`
        positions on the sphere, tensor of shape ``(res_beta, res_alpha, 3)``
    """

    def __init__(
        self, lmax=None, res=None, normalization="component", dtype=None, device=None
    ):
        super().__init__()
        assert normalization in ["norm", "component", "integral"] or paddle.is_tensor(
            x=normalization
        ), "normalization needs to be 'norm', 'component' or 'integral'"
        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)
        betas, alphas, shb, sha = spherical_harmonics_s2_grid(
            lmax, res_beta, res_alpha, dtype=dtype, device=device
        )
        n = None
        if normalization == "component":
            n = (
                math.sqrt(4 * math.pi)
                * paddle.to_tensor(
                    data=[(1 / math.sqrt(2 * l + 1)) for l in range(lmax + 1)],
                    dtype=betas.dtype,
                )
                / math.sqrt(lmax + 1)
            )
        if normalization == "norm":
            n = (
                math.sqrt(4 * math.pi)
                * paddle.ones(shape=lmax + 1, dtype=betas.dtype)
                / math.sqrt(lmax + 1)
            )
        if normalization == "integral":
            n = paddle.ones(shape=lmax + 1, dtype=betas.dtype)
        if paddle.is_tensor(x=normalization):
            n = normalization
        m = _expand_matrix(range(lmax + 1), dtype=dtype, device=device)
        shb = paddle.einsum("lmj,bj,lmi,l->mbi", m, shb, m, n)
        self.lmax, self.res_beta, self.res_alpha = lmax, res_beta, res_alpha
        self.register_buffer(name="alphas", tensor=alphas)
        self.register_buffer(name="betas", tensor=betas)
        self.register_buffer(name="sha", tensor=sha)
        self.register_buffer(name="shb", tensor=shb)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    @property
    def grid(self):
        beta, alpha = paddle.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        """Evaluate

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(..., (l+1)^2)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``[..., beta, alpha]``
        """
        size = tuple(x.shape)[:-1]
        x = x.reshape(-1, tuple(x.shape)[-1])
        x = paddle.einsum("mbi,zi->zbm", self.shb, x)
        sa, sm = tuple(self.sha.shape)
        if sa >= sm and sa % 2 == 1:
            x = irfft(x, sa)
        else:
            x = paddle.einsum("am,zbm->zba", self.sha, x)
        return x.reshape(*size, *tuple(x.shape)[1:])

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (paddle.randn(shape=self.lmax**2),)} for _ in range(n)]


class FromS2Grid(paddle.nn.Layer):
    """Transform signal on the sphere into spherical tensor

    The inverse transformation of `ToS2Grid`

    Parameters
    ----------
    res : int, tuple of int
        resolution in ``beta`` and in ``alpha``

    lmax : int
    normalization : {'norm', 'component', 'integral'}
    lmax_in : int, optional
    dtype : torch.dtype or None, optional
    device : torch.device or None, optional

    Examples
    --------

    >>> m = FromS2Grid((100, 101), 6)
    >>> x = torch.randn(3, 100, 101)
    >>> m(x).shape
    torch.Size([3, 49])


    `ToS2Grid` and `FromS2Grid` are inverse of each other

    >>> m = FromS2Grid((100, 101), 6)
    >>> k = ToS2Grid(6, (100, 101))
    >>> x = torch.randn(3, 100, 101)
    >>> x = k(m(x))  # remove high frequencies
    >>> y = k(m(x))
    >>> (x - y).abs().max().item() < 1e-4
    True

    Attributes
    ----------
    grid : `torch.Tensor`
        positions on the sphere, tensor of shape ``(res_beta, res_alpha, 3)``

    """

    def __init__(
        self,
        res=None,
        lmax=None,
        normalization="component",
        lmax_in=None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        assert normalization in ["norm", "component", "integral"] or paddle.is_tensor(
            x=normalization
        ), "normalization needs to be 'norm', 'component' or 'integral'"
        if isinstance(res, int) or res is None:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
        else:
            lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)
        if lmax_in is None:
            lmax_in = lmax
        betas, alphas, shb, sha = spherical_harmonics_s2_grid(
            lmax, res_beta, res_alpha, dtype=dtype, device=device
        )
        n = None
        if normalization == "component":
            n = (
                math.sqrt(4 * math.pi)
                * paddle.to_tensor(
                    data=[math.sqrt(2 * l + 1) for l in range(lmax + 1)],
                    dtype=betas.dtype,
                )
                * math.sqrt(lmax_in + 1)
            )
        if normalization == "norm":
            n = (
                math.sqrt(4 * math.pi)
                * paddle.ones(shape=lmax + 1, dtype=betas.dtype)
                * math.sqrt(lmax_in + 1)
            )
        if normalization == "integral":
            n = 4 * math.pi * paddle.ones(shape=lmax + 1, dtype=betas.dtype)
        if paddle.is_tensor(x=normalization):
            n = normalization
        m = _expand_matrix(range(lmax + 1), dtype=dtype, device=device)
        assert res_beta % 2 == 0
        qw = (
            _quadrature_weights(res_beta // 2, dtype=dtype, device=device)
            * res_beta**2
            / res_alpha
        )
        shb = paddle.einsum("lmj,bj,lmi,l,b->mbi", m, shb, m, n, qw)
        self.lmax, self.res_beta, self.res_alpha = lmax, res_beta, res_alpha
        self.register_buffer(name="alphas", tensor=alphas)
        self.register_buffer(name="betas", tensor=betas)
        self.register_buffer(name="sha", tensor=sha)
        self.register_buffer(name="shb", tensor=shb)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    @property
    def grid(self):
        beta, alpha = paddle.meshgrid(self.betas, self.alphas)
        return o3.angles_to_xyz(alpha, beta)

    def forward(self, x):
        """Evaluate

        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``[..., beta, alpha]``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., (l+1)^2)``
        """
        size = tuple(x.shape)[:-2]
        res_beta, res_alpha = tuple(x.shape)[-2:]
        x = x.reshape(-1, res_beta, res_alpha)
        sa, sm = tuple(self.sha.shape)
        if sm <= sa and sa % 2 == 1:
            x = rfft(x, sm // 2)
        else:
            x = paddle.einsum("am,zba->zbm", self.sha, x)
        x = paddle.einsum("mbi,zbm->zi", self.shb, x)
        return x.reshape(*size, tuple(x.shape)[1])

    def _make_tracing_inputs(self, n: int):
        return [
            {"forward": (paddle.randn(shape=[self.res_beta, self.res_alpha]),)}
            for _ in range(n)
        ]
