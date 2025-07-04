from collections import namedtuple
from math import pi

import paddle
import scipy.signal

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.o3 import FromS2Grid
from ppmat.models.common.e3nn.o3 import ToS2Grid
from ppmat.models.common.e3nn.paddle_utils import *


def _find_peaks_2d(x):
    iii = []
    for i in range(tuple(x.shape)[0]):
        jj, _ = scipy.signal.find_peaks(x[i, :])
        iii += [(i, j) for j in jj]
    jjj = []
    for j in range(tuple(x.shape)[1]):
        ii, _ = scipy.signal.find_peaks(x[:, j])
        jjj += [(i, j) for i in ii]
    return list(set(iii).intersection(set(jjj)))


class SphericalTensor(o3.Irreps):
    """representation of a signal on the sphere

    A `SphericalTensor` contains the coefficients :math:`A^l` of a function :math:`f` defined on the sphere

    .. math::
        f(x) = \\sum_{l=0}^{l_\\mathrm{max}} A^l \\cdot Y^l(x)


    The way this function is transformed by parity :math:`f \\longrightarrow P f` is described by the two parameters :math:`p_v`
    and :math:`p_a`

    .. math::
        (P f)(x) &= p_v f(p_a x)

        &= \\sum_{l=0}^{l_\\mathrm{max}} p_v p_a^l A^l \\cdot Y^l(x)


    Parameters
    ----------
    lmax : int
        :math:`l_\\mathrm{max}`

    p_val : {+1, -1}
        :math:`p_v`

    p_arg : {+1, -1}
        :math:`p_a`


    Examples
    --------

    >>> SphericalTensor(3, 1, 1)
    1x0e+1x1e+1x2e+1x3e

    >>> SphericalTensor(3, 1, -1)
    1x0e+1x1o+1x2e+1x3o
    """

    def __new__(cls, lmax, p_val, p_arg):
        return super().__new__(
            cls, [(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)]
        )

    def with_peaks_at(self, vectors, values=None):
        """Create a spherical tensor with peaks

        The peaks are located in :math:`\\vec r_i` and have amplitude :math:`\\|\\vec r_i \\|`

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\\vec r_i` tensor of shape ``(N, 3)``

        values : `torch.Tensor`, optional
            value on the peak, tensor of shape ``(N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(self.dim,)``

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [3.0, 4.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> s.signal_xyz(x, pos).long()
        tensor([1, 5])

        >>> val = torch.tensor([
        ...     -1.5,
        ...     2.0,
        ... ])
        >>> x = s.with_peaks_at(pos, val)
        >>> s.signal_xyz(x, pos)
        tensor([-1.5000,  2.0000])
        """
        if values is not None:
            vectors, values = paddle.broadcast_tensors(
                input=[vectors, values[..., None]]
            )
            values = values[..., 0]
        if vectors.size == 0:
            return paddle.zeros(shape=tuple(vectors.shape)[:-2] + (self.dim,))
        assert (
            self[0][1].p == 1
        ), "since the value is set by the radii who is even, p_val has to be 1"
        assert vectors.dim() == 2 and tuple(vectors.shape)[1] == 3
        if values is None:
            values = vectors.norm(axis=1)
        vectors = vectors[values != 0]
        values = values[values != 0]
        coeff = o3.spherical_harmonics(self, vectors, normalize=True)
        A = paddle.einsum("ai,bi->ab", coeff, coeff)
        solution = paddle.linalg.lstsq(A, values).solution.reshape(-1)
        assert (
            values - A @ solution
        ).abs().max_func() < 1e-05 * values.abs().max_func()
        return solution @ coeff

    def sum_of_diracs(
        self, positions: paddle.Tensor, values: paddle.Tensor
    ) -> paddle.Tensor:
        """Sum (almost-) dirac deltas

        .. math::

            f(x) = \\sum_i v_i \\delta^L(\\vec r_i)

        where :math:`\\delta^L` is the apporximation of a dirac delta.

        Parameters
        ----------
        positions : `torch.Tensor`
            :math:`\\vec r_i` tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            :math:`v_i` tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(7, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ... ])
        >>> val = torch.tensor([
        ...     -1.0,
        ...     1.0,
        ... ])
        >>> x = s.sum_of_diracs(pos, val)
        >>> s.signal_xyz(x, torch.eye(3)).mul(10.0).round()
        tensor([-10.,  10.,  -0.])

        >>> s.sum_of_diracs(torch.empty(1, 0, 2, 3), torch.empty(2, 0, 1)).shape
        torch.Size([2, 0, 64])

        >>> s.sum_of_diracs(torch.randn(1, 3, 2, 3), torch.randn(2, 1, 1)).shape
        torch.Size([2, 3, 64])
        """
        positions, values = paddle.broadcast_tensors(
            input=[positions, values[..., None]]
        )
        values = values[..., 0]
        if positions.size == 0:
            return paddle.zeros(shape=tuple(values.shape)[:-1] + (self.dim,))
        y = o3.spherical_harmonics(self, positions, True)
        v = values[..., None]
        return 4 * pi / (self.lmax + 1) ** 2 * (y * v).sum(axis=-2)

    def from_samples_on_s2(
        self, positions: paddle.Tensor, values: paddle.Tensor, res=100
    ) -> paddle.Tensor:
        """Convert a set of position on the sphere and values into a spherical tensor

        Parameters
        ----------
        positions : `torch.Tensor`
            tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(2, 1, 1)
        >>> pos = torch.tensor([
        ...     [
        ...         [0.0, 0.0, 1.0],
        ...         [0.0, 0.0, -1.0],
        ...     ],
        ...     [
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, -1.0, 0.0],
        ...     ],
        ... ], dtype=torch.float64)
        >>> val = torch.tensor([
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ... ], dtype=torch.float64)
        >>> s.from_samples_on_s2(pos, val, res=200).long()
        tensor([[0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0]])

        >>> pos = torch.empty(2, 0, 10, 3)
        >>> val = torch.empty(2, 0, 10)
        >>> s.from_samples_on_s2(pos, val)
        tensor([], size=(2, 0, 9))

        """
        positions, values = paddle.broadcast_tensors(
            input=[positions, values[..., None]]
        )
        values = values[..., 0]
        if positions.size == 0:
            return paddle.zeros(shape=tuple(values.shape)[:-1] + (self.dim,))
        positions = paddle.nn.functional.normalize(x=positions, axis=-1)
        size = tuple(positions.shape)[:-2]
        n = tuple(positions.shape)[-2]
        positions = positions.reshape(-1, n, 3)
        values = values.reshape(-1, n)
        s2 = FromS2Grid(
            res=res,
            lmax=self.lmax,
            normalization="integral",
            dtype=values.dtype,
            device=values.place,
        )
        pos = s2.grid.reshape(1, -1, 3)
        cd = paddle.cdist(x=pos, y=positions)
        i = paddle.arange(end=len(values)).view(-1, 1)
        j = cd.argmin(axis=2)
        val = values[i, j]
        val = val.reshape(*size, s2.res_beta, s2.res_alpha)
        return s2(val)

    def norms(self, signal):
        """The norms of each l component

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(..., dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., lmax+1)``

        Examples
        --------
        Examples
        --------
        >>> s = SphericalTensor(1, 1, -1)
        >>> s.norms(torch.tensor([1.5, 0.0, 3.0, 4.0]))
        tensor([1.5000, 5.0000])
        """
        i = 0
        norms = []
        for _, ir in self:
            norms += [signal[..., i : i + ir.dim].norm(axis=-1)]
            i += ir.dim
        return paddle.stack(x=norms, axis=-1)

    def signal_xyz(self, signal, r):
        """Evaluate the signal on given points on the sphere

        .. math::

            f(\\vec x / \\|\\vec x\\|)

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(*A, self.dim)``

        r : `torch.Tensor`
            tensor of shape ``(*B, 3)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(*A, *B)``

        Examples
        --------
        >>> s = SphericalTensor(3, 1, -1)
        >>> s.signal_xyz(s.randn(2, 1, 3, -1), torch.randn(2, 4, 3)).shape
        torch.Size([2, 1, 3, 2, 4])
        """
        sh = o3.spherical_harmonics(self, r, normalize=True)
        dim = (self.lmax + 1) ** 2
        output = paddle.einsum(
            "bi,ai->ab", sh.reshape(-1, dim), signal.reshape(-1, dim)
        )
        return output.reshape(tuple(signal.shape)[:-1] + tuple(r.shape)[:-1])

    def signal_on_grid(self, signal, res=100, normalization="integral"):
        """Evaluate the signal on a grid on the sphere"""
        Ret = namedtuple("Return", "grid, values")
        s2 = ToS2Grid(lmax=self.lmax, res=res, normalization=normalization)
        return Ret(s2.grid, s2(signal))

    def plotly_surface(
        self,
        signals,
        centers=None,
        res=100,
        radius=True,
        relu=False,
        normalization="integral",
    ):
        """Create traces for plotly

        Examples
        --------
        >>> import plotly.graph_objects as go
        >>> x = SphericalTensor(4, +1, +1)
        >>> traces = x.plotly_surface(x.randn(-1))
        >>> traces = [go.Surface(**d) for d in traces]
        >>> fig = go.Figure(data=traces)
        """
        signals = signals.reshape(-1, self.dim)
        if centers is None:
            centers = [None] * len(signals)
        else:
            centers = centers.reshape(-1, 3)
        traces = []
        for signal, center in zip(signals, centers):
            r, f = self.plot(signal, center, res, radius, relu, normalization)
            traces += [
                dict(
                    x=r[:, :, 0].numpy(),
                    y=r[:, :, 1].numpy(),
                    z=r[:, :, 2].numpy(),
                    surfacecolor=f.numpy(),
                )
            ]
        return traces

    def plot(
        self,
        signal,
        center=None,
        res=100,
        radius=True,
        relu=False,
        normalization="integral",
    ):
        """Create surface in order to make a plot"""
        assert signal.dim() == 1
        r, f = self.signal_on_grid(signal, res, normalization)
        f = f.relu() if relu else f
        r[0] = paddle.to_tensor(data=[0.0, 1.0, 0.0], dtype=r.dtype)
        r[-1] = paddle.to_tensor(data=[0.0, -1.0, 0.0], dtype=r.dtype)
        f[0] = f[0].mean()
        f[-1] = f[-1].mean()
        r = paddle.concat(x=[r, r[:, :1]], axis=1)
        f = paddle.concat(x=[f, f[:, :1]], axis=1)
        if radius:
            r *= f.abs().unsqueeze(axis=-1)
        if center is not None:
            r += center
        return r, f

    def find_peaks(self, signal, res=100):
        """Locate peaks on the sphere

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [4.0, 0.0, 4.0],
        ...     [0.0, 5.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> pos, val = s.find_peaks(x)
        >>> pos[val > 4.0].mul(10).round().abs()
        tensor([[ 7.,  0.,  7.],
                [ 0., 10.,  0.]])
        >>> val[val > 4.0].mul(10).round().abs()
        tensor([57., 50.])
        """
        x1, f1 = self.signal_on_grid(signal, res)
        abc = paddle.to_tensor(data=[pi / 2, pi / 2, pi / 2])
        R = o3.angles_to_matrix(*abc)
        D = self.D_from_matrix(R)
        r_signal = D @ signal
        rx2, f2 = self.signal_on_grid(r_signal, res)
        x2 = paddle.einsum("ij,baj->bai", R.T, rx2)
        ij = _find_peaks_2d(f1)
        x1p = paddle.stack(x=[x1[i, j] for i, j in ij])
        f1p = paddle.stack(x=[f1[i, j] for i, j in ij])
        ij = _find_peaks_2d(f2)
        x2p = paddle.stack(x=[x2[i, j] for i, j in ij])
        f2p = paddle.stack(x=[f2[i, j] for i, j in ij])
        mask = paddle.cdist(x=x1p, y=x2p) < 2 * pi / res
        x = paddle.concat(x=[x1p[mask.sum(axis=1) == 0], x2p])
        f = paddle.concat(x=[f1p[mask.sum(axis=1) == 0], f2p])
        return x, f
