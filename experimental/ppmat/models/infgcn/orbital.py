import math

import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.infgcn.paddle_utils import * # noqa: F403

from .utils import BroadcastGTOTensor


class GaussianOrbital(paddle.nn.Layer):
    """
    Gaussian-type orbital

    .. math::
        \\psi_{n\\ell m}(\\mathbf{r})=\\sqrt{\\frac{2(2a_n)^{\\ell+3/2}}{\\Gamma(\\ell+3/2)}}
        \\exp(-a_n r^2) r^\\ell Y_{\\ell}^m(\\hat{\\mathbf{r}})

    """

    def __init__(self, gauss_start, gauss_end, num_gauss, lmax=7):
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax
        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src="lc", dst="lcm")
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src="m", dst="lcm")
        self.gauss: paddle.Tensor
        self.lognorm: paddle.Tensor
        self.register_buffer(
            name="gauss",
            tensor=paddle.linspace(start=gauss_start, stop=gauss_end, num=num_gauss),
        )
        self.register_buffer(name="lognorm", tensor=self._generate_lognorm())

    def _generate_lognorm(self):
        power = (paddle.arange(end=self.lmax + 1) + 1.5).unsqueeze(axis=-1)
        numerator = power * paddle.log(x=2 * self.gauss).unsqueeze(axis=0) + math.log(2)
        denominator = paddle.lgamma(x=power)
        lognorm = (numerator - denominator) / 2
        return lognorm.view(-1)

    def forward(self, vec):
        """
        Evaluate the basis functions
        :param vec: un-normalized vectors of (..., 3)
        :return: basis values of (..., (l+1)^2 * c)
        """
        device = vec.place
        r = vec.norm(axis=-1) + 1e-08
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            vec / r[..., None],
            normalize=False,
            normalization="integral",
        )
        r = r.unsqueeze(axis=-1)
        lognorm = self.lognorm * paddle.ones_like(x=r)
        exponent = -self.gauss * (r * r)
        poly = paddle.arange(dtype="float32", end=self.lmax + 1) * paddle.log(x=r)
        log = exponent.unsqueeze(axis=-2) + poly.unsqueeze(axis=-1)
        radial = paddle.exp(x=log.view(*tuple(log.shape)[:-2], -1) + lognorm)
        return self.lc2lcm(radial) * self.m2lcm(spherical)
