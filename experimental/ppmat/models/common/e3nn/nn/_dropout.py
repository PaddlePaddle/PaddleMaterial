import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.paddle_utils import *


class Dropout(paddle.nn.Layer):
    """Equivariant Dropout

        :math:`A_{zai}` is the input and :math:`B_{zai}` is the output where

        - ``z`` is the batch index
        - ``a`` any non-batch and non-irrep index
        - ``i`` is the irrep index, for instance if ``irreps="0e + 2x1e"`` then ``i=2`` select the *second vector*

        .. math::

            B_{zai} =
    rac{x_{zi}}{1-p} A_{zai}

        where :math:`p` is the dropout probability and :math:`x` is a Bernoulli random variable with parameter :math:`1-p`.

        Parameters
        ----------
        irreps : `o3.Irreps`
            representation

        p : float
            probability to drop
    """

    def __init__(self, irreps, p):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, p={self.p})"

    def forward(self, x):
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        if not self.training:
            return x
        batch = tuple(x.shape)[0]
        noises = []
        for mul, (l, _p) in self.irreps:
            dim = 2 * l + 1
            noise = paddle.empty(shape=[batch, mul], dtype=x.dtype)
            if self.p >= 1:
                noise.fill_(value=0)
            elif self.p <= 0:
                noise.fill_(value=1)
            else:
                """Not Support auto convert *.bernoulli_, please judge whether it is Pytorch API and convert by yourself"""
                noise = paddle.bernoulli(paddle.full_like(noise, 1 - self.p))
                noise = noise / (1 - self.p)
            noise = (
                noise[:, :, None].expand(shape=[-1, -1, dim]).reshape(batch, mul * dim)
            )
            noises.append(noise)
        noise = paddle.concat(x=noises, axis=-1)
        while noise.dim() < x.dim():
            noise = noise[:, None]
        return x * noise
