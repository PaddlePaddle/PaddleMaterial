import math

import paddle


class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(tuple(x.shape)) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        emb = emb.astype(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb
