import math

import paddle


class SinusoidalTimeEmbeddings(paddle.nn.Layer):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.place
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(x=paddle.arange(end=half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = paddle.concat(x=(embeddings.sin(), embeddings.cos()), axis=-1)
        return embeddings
