import math

import numpy as np
import paddle


def uniform_sample_t(batch_size, timesteps):
    times = np.random.choice(np.arange(1, timesteps + 1), batch_size)
    return paddle.to_tensor(times)


class SinusoidalEmbeddings(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.embeddings = paddle.exp(x=paddle.arange(end=half_dim) * -embeddings)

    def forward(self, origin):
        origin = origin.astype(paddle.get_default_dtype())
        embeddings = origin[:, None] * self.embeddings[None, :]
        embeddings = paddle.concat(x=(embeddings.sin(), embeddings.cos()), axis=-1)
        return embeddings

class SinusoidalTimeEmbeddings(SinusoidalEmbeddings):
    def __init__(dim):
        super().__init__(dim)


class SinusoidalPosEmbeddings(SinusoidalEmbeddings):
    def __init__(dim):
        super().__init__(dim)