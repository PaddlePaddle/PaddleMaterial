# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
import paddle


def uniform_sample_t(batch_size, timesteps):
    times = np.random.choice(np.arange(0, timesteps), batch_size)
    return paddle.to_tensor(times)


class SinusoidalTimeEmbeddings(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.embeddings = paddle.exp(x=paddle.arange(end=half_dim) * -embeddings)

    def forward(self, time):
        time = time.astype(paddle.get_default_dtype())
        embeddings = time[:, None] * self.embeddings[None, :]
        embeddings = paddle.concat(x=(embeddings.sin(), embeddings.cos()), axis=-1)
        return embeddings


class NoiseLevelEncoding(paddle.nn.Layer):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = paddle.exp(
            x=paddle.arange(start=0, end=d_model, step=2)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer(name="div_term", tensor=div_term)

    def forward(self, t: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = paddle.zeros(shape=(tuple(t.shape)[0], self.d_model))
        x[:, 0::2] = paddle.sin(x=t[:, None] * self.div_term[None])
        x[:, 1::2] = paddle.cos(x=t[:, None] * self.div_term[None])
        return self.dropout(x)
