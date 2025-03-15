import math

import paddle

from ppmat.models.denmr.utils.clip_utils import dim2perm  # noqa

"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""


class ScaleDotProductAttention(paddle.nn.Layer):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = paddle.nn.Softmax(axis=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = tuple(k.shape)
        k_t = k.transpose(perm=dim2perm(k.ndim, 2, 3))
        score = q @ k_t / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask=mask == 0, value=-10000)
        score = self.softmax(score)
        v = score @ v
        return v, score
