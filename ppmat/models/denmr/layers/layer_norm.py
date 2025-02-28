import paddle

"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""


class LayerNorm(paddle.nn.Layer):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=d_model)
        )
        self.beta = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=d_model)
        )
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, unbiased=False, keepdim=True)
        out = (x - mean) / paddle.sqrt(x=var + self.eps)
        out = self.gamma * out + self.beta
        return out
