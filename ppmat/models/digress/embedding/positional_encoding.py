import paddle

"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""


class PositionalEncoding(paddle.nn.Layer):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = paddle.zeros(shape=[max_len, d_model])
        self.encoding.stop_gradient = not False
        pos = paddle.arange(start=0, end=max_len)
        pos = pos.astype(dtype="float32").unsqueeze(axis=1)
        _2i = paddle.arange(start=0, end=d_model, step=2).astype(dtype="float32")
        self.encoding[:, 0::2] = paddle.sin(x=pos / 10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = paddle.cos(x=pos / 10000 ** (_2i / d_model))

    def forward(self, x):
        batch_size, seq_len = tuple(x.shape)
        return self.encoding[:seq_len, :]
