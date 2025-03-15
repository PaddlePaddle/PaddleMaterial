import paddle

from ppmat.models.denmr.utils.clip_utils import dim2perm  # noqa

from .scale_dot_product_attention import ScaleDotProductAttention

"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""


class MultiHeadAttention(paddle.nn.Layer):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = paddle.nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = paddle.nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = paddle.nn.Linear(in_features=d_model, out_features=d_model)
        self.w_concat = paddle.nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tuple(tensor.shape)
        d_tensor = d_model // self.n_head
        tensor = tensor.reshape([batch_size, length, self.n_head, d_tensor]).transpose(
            perm=dim2perm(
                tensor.reshape([batch_size, length, self.n_head, d_tensor]).ndim, 1, 2
            )
        )
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tuple(tensor.shape)
        d_model = head * d_tensor
        tensor = (
            tensor.transpose(perm=dim2perm(tensor.ndim, 1, 2))
            .contiguous()
            .reshape([batch_size, length, d_model])
        )
        return tensor
