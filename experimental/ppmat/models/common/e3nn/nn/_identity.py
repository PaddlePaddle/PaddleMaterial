import paddle

from ppmat.models.common.e3nn import o3


class Identity(paddle.nn.Layer):
    """Identity operation

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`

    irreps_out : `e3nn.o3.Irreps`
    """

    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        assert self.irreps_in == self.irreps_out
        output_mask = paddle.concat(
            x=[paddle.ones(shape=mul * (2 * l + 1)) for mul, (l, _p) in self.irreps_out]
        )
        self.register_buffer(name="output_mask", tensor=output_mask)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        """evaluate"""
        return features
