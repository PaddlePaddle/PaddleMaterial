import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.paddle_utils import *


class BatchNorm(paddle.nn.Layer):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    instance : bool
        apply instance norm instead of batch norm
    """

    def __init__(
        self,
        irreps,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        reduce="mean",
        instance=False,
        normalization="component",
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance
        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        if self.instance:
            self.register_buffer(name="running_mean", tensor=None)
            self.register_buffer(name="running_var", tensor=None)
        else:
            self.register_buffer(
                name="running_mean", tensor=paddle.zeros(shape=num_scalar)
            )
            self.register_buffer(
                name="running_var", tensor=paddle.ones(shape=num_features)
            )
        if affine:
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.ones(shape=num_features)
            )
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=num_scalar)
            )
        else:
            self.add_parameter(name="weight", parameter=None)
            self.add_parameter(name="bias", parameter=None)
        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce
        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update.detach()

    def forward(self, input):
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
        batch, *size, dim = tuple(input.shape)
        input = input.reshape(batch, -1, dim)
        if self.training and not self.instance:
            new_means = []
            new_vars = []
        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0
        for mul, ir in self.irreps:
            d = ir.dim
            field = input[:, :, ix : ix + mul * d]
            ix += mul * d
            field = field.reshape(batch, -1, mul, d)
            if ir.is_scalar():
                if self.training or self.instance:
                    if self.instance:
                        field_mean = field.mean(axis=1).reshape(batch, mul)
                    else:
                        field_mean = field.mean(axis=[0, 1]).reshape(mul)
                        new_means.append(
                            self._roll_avg(
                                self.running_mean[irm : irm + mul], field_mean
                            )
                        )
                else:
                    field_mean = self.running_mean[irm : irm + mul]
                irm += mul
                field = field - field_mean.reshape(-1, 1, mul, 1)
            if self.training or self.instance:
                if self.normalization == "norm":
                    field_norm = field.pow(y=2).sum(axis=3)
                elif self.normalization == "component":
                    field_norm = field.pow(y=2).mean(axis=3)
                else:
                    raise ValueError(
                        "Invalid normalization option {}".format(self.normalization)
                    )
                if self.reduce == "mean":
                    field_norm = field_norm.mean(axis=1)
                elif self.reduce == "max":
                    field_norm = field_norm.max_func(1).values
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))
                if not self.instance:
                    field_norm = field_norm.mean(axis=0)
                    new_vars.append(
                        self._roll_avg(self.running_var[irv : irv + mul], field_norm)
                    )
            else:
                field_norm = self.running_var[irv : irv + mul]
            irv += mul
            field_norm = (field_norm + self.eps).pow(y=-0.5)
            if self.affine:
                weight = self.weight[iw : iw + mul]
                iw += mul
                field_norm = field_norm * weight
            field = field * field_norm.reshape(-1, 1, mul, 1)
            if self.affine and ir.is_scalar():
                bias = self.bias[ib : ib + mul]
                ib += mul
                field += bias.reshape(mul, 1)
            fields.append(field.reshape(batch, -1, mul * d))
        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)
        if self.training and not self.instance:
            assert irm == self.running_mean.size
            assert irv == self.running_var.shape[0]
        if self.affine:
            assert iw == self.weight.shape[0]
            assert ib == self.bias.size
        if self.training and not self.instance:
            if len(new_means) > 0:
                paddle.assign(paddle.concat(x=new_means), output=self.running_mean)
            if len(new_vars) > 0:
                paddle.assign(paddle.concat(x=new_vars), output=self.running_var)
        output = paddle.concat(x=fields, axis=2)
        return output.reshape(batch, *size, dim)
