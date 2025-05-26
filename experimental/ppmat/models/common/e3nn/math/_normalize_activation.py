import paddle

from ppmat.models.common.e3nn.paddle_utils import *
from ppmat.models.common.e3nn.util import explicit_default_types


def moment(f, n, dtype=None, device=None):
    """
    compute n th moment
    <f(z)^n> for z normal
    """
    dtype, device = explicit_default_types(dtype, device)
    gen = paddle.framework.core.default_cpu_generator().manual_seed(0)
    z = paddle.randn(shape=[1000000], dtype="float64").to(dtype=dtype, device=device)
    return f(z).pow(y=n).mean()


class normalize2mom(paddle.nn.Layer):
    _is_id: bool
    cst: float

    def __init__(self, f, dtype=None, device=None):
        super().__init__()
        if device is None and isinstance(f, paddle.nn.Layer):
            from e3nn.util._argtools import _get_device

            device = _get_device(f)
        with paddle.no_grad():
            cst = moment(f, 2, dtype="float64", device="cpu").pow(y=-0.5).item()
        if abs(cst - 1) < 0.0001:
            self._is_id = True
        else:
            self._is_id = False
        self.f = f
        self.cst = cst

    def forward(self, x):
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x).multiply(
                paddle.to_tensor(self.cst)
            )  

    @staticmethod
    def _make_tracing_inputs(n: int):
        return [{"forward": (paddle.zeros(shape=(1,)),)}]
