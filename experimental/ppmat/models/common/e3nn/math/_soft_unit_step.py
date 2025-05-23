import paddle


class _SoftUnitStep(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = paddle.zeros_like(x=x)
        m = x > 0.0
        y[m] = (-1 / x[m]).exp()
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensor()
        dx = paddle.zeros_like(x=x)
        m = x > 0.0
        xm = x[m]
        dx[m] = (-1 / xm).exp() / xm.pow(y=2)
        return dx * dy


def soft_unit_step(x):
    """smooth :math:`C^\\infty` version of the unit step function

    .. math::

        x \\mapsto \\theta(x) e^{-1/x}


    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(...)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_unit_step
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = torch.linspace(-1.0, 10.0, 1000)
        plt.plot(x, soft_unit_step(x));
    """
    return _SoftUnitStep.apply(x)
