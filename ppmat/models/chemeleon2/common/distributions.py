import paddle


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * paddle.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        # Determine which axes to sum over based on tensor dimensionality
        # For 4D tensors (images): sum over [1, 2, 3]
        # For 2D tensors (latent vectors): sum over [1]
        if self.mean.ndim == 4:
            sum_axis = [1, 2, 3]
        else:
            sum_axis = [1]

        if other is None:
            return 0.5 * paddle.sum(
                paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                axis=sum_axis
            )
        else:
            return 0.5 * paddle.sum(
                paddle.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                axis=sum_axis
            )

    def mode(self):
        return self.mean
