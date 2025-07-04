from typing import Optional

import numpy as np
import paddle


class RBFExpansion(paddle.nn.Layer):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            name="centers",
            tensor=paddle.linspace(start=self.vmin, stop=self.vmax, num=self.bins),
        )
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / lengthscale**2

    def forward(self, distance: paddle.Tensor) -> paddle.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return paddle.exp(
            x=-self.gamma * (distance.unsqueeze(axis=1) - self.centers) ** 2
        )
