import paddle
import paddle.nn as nn


class Xtoy(nn.Layer):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * dx, out_features=dy)

    def forward(self, X, x_mask):
        """X: bs, n, dx."""
        x_mask = paddle.expand(x_mask, shape=[-1, -1, X.shape[-1]])
        float_imask = 1 - x_mask.astype("float32")
        m = paddle.sum(X, axis=1) / paddle.sum(x_mask, axis=1)
        mi = paddle.min(X + 1e6 * float_imask, axis=1)
        ma = paddle.max(X - 1e6 * float_imask, axis=1)
        std = paddle.sum((X - m.unsqueeze(1)) ** 2 * x_mask, axis=1) / paddle.sum(
            x_mask, axis=1
        )
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


class Etoy(nn.Layer):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = paddle.nn.Linear(in_features=4 * d, out_features=dy)

    def forward(self, E, e_mask1, e_mask2):
        """
        E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        mask = paddle.expand(e_mask1 * e_mask2, shape=[-1, -1, -1, E.shape[-1]])
        float_imask = 1 - mask.astype("float32")
        divide = paddle.sum(mask, axis=(1, 2))
        m = paddle.sum(E, axis=(1, 2)) / divide
        mi = paddle.min(paddle.min(E + 1e6 * float_imask, axis=2), axis=1)
        ma = paddle.max(paddle.max(E - 1e6 * float_imask, axis=2), axis=1)
        std = (
            paddle.sum((E - m.unsqueeze(1).unsqueeze(1)) ** 2 * mask, axis=(1, 2))
            / divide
        )
        z = paddle.concat([m, mi, ma, std], axis=1)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, axis=-1):
    """
    Perform softmax over masked values in `x`.

    Args:
        x: Tensor, the input data.
        mask: Tensor, the binary mask of the same shape as `x`.
        axis: The axis to apply softmax.

    Returns:
        Tensor with masked softmax applied.
    """
    if paddle.sum(mask) == 0:
        return x

    # TODO: ndim check: only support adding dimensions backwards now
    x_dims = x.ndim
    mask_dims = mask.ndim
    if mask_dims < x_dims:
        diff = x_dims - mask_dims
        mask = paddle.unsqueeze(mask, axis=[-1] * diff)
        repeat_times = [1] * mask_dims + [x.shape[i] for i in range(mask_dims, x_dims)]
        mask = paddle.tile(mask, repeat_times=repeat_times)

    x_masked = x.clone()
    x_masked = paddle.where(
        mask == 0, paddle.to_tensor(-float("inf"), dtype=x.dtype), x_masked
    )

    return paddle.nn.functional.softmax(x_masked, axis=axis)
