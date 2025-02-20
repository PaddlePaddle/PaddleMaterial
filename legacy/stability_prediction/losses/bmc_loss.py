import paddle
import paddle.nn as nn


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = -(pred - target.T).pow(y=2) / (2 * noise_var)
    loss = paddle.nn.functional.cross_entropy(
        input=logits, label=paddle.arange(end=tuple(pred.shape)[0])
    )
    loss = loss * (2 * noise_var).detach()
    return loss


class BMCLoss(nn.Layer):
    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        out_0 = paddle.create_parameter(
            shape=paddle.to_tensor(data=init_noise_sigma).shape,
            dtype=paddle.to_tensor(data=init_noise_sigma).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(data=init_noise_sigma)
            ),
        )
        out_0.stop_gradient = False
        self.noise_sigma = out_0

    def forward(self, pred, target):
        if len(pred.shape) == 1:
            pred = pred.reshape([-1, 1])
        if len(target.shape) == 1:
            target = target.reshape([-1, 1])
        print(self.noise_sigma)
        noise_var = self.noise_sigma**2
        return bmc_loss(pred, target, noise_var)
