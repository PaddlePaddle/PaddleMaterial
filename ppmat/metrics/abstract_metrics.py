import paddle
import paddle.nn.functional as F
from paddle.metric import Metric


# =========================
# Concrete Metric Classes
# =========================
class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        self.name = self.__class__.__name__

    def reset(self):
        """
        Reset the internal states.
        """
        self.total_value = paddle.to_tensor(0.0, dtype="float32")
        self.total_samples = paddle.to_tensor(0.0, dtype="float32")

    def update(self, values: paddle.Tensor) -> None:
        """
        Update the metric with new values.

        Args:
            values (paddle.Tensor): Tensor containing the values to sum.
                Shape: (batch_size, ...)
        """
        self.total_value += paddle.sum(values)
        self.total_samples += paddle.shape(values)[0]

    def __call__(self, values: paddle.Tensor):
        self.reset()
        self.update(values)
        summetric = self.total_value / self.total_samples
        # self.reset()
        return summetric

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average of the summed values.

        Returns:
            paddle.Tensor: The average value.
        """
        return self.total_value / self.total_samples


class SumExceptBatchKL(Metric):
    """
    Metric that computes the sum of KL divergences over all dimensions except the batch,
    and calculates the average KL divergence.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        self.name = self.__class__.__name__

    def reset(self):
        """
        Reset the internal states.
        """
        self.total_value = paddle.to_tensor(0.0, dtype="float32")
        self.total_samples = paddle.to_tensor(0.0, dtype="float32")

    def update(self, p: paddle.Tensor, q: paddle.Tensor) -> None:
        """
        Update the metric with new distributions.

        Args:
            p (paddle.Tensor): Target distribution. Shape: (batch_size, ...)
            q (paddle.Tensor): Predicted distribution. Shape: (batch_size, ...)
        """
        self.reset()
        kl = F.kl_div(q, p, reduction="sum")
        self.total_value += kl
        self.total_samples += paddle.shape(p)[0]

    def __call__(self, preds: paddle.Tensor, target: paddle.Tensor):
        """
        Compute the average KL divergences.

        Returns:
            paddle.Tensor: The average cross-entropy loss.
        """
        self.reset()
        self.update(preds, target)
        kl_divergence = self.total_value / self.total_samples
        return kl_divergence

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average KL divergence.

        Returns:
            paddle.Tensor: The average KL divergence.
        """
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        self.name = self.__class__.__name__

    def reset(self):
        """
        Reset the internal states.
        """
        self.total_ce = paddle.to_tensor(0.0, dtype="float32")
        self.total_samples = paddle.to_tensor(0.0, dtype="float32")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """
        Update the metric with new predictions and targets.

        Args:
            preds (paddle.Tensor): Predictions from the model (logits).
                Shape: (batch_size, num_classes) or (batch_size, num_classes, ...)
            target (paddle.Tensor): Ground truth one-hot encoded labels.
                Shape: (batch_size, num_classes) or (batch_size, num_classes, ...)
        """
        # Convert one-hot to class indices
        target = paddle.argmax(target, axis=-1)
        # Compute cross-entropy with sum reduction
        ce = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += ce
        self.total_samples += paddle.shape(preds)[0]

    def __call__(self, preds: paddle.Tensor, target: paddle.Tensor):
        """
        Compute the average cross-entropy loss.

        Returns:
            paddle.Tensor: The average cross-entropy loss.
        """
        self.reset()
        self.update(preds, target)
        ce = self.total_ce / self.total_samples
        self.reset()
        return ce

    def accumulate(self) -> paddle.Tensor:
        return self.accumulate


class NLL(Metric):
    """
    Metric for Negative Log-Likelihood.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        self.name = self.__class__.__name__

    def reset(self):
        """
        Reset the internal states.
        """
        self.total_nll = paddle.to_tensor(0.0, dtype="float32")
        self.total_samples = paddle.to_tensor(0.0, dtype="float32")

    def update(self, value_sum, value_numel) -> None:
        """
        Update the NLL metric with new batch NLL values.
        """
        self.total_nll += value_sum
        self.total_samples += value_numel

    def __call__(self, batch_nll: paddle.Tensor):
        value_sum = paddle.sum(batch_nll)
        value_numel = paddle.numel(batch_nll)
        self.update(value_sum, value_numel)
        return value_sum / value_numel

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average NLL.
        Returns:
            paddle.Tensor: The average NLL over all samples.
        """
        res = self.total_nll / self.total_samples
        self.reset()
        return res
