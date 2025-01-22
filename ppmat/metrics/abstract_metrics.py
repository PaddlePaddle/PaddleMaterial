import paddle
import paddle.nn.functional as F
from paddle.metric import Metric

# =========================
# Abstract Metrics Classes
# =========================


class TrainAbstractMetricsDiscrete(paddle.nn.Layer):
    """
    Abstract base class for discrete training metrics.
    This class serves as a base for specific metric implementations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        """
        Forward method to compute metrics.

        Args:
            masked_pred_X (paddle.Tensor): Predicted X with masking applied.
            masked_pred_E (paddle.Tensor): Predicted E with masking applied.
            true_X (paddle.Tensor): True X values.
            true_E (paddle.Tensor): True E values.
            log (bool): Whether to log the metrics.
        """
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        masked_pred_epsX,
        masked_pred_epsE,
        pred_y,
        true_epsX,
        true_epsE,
        true_y,
        log,
    ):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


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

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average of the summed values.

        Returns:
            paddle.Tensor: The average value.
        """
        return self.total_value / self.total_samples


class SumExceptBatchMSE(Metric):
    """
    Metric that computes the sum of squared errors over all dimensions except the batch,
    and calculates the mean squared error.
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
        self.sum_squared_error = paddle.to_tensor(0.0, dtype="float32")
        self.total = paddle.to_tensor(0.0, dtype="float32")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """
        Update the metric with new predictions and targets.

        Args:
            preds (paddle.Tensor): Predictions from the model. Shape: (batch_size, ...)
            target (paddle.Tensor): Ground truth values. Shape: (batch_size, ...)
        """
        assert preds.shape == target.shape, "preds and target must have the same shape."
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)
        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: paddle.Tensor, target: paddle.Tensor):
        """
        Compute sum of squared errors and number of observations.

        Args:
            preds (paddle.Tensor): Predictions from the model.
            target (paddle.Tensor): Ground truth values.

        Returns:
            tuple: (sum_squared_error, n_obs)
        """
        diff = preds - target
        sum_squared_error = paddle.sum(diff * diff)
        n_obs = paddle.shape(preds)[0]
        return sum_squared_error, n_obs

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the mean squared error.

        Returns:
            paddle.Tensor: The mean squared error.
        """
        return self.sum_squared_error / self.total


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
        kl = F.kl_div(q, p, reduction="sum")
        self.total_value += kl
        self.total_samples += paddle.shape(p)[0]

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
        self.accumulate = paddle.to_tensor(0.0, dtype="float32")

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
        self.update(preds, target)
        self.accumulate += self.total_ce
        loss = self.total_ce / self.total_samples
        self.reset()
        return loss

    def accumulate(self) -> paddle.Tensor:
        return self.accumulate


class ProbabilityMetric(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def name(self):
        self.name = self.__class__.__name__

    def reset(self):
        """
        Reset the internal states.
        """
        self.prob = paddle.to_tensor(0.0, dtype="float32")
        self.total = paddle.to_tensor(0.0, dtype="float32")

    def update(self, preds: paddle.Tensor) -> None:
        """
        Update the probability metric with new predictions.

        Args:
            preds (paddle.Tensor): Predicted probabilities. Shape: (batch_size, ...)
        """
        self.prob += paddle.sum(preds)
        self.total += paddle.numel(preds)

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average predicted probability.

        Returns:
            paddle.Tensor: The average predicted probability.
        """
        return self.prob / self.total


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

    def update(self, batch_nll: paddle.Tensor) -> None:
        """
        Update the NLL metric with new batch NLL values.

        Args:
            batch_nll (paddle.Tensor): NLL values for the current batch.
                Shape: (batch_size, ...)
        """
        self.total_nll += paddle.sum(batch_nll)
        self.total_samples += paddle.numel(batch_nll)

    def accumulate(self) -> paddle.Tensor:
        """
        Compute the average NLL.

        Returns:
            paddle.Tensor: The average NLL over all samples.
        """
        return self.total_nll / self.total_samples
