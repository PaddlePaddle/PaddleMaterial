import paddle
import paddle.nn as nn

from ppmat.metrics.abstract_metrics import CrossEntropyMetric


class TrainLossDiscrete(nn.Layer):
    """Train with Cross Entropy"""

    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(
        self,
        masked_pred_X,
        masked_pred_E,
        pred_y,
        true_X,
        true_E,
        true_y,
    ):
        """
        Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        """
        # reshape tensor
        true_X = paddle.reshape(true_X, [-1, true_X.shape[-1]])  # (bs * n, dx)
        true_E = paddle.reshape(true_E, [-1, true_E.shape[-1]])  # (bs * n * n, de)
        masked_pred_X = paddle.reshape(
            masked_pred_X, [-1, masked_pred_X.shape[-1]]
        )  # (bs * n, dx)
        masked_pred_E = paddle.reshape(
            masked_pred_E, [-1, masked_pred_E.shape[-1]]
        )  # (bs * n * n, de)

        # Apply mask to remove masked rows
        mask_X = paddle.sum(true_X != 0.0, axis=-1) > 0  # (bs * n,)
        mask_E = paddle.sum(true_E != 0.0, axis=-1) > 0  # (bs * n * n,)

        flat_true_X = true_X[mask_X]
        flat_pred_X = masked_pred_X[mask_X]

        flat_true_E = true_E[mask_E]
        flat_pred_E = masked_pred_E[mask_E]

        # 计算交叉熵损失
        loss_X = (
            self.node_loss(flat_pred_X, flat_true_X)
            if true_X.numel() > 0
            else paddle.to_tensor(0.0)
        )
        loss_E = (
            self.edge_loss(flat_pred_E, flat_true_E)
            if true_E.numel() > 0
            else paddle.to_tensor(0.0)
        )
        loss_y = (
            self.y_loss(pred_y, true_y) if true_y.numel() > 0 else paddle.to_tensor(0.0)
        )

        # 返回加权损失
        Sloss = loss_X + loss_E + loss_y
        Wloss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

        return {
            "train_loss": Wloss,
            "batch_CE": Sloss,
            "X_CE": loss_X,
            "E_CE": loss_E,
            "Y_CE": loss_y,
        }

    def reset(self):
        # 重置所有交叉熵指标
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.accumulate().item()
            if self.node_loss.total_samples > 0
            else -1
        )
        epoch_edge_loss = (
            self.edge_loss.accumulate().item()
            if self.edge_loss.total_samples > 0
            else -1
        )
        epoch_y_loss = (
            self.y_loss.accumulate().item() if self.y_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
        }
        return to_log
