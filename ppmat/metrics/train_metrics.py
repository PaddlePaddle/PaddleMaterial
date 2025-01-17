import paddle
import paddle.nn as nn
import wandb

from ppmat.metrics.abstract_metrics import CrossEntropyMetric
from ppmat.metrics.abstract_metrics import SumExceptBatchMSE


class NodeMSE(SumExceptBatchMSE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EdgeMSE(SumExceptBatchMSE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TrainLoss(nn.Layer):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = (
            SumExceptBatchMSE()
        )  # 使用 SumExceptBatchMSE 代替 MeanSquaredError

    def forward(
        self,
        masked_pred_epsX,
        masked_pred_epsE,
        pred_y,
        true_epsX,
        true_epsE,
        true_y,
        log: bool,
    ):
        # 计算 MSE，仅在对应的真实值存在时计算
        mse_X = (
            self.train_node_mse(masked_pred_epsX, true_epsX)
            if paddle.sum(true_epsX).item() > 0
            else paddle.to_tensor(0.0)
        )
        mse_E = (
            self.train_edge_mse(masked_pred_epsE, true_epsE)
            if paddle.sum(true_epsE).item() > 0
            else paddle.to_tensor(0.0)
        )
        mse_y = (
            self.train_y_mse(pred_y, true_y)
            if paddle.sum(true_y).item() > 0
            else paddle.to_tensor(0.0)
        )

        # 总损失
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {
                "train_loss/batch_mse": mse.detach(),
                "train_loss/node_MSE": self.train_node_mse.accumulate().detach(),
                "train_loss/edge_MSE": self.train_edge_mse.accumulate().detach(),
                "train_loss/y_mse": self.train_y_mse.accumulate().detach(),
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        # 重置所有 MSE 指标
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        # 计算并记录每个 epoch 的指标
        epoch_node_mse = (
            self.train_node_mse.accumulate().item()
            if self.train_node_mse.total > 0
            else -1
        )
        epoch_edge_mse = (
            self.train_edge_mse.accumulate().item()
            if self.train_edge_mse.total > 0
            else -1
        )
        epoch_y_mse = (
            self.train_y_mse.accumulate().item() if self.train_y_mse.total > 0 else -1
        )

        to_log = {
            "train_epoch/epoch_X_mse": epoch_node_mse,
            "train_epoch/epoch_E_mse": epoch_edge_mse,
            "train_epoch/epoch_y_mse": epoch_y_mse,
        }
        if wandb.run:
            wandb.log(to_log)
        return to_log


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
        log: bool = False,
    ):
        """
        Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean.
        """
        # 重塑张量
        true_X = paddle.reshape(true_X, [-1, true_X.shape[-1]])  # (bs * n, dx)
        true_E = paddle.reshape(true_E, [-1, true_E.shape[-1]])  # (bs * n * n, de)
        masked_pred_X = paddle.reshape(
            masked_pred_X, [-1, masked_pred_X.shape[-1]]
        )  # (bs * n, dx)
        masked_pred_E = paddle.reshape(
            masked_pred_E, [-1, masked_pred_E.shape[-1]]
        )  # (bs * n * n, de)

        # 应用掩码，移除被掩盖的行
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

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": self.node_loss.accumulate().item()
                if true_X.numel() > 0
                else -1,
                "train_loss/E_CE": self.edge_loss.accumulate().item()
                if true_E.numel() > 0
                else -1,
                "train_loss/y_CE": self.y_loss.accumulate().item()
                if true_y.numel() > 0
                else -1,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        # 返回加权损失
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        # 重置所有交叉熵指标
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        # 计算并记录每个 epoch 的指标
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
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
