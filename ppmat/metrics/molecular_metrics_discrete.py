import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# ================================
# 1. 自定义 MetricCollection 类
# ================================


class PaddleMetricCollection:
    """
    用于收集多个自定义 Metric，并统一调用 update / compute / reset。
    功能类似 TorchMetrics 中的 MetricCollection。
    """

    def __init__(self, metrics_list):
        """
        metrics_list: 一个列表，其中每个元素都是 Paddle Metric 对象
                      (比如本示例中自定义的 CEPerClass 子类实例)。
        """
        self.metrics_list = metrics_list

    def update(self, preds, target):
        """调用所有 metric 的 update 方法。"""
        for metric in self.metrics_list:
            metric.update(preds, target)

    def compute(self):
        """
        与 TorchMetrics 中的 compute 类似，返回一个字典：
        { metric_name: metric结果, ... }
        """
        results = {}
        for metric in self.metrics_list:
            # 使用 metric.name 来区别不同指标
            results[metric.name()] = metric.accumulate()
        return results

    def reset(self):
        """重置所有 metric 的内部状态。"""
        for metric in self.metrics_list:
            metric.reset()


# ================================
# 2. 自定义 CEPerClass (Paddle Metric)
# ================================


class CEPerClass(paddle.metric.Metric):
    """
    Paddle 中自定义的交叉熵度量示例，用于对特定类(class_id)的预测进行二进制交叉熵计算。
    """

    def __init__(self, class_id):
        super().__init__()
        self._class_id = class_id
        # 这里用普通 Python 变量累加，也可使用 paddle.to_tensor(0.0)
        self._total_ce = 0.0
        self._total_samples = 0

        # softmax 用于多分类变成相应类别的概率
        self.softmax = nn.Softmax(axis=-1)
        # BCELoss 在 Paddle 中通过 paddle.nn.BCELoss(reduction='sum') 或 F.binary_cross_entropy
        self.bce_loss = nn.BCELoss(reduction="sum")

    def name(self):
        """
        Paddle 中可以通过 name() 方法返回该指标的名称。
        也可以在 __init__ 中设置 self._name = xxx，然后在这里 return。
        """
        # 例如返回 "ce_class_{id}"；在集合中也可结合外层类来命名。
        return f"CE_class_{self._class_id}"

    def update(self, preds, target):
        """
        批量更新内部状态。
        preds.shape: (bs, n, d) 或 (bs, n, n, d)
        target.shape: 同 preds，一般 (bs, n, d) 或 (bs, n, n, d)
        """
        # 将 (bs, n, ...) 展开到 (X, d)
        # Paddle 中可以用 reshape，注意与 torch.reshape 对应
        last_dim = target.shape[-1]  # d
        target = paddle.reshape(target, [-1, last_dim])

        # 创建 mask，排除全0行： (target != 0).any(axis=-1)
        mask = paddle.any(target != 0.0, axis=-1)  # bool张量

        # 取特定类别的概率
        prob_full = self.softmax(preds)  # preds => softmax => [batch, ..., d]
        # 先 reshape preds => [X, d]
        prob_full = paddle.reshape(prob_full, [-1, last_dim])
        prob = prob_full[:, self._class_id]  # 取第 class_id 列
        # 根据 mask 选取有效元素
        prob = paddle.masked_select(prob, mask)

        # 同理，取对应的 target
        t = paddle.masked_select(target[:, self._class_id], mask)

        # 计算 BCELoss (reduction='sum')
        # 注意：BCELoss 要求输入是概率，标签是0或1
        loss = self.bce_loss(prob, t)

        # 累加结果
        self._total_ce += float(loss.numpy()[0])  # .item() 在 Paddle 中通常是 .numpy()[0]
        self._total_samples += prob.shape[0]

    def accumulate(self):
        """
        计算当前 metric 的结果，与 TorchMetrics 中的 compute 类似。
        """
        if self._total_samples == 0:
            return 0.0
        return self._total_ce / self._total_samples

    def reset(self):
        """重置内部状态。"""
        self._total_ce = 0.0
        self._total_samples = 0

    def clear(self):
        """有些版本 Paddle Metric 会用 clear()，这里跟 reset() 同义。"""
        self.reset()


# ================================
# 3. 定义子类：各元素/键类型的 CE
# ================================


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


# Bond 类型
class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


# ================================
# 4. 原先的 AtomMetricsCE 与 BondMetricsCE
#    在 Paddle 中使用自定义的 PaddleMetricCollection
# ================================


class AtomMetricsCE(PaddleMetricCollection):
    """
    用于管理与原子相关的多个 CEPerClass 子类。
    """

    def __init__(self, dataset_infos):
        """
        dataset_infos: 假设里边包含 'atom_decoder'，
                       类似于原先 PyTorch 里 AtomDecoder 用来对应 i->原子类型。
        """
        atom_decoder = dataset_infos.atom_decoder  # 假设是个列表，比如 ['H','C','N',...]
        class_dict = {
            "H": HydrogenCE,
            "C": CarbonCE,
            "N": NitroCE,
            "O": OxyCE,
            "F": FluorCE,
            "B": BoronCE,
            "Br": BrCE,
            "Cl": ClCE,
            "I": IodineCE,
            "P": PhosphorusCE,
            "S": SulfurCE,
            "Se": SeCE,
            "Si": SiCE,
        }

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            ce_class = class_dict[atom_type]
            metrics_list.append(ce_class(i))

        super().__init__(metrics_list)


class BondMetricsCE(PaddleMetricCollection):
    """
    用于管理与键类型相关的多个 CEPerClass 子类。
    """

    def __init__(self):
        # 假设顺序：NoBond=0, Single=1, Double=2, Triple=3, Aromatic=4
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)

        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


# ================================
# 5. 训练/验证流程中的模块
#    对应原先的 TrainMolecularMetricsDiscrete(nn.Module)
#    在 Paddle 中使用 paddle.nn.Layer
# ================================


class TrainMolecularMetricsDiscrete(nn.Layer):
    """
    用于在训练/验证循环中调用 update / log / reset 等方法。
    """

    def __init__(self, dataset_infos):
        super().__init__()
        # 替代 PyTorch 里的 AtomMetricsCE / BondMetricsCE
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E):
        """
        与 PyTorch 里的 forward 类似。
        - masked_pred_X, masked_pred_E: 模型输出
        - true_X, true_E: 真实标签
        """
        # 调用 update
        self.train_atom_metrics.update(masked_pred_X, true_X)
        self.train_bond_metrics.update(masked_pred_E, true_E)

        if log_flag:
            to_log = {}
            # compute() = accumulate()，得到每个指标的当前值
            atom_results = self.train_atom_metrics.compute()
            bond_results = self.train_bond_metrics.compute()

            for key, val in atom_results.items():
                to_log[f"train/{key}"] = val
            for key, val in bond_results.items():
                to_log[f"train/{key}"] = val

    def reset(self):
        """
        每个 epoch 或特定阶段结束后重置，便于下一轮统计。
        """
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()

    def log_epoch_metrics(self):
        """
        计算并 log 本轮 epoch 累计下来的指标，可在 epoch 结束时调用。
        """
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log[f"train_epoch/{key}"] = val
        for key, val in epoch_bond_metrics.items():
            to_log[f"train_epoch/{key}"] = val

        # 也可返回这些值给外层处理或打印
        return epoch_atom_metrics, epoch_bond_metrics


# ================
# 使用示例（伪代码）
# ================
if __name__ == "__main__":
    # 模拟 dataset_infos
    class DatasetInfosMock:
        def __init__(self):
            self.atom_decoder = ["H", "C", "N", "O", "F"]  # 仅作示例

    dataset_infos = DatasetInfosMock()

    # 初始化
    metrics_layer = TrainMolecularMetricsDiscrete(dataset_infos)

    # 假设训练循环里
    for epoch in range(3):
        metrics_layer.reset()

        for step in range(5):
            # 模拟网络输出 preds / true
            batch_pred_X = paddle.rand(
                [2, 10, len(dataset_infos.atom_decoder)]
            )  # (bs=2, n=10, d=5)
            batch_true_X = paddle.rand(
                [2, 10, len(dataset_infos.atom_decoder)]
            )  # 同样 shape

            batch_pred_E = paddle.rand([2, 10, 5])  # 例如 (bs=2, n=10, d=5)
            batch_true_E = paddle.rand([2, 10, 5])  # 同样 shape

            # 调用 forward 更新指标
            metrics_layer(
                batch_pred_X,
                batch_pred_E,
                batch_true_X,
                batch_true_E,
                log_flag=(step % 2 == 0),
            )

        # epoch 结束后，打印或 log 一次 epoch 级别指标
        epoch_atom_metrics, epoch_bond_metrics = metrics_layer.log_epoch_metrics()
        print(f"Epoch {epoch} Atom Metrics:", epoch_atom_metrics)
        print(f"Epoch {epoch} Bond Metrics:", epoch_bond_metrics)
