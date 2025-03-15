import paddle
import paddle.nn as nn

# ================================
# 1. Custom MetricCollection Class
# ================================


class PaddleMetricCollection:
    """
    Used for collecting multiple custom metrics and calling
    their update, compute, and reset methods uniformly.
    """

    def __init__(self, metrics_list):
        """
        metrics_list: Used to store multiple custom metrics and
        each metric can be an instance of a class derived from Metric.
        providing unified calls to update, compute, and reset.
        """
        self.metrics_list = metrics_list

    def update(self, preds, target):
        """call update methods of all metric"""
        for metric in self.metrics_list:
            metric.update(preds, target)

    def compute(self):
        """
        returns a dictionary of the form {metric_name: metric_value}
        """
        results = {}
        for metric in self.metrics_list:
            # Use metric.name to distinguish between different metrics
            results[metric.name()] = metric.accumulate()
        return results

    def reset(self):
        """reset internal state of metrics"""
        for metric in self.metrics_list:
            metric.reset()


# ================================
# 2. Custom CEPerClass (Paddle Metric)
# ================================


class CEPerClass(paddle.metric.Metric):
    """
    A custom metric in PaddlePaddle that computes the binary cross-entropy (BCE) loss
    for a specific class
    """

    def __init__(self, class_id):
        super().__init__()
        self._class_id = class_id
        # using Python variable, alternatively paddle.to_tensor(0.0) for accumulation.
        self._total_ce = 0.0
        self._total_samples = 0

        # softmax: Used to convert multi-class outputs into class-specific probabilities
        self.softmax = nn.Softmax(axis=-1)
        # paddle.nn.BCELoss() or paddle.nn.BCEWithLogitsLoss() for layer-based usage,
        # or The functional API: paddle.nn.functional.binary_cross_entropy()
        self.bce_loss = nn.BCELoss(reduction="sum")

    def name(self):
        """
        Binary cross-entropy (BCE) loss in Paddle can be computed using:
        paddle.nn.BCELoss(), paddle.nn.functional.binary_cross_entropy()
        or paddle.nn.functional.binary_cross_entropy_with_logits()
        """
        # For example, return "ce_class_{id}";
        # when used within a metric collection, naming can also incorporate
        # the outer class name.
        cls_name = str(self.__class__).split(".")[-1].split("'")[0]
        metric_name = "".join(filter(str.isalpha, cls_name[:-2]))
        return f"CE_class_{metric_name}"

    def update(self, preds, target):
        """
        batch update internal states。
        preds.shape: (bs, n, d) or (bs, n, n, d)
        target.shape: same as preds: (bs, n, d) or (bs, n, n, d)
        """
        # flatten (bs, n, ...) inot (X, d)
        last_dim = target.shape[-1]  # d
        target = paddle.reshape(target, [-1, last_dim])

        # create mask to exclude rows that are entirely zero:(target != 0).any(axis=-1)
        mask = paddle.any(target != 0.0, axis=-1)  # bool张量

        # Retrieve the probability of a specific class.
        prob_full = self.softmax(preds)  # preds => softmax => [batch, ..., d]
        # reshape preds => [X, d]
        prob_full = paddle.reshape(prob_full, [-1, last_dim])
        prob = prob_full[:, self._class_id]  # Select the column of `class_id`.
        # Select valid elements according to the mask
        prob = paddle.masked_select(prob, mask)

        # obtain the corresponding target
        t = paddle.masked_select(target[:, self._class_id], mask)

        # Calculate BCELoss (reduction='sum')
        # Note: BCELoss requires the input to be probabilities and labels to be 0 or 1
        loss = self.bce_loss(prob, t)

        # Accumulate results
        self._total_ce += float(loss.numpy())  # .item() 在 Paddle 中通常是 .numpy()[0]
        self._total_samples += prob.shape[0]

    def accumulate(self):
        if self._total_samples == 0:
            return 0.0
        return self._total_ce / self._total_samples

    def reset(self):
        self._total_ce = 0.0
        self._total_samples = 0

    def clear(self):
        self.reset()


# ================================
# 3. Define subclasses: CE for each element/key type
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


# Bond type
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
# 4. The original AtomMetricsCE and BondMetricsCE
# Use custom PaddleMetricCollection in Paddle
# ================================


class AtomMetricsCE(PaddleMetricCollection):
    """
    Used to manage multiple CEPerClass subclasses related to atoms.
    """

    def __init__(self, dataset_infos):
        """
        dataset_infos: assuming it contains 'atom_decoder',
        """
        atom_decoder = dataset_infos.atom_decoder  # ['H','C','N',...]
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
    Used to manage multiple CEPerClass subclasses related to bond types.
    """

    def __init__(self):
        # assume order：NoBond=0, Single=1, Double=2, Triple=3, Aromatic=4
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)

        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


# ================================
# 5. Modules in the training/validation process
# ================================


class TrainMolecularMetricsDiscrete(nn.Layer):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool = False):
        """
        - masked_pred_X, masked_pred_E: model output
        - true_X, true_E: true labels
        """
        self.train_atom_metrics.update(masked_pred_X, true_X)
        self.train_bond_metrics.update(masked_pred_E, true_E)

        if log:
            to_log = {}
            atom_results = self.train_atom_metrics.compute()
            bond_results = self.train_bond_metrics.compute()

            for key, val in atom_results.items():
                to_log[f"train/{key}"] = val
            for key, val in bond_results.items():
                to_log[f"train/{key}"] = val
        return to_log

    def reset(self):
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log[f"train_epoch/{key}"] = val
        for key, val in epoch_bond_metrics.items():
            to_log[f"train_epoch/{key}"] = val

        return epoch_atom_metrics, epoch_bond_metrics


# ================
# Usage example (pseudocode)
# ================
if __name__ == "__main__":
    # simulate dataset_infos
    class DatasetInfosMock:
        def __init__(self):
            self.atom_decoder = ["H", "C", "N", "O", "F"]  # 仅作示例

    dataset_infos = DatasetInfosMock()

    # init
    metrics_layer = TrainMolecularMetricsDiscrete(dataset_infos)

    # Assuming within the training loop
    for epoch in range(3):
        metrics_layer.reset()

        for step in range(5):
            # Simulate network output preds / true
            batch_pred_X = paddle.rand(
                [2, 10, len(dataset_infos.atom_decoder)]
            )  # (bs=2, n=10, d=5)
            batch_true_X = paddle.rand(
                [2, 10, len(dataset_infos.atom_decoder)]
            )  # same shape

            batch_pred_E = paddle.rand([2, 10, 5])  # such (bs=2, n=10, d=5)
            batch_true_E = paddle.rand([2, 10, 5])  # same shape

            # Call forward to update metrics
            metrics_layer(
                batch_pred_X,
                batch_pred_E,
                batch_true_X,
                batch_true_E,
                log=(step % 2 == 0),
            )

        # After each epoch, print or log epoch-level metrics once
        epoch_atom_metrics, epoch_bond_metrics = metrics_layer.log_epoch_metrics()
        print(f"Epoch {epoch} Atom Metrics:", epoch_atom_metrics)
        print(f"Epoch {epoch} Bond Metrics:", epoch_bond_metrics)
