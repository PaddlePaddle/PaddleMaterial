from typing import Any, List
from typing import Union

import paddle

from ppmat.datasets.ext_rdkit import compute_molecular_metrics
from ppmat.utils import logger

import os

class Metric(paddle.metric.Metric):
    def __init__(self):
        super().__init__()
        self._initial_states = {}
        self._accumulate_states = {}
        self.name = self.__class__.__name__
        self.reset()

    def add_state(
        self,
        name: str,
        default: Union[List, paddle.Tensor],
        dist_reduce_fx: str = None,
    ) -> None:
        self._name = name
        self._reduce = dist_reduce_fx
        self._initial_states[name] = default
        setattr(self, name, default)

    def name(self):
        return self._name

    def register_state(self, key, value):
        self._initial_states[key] = value
        setattr(self, key, value)

    def reset(self):
        for key, value in self._initial_states.items():
            setattr(self, key, value)

    def accumulate(self):
        for key, value in self._initial_states.items():
            self._accumulate_states[key] = paddle.sum(self._initial_states[key])


###### custom base metrics ######
# Mean Absolute Error (MAE)
class MeanAbsoluteError(paddle.metric.Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.sum_abs_error = 0.0
        self.total_samples = 0

    def __call__(self, preds, target):
        self.update(preds, target)
        return self.accumulate()
    
    def update(self, preds, target):
        abs_error = paddle.abs(preds - target).sum().item()
        self.sum_abs_error += abs_error
        self.total_samples += target.shape[0]

    def accumulate(self):
        return (
            self.sum_abs_error / self.total_samples if self.total_samples > 0 else 0.0
        )

    def name(self):
        return "mean_absolute_error"


# Mean Squared Error (MSE)
class MeanSquaredError(paddle.metric.Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.sum_squared_error = 0.0
        self.total_samples = 0

    def update(self, preds, target):
        squared_error = paddle.square(preds - target).sum().item()
        self.sum_squared_error += squared_error
        self.total_samples += target.shape[0]
        
    def __call__(self, preds, target):
        self.update(preds, target)
        return self.accumulate()

    def accumulate(self):
        return (
            self.sum_squared_error / self.total_samples
            if self.total_samples > 0
            else 0.0
        )

    def name(self):
        return "mean_squared_error"


# Metric Collection
class MetricCollection:
    def __init__(self, metrics):
        self.metrics = metrics

    def update(self, preds, target):
        for metric in self.metrics.values():
            metric.update(preds, target)

    def accumulate(self):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.accumulate()
        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


##############################################################

class SamplingMolecularMetrics(paddle.nn.Layer):
    def __init__(self, dataset_infos, train_smiles):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims["X"])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims["E"])
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        n_target_dist = di.n_nodes.astype(self.generated_n_dist.n_dist.dtype)
        n_target_dist = n_target_dist / paddle.sum(x=n_target_dist)
        self.register_buffer(name="n_target_dist", tensor=n_target_dist)

        node_target_dist = di.node_types.astype(
            self.generated_node_dist.node_dist.dtype
        )
        node_target_dist = node_target_dist / paddle.sum(x=node_target_dist)
        self.register_buffer(name="node_target_dist", tensor=node_target_dist)

        edge_target_dist = di.edge_types.astype(
            self.generated_edge_dist.edge_dist.dtype
        )
        edge_target_dist = edge_target_dist / paddle.sum(x=edge_target_dist)
        self.register_buffer(name="edge_target_dist", tensor=edge_target_dist)

        valency_target_dist = di.valency_distribution.astype(
            self.generated_valency_dist.edgepernode_dist.dtype
        )
        valency_target_dist = valency_target_dist / paddle.sum(x=valency_target_dist)
        self.register_buffer(name="valency_target_dist", tensor=valency_target_dist)

        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = train_smiles
        self.dataset_info = di

    def forward(
        self, molecules: list, current_epoch, val_counter, local_rank, output_dir, test=False
    ):
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            molecules, self.train_smiles, self.dataset_info
        )
        if test and local_rank == 0:
            with open("final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    fp.write("%s\n" % smiles)
                logger.message("All smiles saved")
        logger.message("Starting custom metrics")
        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.accumulate()
        self.n_dist_mae(generated_n_dist)
        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.accumulate()
        self.node_dist_mae(generated_node_dist)
        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.accumulate()
        self.edge_dist_mae(generated_edge_dist)
        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.accumulate()
        self.valency_dist_mae(generated_valency_dist)
        to_log = {}
        for i, atom_type in enumerate(self.dataset_info.atom_decoder):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                generated_probability - target_probability
            ).item()
        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
        ):
            generated_probability = generated_edge_dist[j]
            target_probability = self.edge_target_dist[j]
            to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                generated_probability - target_probability
            ).item()
        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                generated_probability - target_probability
            ).item()
        n_mae = self.n_dist_mae.accumulate()
        node_mae = self.node_dist_mae.accumulate()
        edge_mae = self.edge_dist_mae.accumulate()
        valency_mae = self.valency_dist_mae.accumulate()
        if local_rank == 0:
            logger.message("Custom metrics computed.")

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=paddle.zeros(shape=max_n + 1, dtype="float32"),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = tuple(atom_types.shape)[0]
            self.n_dist[n] += 1
            
    def __call__(self, molecules):
        self.update(molecules)
        return self.n_dist / paddle.sum(x=self.n_dist)
                

    def accumulate(self):
        return self.n_dist / paddle.sum(x=self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=paddle.zeros(shape=num_atom_types, dtype="float32"),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            for atom_type in atom_types:
                error_message = (
                    "Mask error, the molecules should already "
                    "be masked at the right shape"
                )
                assert int(atom_type) != -1, error_message
                self.node_dist[int(atom_type)] += 1

    def accumulate(self):
        return self.node_dist / paddle.sum(x=self.node_dist)
    
    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=paddle.zeros(shape=num_edge_types, dtype="float32"),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = paddle.ones_like(x=edge_types)
            mask = paddle.triu(x=mask, diagonal=1).astype(dtype="bool")
            edge_types = edge_types[mask]
            unique_edge_types, counts = paddle.unique(x=edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def accumulate(self):
        return self.edge_dist / paddle.sum(x=self.edge_dist)
    
    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "total_edge", default=paddle.to_tensor(data=0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_samples", default=paddle.to_tensor(data=0.0), dist_reduce_fx="sum"
        )

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = paddle.triu(x=edge_types, diagonal=1)
            bonds = paddle.nonzero(x=triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def accumulate(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=paddle.zeros(shape=3 * max_n - 2, dtype="float32"),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = paddle.sum(x=edge_types, axis=0)
            unique, counts = paddle.unique(x=valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def accumulate(self):
        return self.edgepernode_dist / paddle.sum(x=self.edgepernode_dist)
    
    def __call__(self, molecules):
        self.update(molecules)
        return self.accumulate()


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 0.001
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.astype(dtype=pred.dtype)
        super().update(pred, self.target_histogram)
    
    def __call__(self, pred):
        self.update(pred)
        return self.accumulate()


class MSEPerClass(MeanSquaredError):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds[..., self.class_id]
        target = target[..., self.class_id]
        super().update(preds, target)


class HydroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetrics(MetricCollection):
    def __init__(self, dataset_infos):
        # remove_h = dataset_infos.remove_h
        self.atom_decoder = dataset_infos.atom_decoder
        # num_atom_types = len(self.atom_decoder)
        # types = {
        #     "H": 0,
        #     "C": 1,
        #     "N": 2,
        #     "O": 3,
        #     "F": 4,
        #     "B": 5,
        #     "Br": 6,
        #     "Cl": 7,
        #     "I": 8,
        #     "P": 9,
        #     "S": 10,
        #     "Se": 11,
        #     "Si": 12,
        # }
        class_dict = {
            "H": HydroMSE,
            "C": CarbonMSE,
            "N": NitroMSE,
            "O": OxyMSE,
            "F": FluorMSE,
            "B": BoronMSE,
            "Br": BrMSE,
            "Cl": ClMSE,
            "I": IodineMSE,
            "P": PhosphorusMSE,
            "S": SulfurMSE,
            "Se": SeMSE,
            "Si": SiMSE,
        }
        metrics_list = []
        for i, atom_type in enumerate(self.atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetrics(MetricCollection):
    def __init__(self):
        mse_no_bond = NoBondMSE(0)
        mse_SI = SingleMSE(1)
        mse_DO = DoubleMSE(2)
        mse_TR = TripleMSE(3)
        mse_AR = AromaticMSE(4)
        super().__init__([mse_no_bond, mse_SI, mse_DO, mse_TR, mse_AR])


# 示例用法
if __name__ == "__main__":
    mae = MeanAbsoluteError()
    mse = MeanSquaredError()
    metric_collection = MetricCollection({"mae": mae, "mse": mse})

    preds = paddle.to_tensor([3.0, 2.0, 7.0])
    targets = paddle.to_tensor([2.0, 2.0, 6.0])

    metric_collection.update(preds, targets)
    results = metric_collection.accumulate()
    print("Results:", results)
