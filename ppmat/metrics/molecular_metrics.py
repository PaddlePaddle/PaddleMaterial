import os
from typing import List
from typing import Union

import paddle
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint

from ppmat.datasets.ext_rdkit import compute_molecular_metrics
from ppmat.models.denmr.utils import model_utils as m_utils
from ppmat.utils import logger


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
        self,
        samples: list,
        current_epoch,
        val_counter,
        local_rank,
        output_dir,
        test=False,
        flag_sample_metric=True,
    ):
        # init
        to_log = {}
        molecule_list = samples["pred"]
        molecule_list_True = samples["true"]
        atom_decoder = samples["dict"]
        total_num = samples["n_all"]
        right_num = 0

        for i in range(len(molecule_list)):
            mol = m_utils.mol_from_graphs(
                atom_decoder,
                molecule_list[i][0],
                molecule_list[i][1],
            )
            mol_true = m_utils.mol_from_graphs(
                atom_decoder,
                molecule_list_True[i][0],
                molecule_list_True[i][1],
            )

            smiles_gen = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles_true = Chem.MolToSmiles(mol_true, isomericSmiles=True)

            try:
                fp1 = RDKFingerprint(mol)
                fp2 = RDKFingerprint(mol_true)
                # Calculate Tanimoto Similarity
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                # log output result
                msg = f" | Generated SMILES: {smiles_gen}"
                msg += f" | True SMILES: {smiles_true}"
                msg += f" | Tanimoto Similarity: {similarity}"

                if similarity == 1 and smiles_gen != smiles_true:
                    msg += f" | different_index:{i}"
                if smiles_gen == smiles_true:
                    right_num += 1
                    msg += f" | same_index:{i}"
            except Exception as e:
                msg = f" Error processing molecule at index {i}: {e}."
                msg += f" | Generated SMILES {smiles_gen} | True SMILES {smiles_true}"
            msg = f"Sampling Metric Calculating: {i+1}/{len(molecule_list)}" + msg
            if flag_sample_metric is True:
                logger.info(msg)

        to_log["Accuracy"] = right_num / total_num
        to_log["Right Number"] = right_num
        to_log["Total Number"] = total_num

        # compute molecular metrics
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            molecule_list, self.train_smiles, self.dataset_info
        )
        if local_rank == 0:
            valid_unique_molecules = rdkit_metrics[1]
            textfile = open(
                f"{output_dir}/graph/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt",
                "w",
            )
            textfile.writelines(valid_unique_molecules)
            textfile.close()
            for k, v in stability.items():
                to_log[k] = v
            to_log["Validity"] = rdkit_metrics[0][0]
            to_log["Uniqueness"] = rdkit_metrics[0][1]
            to_log["Novelty"] = rdkit_metrics[0][2]
            to_log["Connected Components"] = rdkit_metrics[0][3]
            for k, v in rdkit_metrics[2].items():
                to_log[k] = v

        if test and local_rank == 0:
            file_path = f"{output_dir}/graphs/final_smiles_e_{current_epoch}.txt"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 确保目录存在
            with open(file_path, "w") as fp:
                for smiles in all_smiles:
                    fp.write("%s\n" % smiles)
            msg = "Sampling Metric Calculation:"
            msg += " All smiles saved"
            logger.message(msg)

        # compute generated distributions metrics
        self.generated_n_dist(molecule_list)
        generated_n_dist = self.generated_n_dist.accumulate()
        self.n_dist_mae(generated_n_dist)
        self.generated_node_dist(molecule_list)
        generated_node_dist = self.generated_node_dist.accumulate()
        self.node_dist_mae(generated_node_dist)
        self.generated_edge_dist(molecule_list)
        generated_edge_dist = self.generated_edge_dist.accumulate()
        self.edge_dist_mae(generated_edge_dist)
        self.generated_valency_dist(molecule_list)
        generated_valency_dist = self.generated_valency_dist.accumulate()
        self.valency_dist_mae(generated_valency_dist)
        if local_rank == 0:
            to_log["Gen n distribution"] = generated_n_dist
            to_log["Gen node distribution"] = generated_node_dist
            to_log["Gen edge distribution"] = generated_edge_dist
            to_log["Gen valency distribution"] = generated_valency_dist

        # compute MAE for distributions
        n_mae = self.n_dist_mae.accumulate()
        node_mae = self.node_dist_mae.accumulate()
        edge_mae = self.edge_dist_mae.accumulate()
        valency_mae = self.valency_dist_mae.accumulate()
        if local_rank == 0:
            to_log["basic_metrics/n_mae"] = (n_mae,)
            to_log["basic_metrics/node_mae"] = (node_mae,)
            to_log["basic_metrics/edge_mae"] = (edge_mae,)
            to_log["basic_metrics/valency_mae"] = valency_mae

        # compute metrics for atom and bond types
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

        return to_log

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
            edge_types = paddle.to_tensor(edge_types)
            mask = paddle.ones_like(x=edge_types)
            mask = paddle.triu(x=mask, diagonal=1).astype(dtype="bool")
            edge_types = edge_types[mask]
            unique_edge_types, counts = paddle.unique(x=edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def accumulate(self):
        return self.edge_dist / paddle.sum(x=self.edge_dist)

    def __call__(self, molecules):
        self.reset()
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
            edge_types = paddle.to_tensor(edge_types)
            edge_types[edge_types == 4] = 1.5
            valencies = paddle.sum(x=edge_types, axis=0)
            unique, counts = paddle.unique(x=valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def accumulate(self):
        return self.edgepernode_dist / paddle.sum(x=self.edgepernode_dist)

    def __call__(self, molecules):
        self.reset()
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
