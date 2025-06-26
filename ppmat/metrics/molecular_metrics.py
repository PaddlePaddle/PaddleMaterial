import os
from typing import List, Union, Any, Dict, Optional
from pathlib import Path
import pandas as pd
import copy

import paddle
import paddle.nn.functional as F

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDKFingerprint

from ppmat.datasets.ext_rdkit import compute_molecular_metrics
from ppmat.models.denmr.utils import model_utils as m_utils
from ppmat.utils import logger
from ppmat.models.denmr.utils import diffgraphformer_utils as utils


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
    """Evaluate a batch (or full epoch) of sampled molecules.
    
    This helper layer centralises **all** post‑generation evaluation logic, namely
    1.  **Structural one‑to‑one accuracy**  — does the generated graph match the
        ground‑truth graph *exactly* (SMILES level)
    2.  **RDKit quality metrics**           — Valid / Unique / Novel / ConnComp &
        common physicochemical statistics.
    3.  **Histogram MAE** on *n‑nodes / atom‑types / bond‑types / valency*.
    4.  **Retrieval metrics** (optional)    — embed molecules with *molVec*, embed
        NMR conditions with *nmrVec*, compute cosine‑similarity, then measure
        **top‑1 / top‑5 / top‑10** hit rate and output a CSV of fingerprint
        similarities.
    
    Parameters
    ----------
    dataset_infos : Any
        Object that contains meta information of the training set. Must expose
        the following attributes used inside the layer::

            max_n_nodes          – Maximum #nodes in any molecule.
            n_nodes              – 1‑D array, histogram of node counts.
            node_types           – 1‑D array, histogram of atom types.
            edge_types           – 1‑D array, histogram of bond types.
            valency_distribution – 1‑D array, histogram of valencies.
            atom_decoder         – List[str], maps atom type ID → element symbol.
    train_smiles : List[str]
        All canonical SMILES strings in the training set. Used for *Novelty*
        metric.
    nmr_encoder / mol_encoder : paddle.nn.Layer
        Pre‑initialised encoders that embed (i) the four‑branch NMR condition
        vector and (ii) the one‑hot molecular graph into a shared latent space.
        If you do **not** need retrieval metrics, you may safely pass
        ``None`` for both arguments.
    num_candidate : int, default=20
        How many candidate molecules were generated **per ground‑truth graph**
        upstream. Needed only for retrieval metrics.
    """

    def __init__(
        self, 
        dataset_infos: Any, 
        train_smiles: List[str],
        clip: Optional[paddle.nn.Layer] = None,
        num_candidate: int = 1,
    ):
        super().__init__()
        # save external handles
        self.di = di = dataset_infos
        if clip:
            self.clip = clip
            self.nmrVec = clip.text_encoder
            self.molVec = clip.graph_encoder
        self.num_candidate = num_candidate
        self.atom_decoder = di.atom_decoder # Alias for convenience
        
        # 1. Reference histograms  (registered as *buffers*)
        self.register_buffer(
            name="n_target_dist", 
            tensor=self._normalise(di.n_nodes)
        )
        self.register_buffer(
            name="node_target_dist", 
            tensor=self._normalise(di.node_types)
        )
        self.register_buffer(
            "edge_target_dist",self._normalise(di.edge_types)
        )
        self.register_buffer(
            "valency_target_dist",
            self._normalise(di.valency_distribution)
        )
        
        # 2. Online histogram accumulators for generated molecules
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        self.generated_node_dist = GeneratedNodesDistribution(di.output_dims["X"])
        self.generated_edge_dist = GeneratedEdgesDistribution(di.output_dims["E"])
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)

        # 3. Metric objects (MAE = mean absolute error between two PDFs)
        self.n_dist_mae = HistogramsMAE(self.n_target_dist)
        self.node_dist_mae = HistogramsMAE(self.node_target_dist)
        self.edge_dist_mae = HistogramsMAE(self.edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(self.valency_target_dist)

        # Training set SMILES: used only for Novelty metric
        self.train_smiles = train_smiles

    def forward(
        self,
        samples: Dict[str, Any],
        current_epoch: int,
        local_rank: int,
        output_dir: str,
        flag_test=False,
        log_each_molecule: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate **all** evaluation metrics into a single dict.

        Expected structure of ``samples``
        ----------------------------------
        Mandatory keys
        ^^^^^^^^^^^^^^
        ``pred``  : ``List[B]`` – generated molecules, each item is
                     ``[atom_idx (n,), edge_idx (n,n)]``.
        ``true``  : ``List[B]`` – ground‑truth molecules, identical format.
        ``dict``  : ``List[str]`` – atom decoder.
        ``n_all`` : ``int``      – total #molecules evaluated.

        Optional keys (required only if retrieval metrics are desired)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ``candidates``      : ``List[num_candidate][B]`` – list of candidate
                              molecule lists.
        ``candidates_X/E``  : parallel list of one‑hot tensors fed to molVec.
        ``batch_condition`` : list(4) of Paddle tensors (NMR encoding).
        ``node_mask_meta``  : Paddle int tensor of shape ``[B]`` (n_nodes per
                              molecule).
        """
        # contariners
        to_log: Dict[str, Any] = {}
        
        # 1. exact match accuracy
        mol_pred = samples["pred"]
        mol_true = samples["true"]
        total_num = samples["n_all"]
        hit_exact = 0
        for p, t in zip(mol_pred, mol_true):
            m_gen  = m_utils.mol_from_graphs(self.atom_decoder, *p)
            m_true = m_utils.mol_from_graphs(self.atom_decoder, *t)
            if Chem.MolToSmiles(m_gen, True) == Chem.MolToSmiles(m_true, True):
                hit_exact += 1
        to_log.update({
            "Accuracy":       hit_exact / total_num,
            "Right Number":   hit_exact,
            "Total Number":   total_num,
        })

        # 2. RDKit global metris
        stability, rdkit_metrics, all_smiles = compute_molecular_metrics(
            mol_pred, self.train_smiles, self.di)
        if local_rank == 0:
            to_log.update(stability)  # Valid/Unique/Novel/ConnComp inside
            val, uniq, nov, conn = rdkit_metrics[0]
            to_log.update({
                "Validity": val, "Uniqueness": uniq,
                "Novelty": nov, "Connected Components": conn,
            })
            for k, v in rdkit_metrics[2].items(): #TODO need it?
                to_log[k] = v

        # 3. histogram MAE
        for dist_obj, mae_obj, tag in [
            (self.generated_n_dist,       self.n_dist_mae,      "n"),
            (self.generated_node_dist,    self.node_dist_mae,   "node"),
            (self.generated_edge_dist,    self.edge_dist_mae,   "edge"),
            (self.generated_valency_dist, self.valency_dist_mae,"valency"),
        ]:
            dist_obj(mol_pred)           # accumulate new batch
            g_dist = dist_obj.accumulate()
            mae_obj(g_dist)
            if local_rank == 0:
                to_log[f"Gen {tag} distribution"]   = g_dist
                to_log[f"basic_metrics/{tag}_mae"] = mae_obj.accumulate()

        # 4. compute metrics for atom and bond types distrubution
        for i, atom_type in enumerate(self.atom_decoder):
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                self.generated_node_dist[i] - self.node_target_dist[i]
            ).item()
        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
        ):
            to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                self.generated_edge_dist[j] - self.edge_target_dist[j]
            ).item()
        for valency in range(6):
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                self.generated_valency_dist[valency] - self.valency_target_dist[valency]
            ).item()

        # 5. retrieval metrics
        if "candidates" in samples:
            to_log.update(
                self._retrieval_metrics(
                    samples, 
                    output_dir, 
                    current_epoch, 
                    local_rank, 
                    verbose=log_each_molecule
                )
            )

        # 6. dump all SMILES
        if flag_test and local_rank == 0:
            file = Path(output_dir) / "graphs" / f"final_smiles_e_{current_epoch}.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("\n".join(s if s is not None else "" for s in all_smiles))

        return to_log

    # ====================================================================
    # Internal helpers
    # ====================================================================
    @staticmethod
    def _normalise(arr: Any) -> paddle.tensor:
        """Convert *array‑like* to FP32 Tensor and L1‑normalise it."""
        arr_fp32 = paddle.to_tensor(arr, dtype="float32")
        return arr_fp32 / paddle.sum(arr_fp32)
    # --------------------------------------------------------------------
    
    def _retrieval_metrics(
        self,
        samples: Dict[str, Any],
        output_dir: str,
        epoch: int,
        local_rank: int,
        *,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Compute molVec‑nmrVec retrieval scores & dump similarity CSV."""
        # unpack
        cand_lists  = samples["candidates"]
        cand_X      = samples["candidates_X"]
        cand_E      = samples["candidates_E"]
        cond_y      = samples["batch_condition"]
        atom_counts = samples["node_mask_meta"]
        true_list   = samples["true"]
        B, C        = len(true_list), len(cand_lists)
        if isinstance(cand_X, list):
            cand_X = [paddle.to_tensor(x) for x in cand_X]
            cand_X = paddle.stack(cand_X, axis=0)    # -> [C, B, n_max, d_x]
            cand_E = [paddle.to_tensor(x) for x in cand_E]
            cand_E = paddle.stack(cand_E, axis=0)    # -> [C, B, n_max, n_max, d_e]

        # 1) Build node mask
        n_max = int(paddle.max(paddle.stack(atom_counts)).item())
        arange = paddle.arange(n_max, dtype="int64")
        node_mask = arange.unsqueeze(0).expand([B, n_max]) < paddle.stack(atom_counts).unsqueeze(1)

        # 2) Embeddings 
        with paddle.no_grad():
            # A. compute NMR embedding vector once 
            nmr_emb = self.nmrVec(cond_y)                     # [B, d]
            
            # B. merge candidate dimension C into batch dimension
            X_flat = cand_X.reshape([C * B, *cand_X.shape[2:]])   # [C·B, n_max, d_x]
            E_flat = cand_E.reshape([C * B, *cand_E.shape[2:]])   # [C·B, n_max, n_max, d_e]
            node_mask_flat = node_mask.tile([C, 1])               # [C·B, n_max] – repeat mask per candidate
            
            # C zerolength y-placeholder requiered by graph encoder
            y_flat = paddle.zeros([C * B, 0], dtype=X_flat.dtype) # [C·B, 0]

            # D. remove padding (PlaceHolder.mask) -> z_t.{X,E,y}
            z_t = (
                utils.PlaceHolder(X=X_flat, E=E_flat, y=y_flat)
                    .type_as(X_flat)            # match dtype
                    .mask(node_mask_flat)       # drop padded rows/cols
            )
            
            # E. compute extra features for all C*B graphs
            extra = m_utils.compute_extra_data(
                self.clip,
                {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask_flat},
                isPure=True,                                # flag preserved from legacy
            )
            
            # F. prepare the input data by concatenating extra features
            X_in = paddle.concat([z_t.X.astype("float32"), extra.X], axis=2)         # [C·B, n_max, d_x']
            E_in = paddle.concat([z_t.E.astype("float32"), extra.E], axis=3)         # [C·B, n_max, n_max, d_e']
            y_in = paddle.concat([z_t.y.astype("float32"), extra.y], axis=1)         # [C·B, d_y']

            # G. compute graph embeddings vector once
            mol_flat: paddle.Tensor = self.molVec(X_in, E_in, y_in, node_mask_flat)   # [C·B, d_embed]

            # H.  Reshape back  →  [C, B, d_embed]  for later cosine-similarity
            mol_embs = mol_flat.reshape([C, B, -1])                                  # final shape

        # 3) Cosine similarity 
        sims = F.cosine_similarity(
            nmr_emb.unsqueeze(0).expand([C, -1, -1]), mol_embs, axis=-1)  # [C,B]
        max_idx   = paddle.argmax(sims, axis=0)             # top‑1 index per GT
        top5_idx  = paddle.topk(sims, k=5,  axis=0)[1]  # [5,B]
        top10_idx = paddle.topk(sims, k=10, axis=0)[1]  # [10,B]

        hit1 = hit5 = hit10 = 0
        csv_records: List[Dict[str, str]] = []

        # 4) Loop over ground‑truth molecules 
        for i in range(B):
            m_true = m_utils.mol_from_graphs(self.atom_decoder, *true_list[i])
            s_true = Chem.MolToSmiles(m_true, True)

            # ---- top‑1 ----
            sel = int(max_idx[i])
            m_pred = m_utils.mol_from_graphs(self.atom_decoder, *cand_lists[sel][i])
            s_pred = Chem.MolToSmiles(m_pred, True)
            if s_pred == s_true:
                hit1 += 1

            # ---- top‑5 / top‑10 ----
            if C >= 5:
                for sel in top5_idx[:, i].astype("int64").tolist():
                    if Chem.MolToSmiles(m_utils.mol_from_graphs(self.atom_decoder, *cand_lists[sel][i]), True) == s_true:
                        hit5 += 1; break
            if C >= 10:
                for sel in top10_idx[:, i].astype("int64").tolist():
                    if Chem.MolToSmiles(m_utils.mol_from_graphs(self.atom_decoder, *cand_lists[sel][i]), True) == s_true:
                        hit10 += 1; break

            # ---- fingerprint similarity (top‑1 vs GT) ----
            try:
                sim = DataStructs.FingerprintSimilarity(RDKFingerprint(m_pred), RDKFingerprint(m_true))
            except Exception:
                sim = 0.0
            csv_records.append({"SMILES": s_true, "Similarity": f"{sim:.4f}"})
            if verbose:
                print(f"[GT {i+1}/{B}] top1={'OK' if s_pred==s_true else 'NO'} sim={sim:.3f}")

        # 5) Dump CSV
        if local_rank == 0:
            csv_path = Path(output_dir) / f"similarity_results_e{epoch}.csv"
            pd.DataFrame(csv_records).to_csv(csv_path, index=False)

        # 6) Prepare metrics
        ks      = (1, 5, 10)          # k-values we care about
        hits    = (hit1, hit5, hit10) # corresponding correct-hit counters

        metrics = {f"retrieval_top{k}": h / B
                for k, h in zip(ks, hits)
                if C >= k}          # keep entry only when C ≥ k

        return metrics

    def reset(self):
        """Clear all running statistics (called at epoch boundaries)."""
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
    
    def __getitem__(self, idx):
        return self.n_dist[idx]


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
    
    def __getitem__(self, idx):
        return self.node_dist[idx]


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
    
    def __getitem__(self, idx):
        return self.edge_dist[idx]


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
    
    def __getitem__(self, idx):
        return self.edgepernode_dist[idx]


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
