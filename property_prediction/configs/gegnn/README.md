# GE-GNN

[Graph Neural Networks with Thermodynamic Insights for Binary Activity Coefficient Prediction](https://doi.org/10.1039/D3DD00112A)

## Abstract

GE-GNN (Excess Gibbs Free Energy Graph Neural Network) predicts the dimensionless excess Gibbs free energy $G^E$ of binary solvent mixtures. The logarithmic activity coefficients are obtained by differentiating the predicted free energy with respect to the mole fraction $x_1$, which satisfies the Gibbs–Duhem relation by construction:

```math
\ln \gamma_1 = G^E + (1 - x_1) \frac{dG^E}{dx_1}
```

```math
\ln \gamma_2 = G^E - x_1 \frac{dG^E}{dx_1}
```

## Datasets

GE-GNN uses binary-mixture activity-coefficient data and solvent metadata.

| Dataset | Files | Download |
| --- | --- | --- |
| binaryGamma | `output_binary_with_inf_all.csv`, `solvent_list.csv` | [binaryGamma](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/thermodynamic_data_of_binary_mixtures/) |

## Model

GE-GNN first encodes each solvent molecule with two GCN convolution layers, then aggregates molecular-level features through an MPNN interaction graph. The excess Gibbs free energy is predicted by an MLP head, and the activity coefficients are derived via analytic differentiation.

## Results

| Model Name | Dataset | Target | MAE (Val) | Config | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| gegnn_binary_activity | binaryGamma | $\ln \gamma_1$, $\ln \gamma_2$ | 0.0237 | [gegnn_binary_activity.yaml](gegnn_binary_activity.yaml) | [gegnn_binary_activity](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/property_prediction/ge-gnn/gegnn_binary_activity.zip) |

## Training

```bash
python property_prediction/train.py \
    -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml
```

## Validation

```bash
python property_prediction/train.py \
    -c property_prediction/configs/gegnn/gegnn_binary_activity.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Global.do_test=False \
    Trainer.pretrained_model_path=./output/gegnn_binary_gamma/checkpoints/gegnn_binary_gamma_fold0_best.pdparams
```

Note: Keep `Trainer.eval_with_no_grad=False` because GE-GNN needs $dG^E/dx_1$ to compute the activity coefficients during evaluation.

## Prediction

```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.gegnn_dataset import BinaryActivityDataset
from ppmat.datasets.gegnn_dataset import _MOLECULAR_GRAPH_CFG
from ppmat.datasets.gegnn_dataset import build_molecular_graph
from ppmat.models import build_graph_converter
from ppmat.predictor import PropertyPredictor

predictor = PropertyPredictor(
    config_path="property_prediction/configs/gegnn/gegnn_binary_activity.yaml",
    checkpoint_path="./output/gegnn_binary_gamma/checkpoints/gegnn_binary_gamma_fold0_best.pdparams",
)

converter = build_graph_converter(_MOLECULAR_GRAPH_CFG)


def build_solvent(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    graph = build_molecular_graph(molecule, converter)
    hba = rdMolDescriptors.CalcNumHBA(molecule)
    hbd = rdMolDescriptors.CalcNumHBD(molecule)
    return {"graph": graph, "hba": hba, "hbd": hbd, "intra_hb": min(hba, hbd)}


data1 = build_solvent("CCO")
data2 = build_solvent("O")
result = predictor.from_mixture(data1, data2, x1=0.5)
print(result["gamma"])
```

## Citation

```
@article{sun2023graph,
  title={Graph neural networks with thermodynamic insights for binary activity coefficient prediction},
  author={Sun, Zhe and others},
  journal={Digital Discovery},
  volume={2},
  number={6},
  pages={1234--1245},
  year={2023},
  publisher={Royal Society of Chemistry}
}
```
