# GDI-NN

[Gibbs-Duhem-Informed Neural Networks for Binary Activity Coefficient Prediction](https://doi.org/10.1039/D3DD00103B)

## Abstract

GDI-NN (Gibbs-Duhem-Informed Neural Networks) predicts binary activity coefficients at varying compositions by explicitly including the Gibbs-Duhem equation in the loss function during training, which is straightforward in standard machine learning (ML) frameworks enabling automatic differentiation. In contrast to hybrid ML approaches that embed a specific thermodynamic model inside the neural network and inherit its prediction limitations, GDI-NN treats Gibbs-Duhem consistency purely as a regularization term, so that the flexibility of ML models is preserved. Experimental results show that GDI-NN improves both thermodynamic consistency and generalization capabilities for activity coefficient predictions on graph neural networks and matrix completion methods. The model architecture, particularly the activation function, is also found to have a strong influence on the prediction quality. The approach can be easily extended to account for other thermodynamic consistency conditions.

![SolvGNN structure](../../docs/GDNN_structure.png)

![GE-GNN structure](../../docs/GE-GNN_structure.png)

## Datasets

The dataset used in this work is the binary activity coefficient dataset created by Qin et al. and adopted in the GDI-NN paper. It consists of 280 000 binary activity coefficients calculated with COSMO-RS at a constant temperature of 298 K, covering 40 000 different binary mixtures over 700 different compounds.

Data files can be obtained from the original [GDI-NN repository](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/tree/6383142feb3b926fd279ae676a211fd8b3f1dac3/data). Only the following two files are required:

| File | Description |
| :--- | :--- |
| [output_binary_with_inf_all.csv](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/raw/6383142feb3b926fd279ae676a211fd8b3f1dac3/data/output_binary_with_inf_all.csv) | Binary activity coefficient data (including the infinite dilution case) |
| [solvent_list.csv](https://git.rwth-aachen.de/avt-svt/public/GDI-NN/-/raw/6383142feb3b926fd279ae676a211fd8b3f1dac3/data/solvent_list.csv) | Solvent list (700 solvents with SMILES) |

The `output_binary_with_inf_all.csv` file uses the GDI-NN format with columns:

```csv
job_id,solv1,solv2,solv1_x,solv2_x,solv1_gamma,solv2_gamma,warnings,solv1_smiles,solv2_smiles,solv1_name,solv2_name,tpsa_binary_avg
0,solvent_587,solvent_604,0.1,0.9,0.47175935,0.00025148,,CN,CC(=O)CC(C)C,METHYL AMINE,METHYL ISOBUTYL KETONE,2
```

Note that `solv1_gamma` and `solv2_gamma` store the natural logarithm `ln(γ)` of the activity coefficients.

Before training, you can split `output_binary_with_inf_all.csv` into `train_binary.csv`, `val_binary.csv`, and `test_binary.csv`, and update the corresponding paths in the configuration file. Place `solvent_list.csv` alongside these files and set `solvent_list_path` accordingly.

## Model

GDI-NN introduces a physics-informed training strategy that augments the standard supervised loss on activity coefficients with a regularization term derived from the Gibbs-Duhem differential equation. For a binary mixture at constant temperature and pressure, the Gibbs-Duhem relation

$$x_1 \left(\frac{\partial \ln \gamma_1}{\partial x_1}\right)_{T,p} + x_2 \left(\frac{\partial \ln \gamma_2}{\partial x_1}\right)_{T,p} = 0$$

is enforced during training by evaluating the partial derivatives of the predicted `ln(γᵢ)` with respect to the input composition `xᵢ` through automatic differentiation, and adding the squared deviation from zero to the loss. The weighting factor `λ` balances the prediction loss and the Gibbs-Duhem loss. A data augmentation strategy randomly samples additional compositions `x ∈ [0, 1]` for which only the Gibbs-Duhem loss is evaluated, so that thermodynamic consistency is induced at compositions without labeled data.

This repository provides four model variants for the binary activity coefficient prediction task:

| Model | Description | Input | Notes |
| :--- | :--- | :--- | :--- |
| SolvGNN | Baseline graph neural network from Qin et al. | Molecular graphs + composition | Composition information is injected before the mixture-level graph convolution |
| SolvGNNxMLP | SolvGNN variant | Molecular graphs + composition | Composition information is injected at the MLP layer |
| GEGNN | Excess Gibbs Free Energy GNN | Molecular graphs + composition | Predicts the excess Gibbs free energy and derives `ln(γᵢ)` from it |
| MCM | Matrix Completion Method (multi-MLP) | Solvent IDs + composition | Embedding-based, does not use molecular graphs |

## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model Name</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">MSE(Val / Test dataset)</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">solvgnn_binary_gamma</td>
            <td  nowrap="nowrap">output_binary_with_inf_all</td>
            <td  nowrap="nowrap">0.054291</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~8min</td>
            <td  nowrap="nowrap"><a href="solvgnn_binary_gamma.yaml">solvgnn_binary_gamma</a></td>
            <td  nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td  nowrap="nowrap">solvgnn_xmlp_binary_gamma</td>
            <td  nowrap="nowrap">output_binary_with_inf_all</td>
            <td  nowrap="nowrap">0.141971</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~8min</td>
            <td  nowrap="nowrap"><a href="solvgnn_xmlp_binary_gamma.yaml">solvgnn_xmlp_binary_gamma</a></td>
            <td  nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td  nowrap="nowrap">gegnn_binary_gamma</td>
            <td  nowrap="nowrap">output_binary_with_inf_all</td>
            <td  nowrap="nowrap">0.342185</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~8min</td>
            <td  nowrap="nowrap"><a href="gegnn_binary_gamma.yaml">gegnn_binary_gamma</a></td>
            <td  nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td  nowrap="nowrap">mcm_multimlp_binary_gamma</td>
            <td  nowrap="nowrap">output_binary_with_inf_all</td>
            <td  nowrap="nowrap">0.086560</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~5min</td>
            <td  nowrap="nowrap"><a href="mcm_multimlp_binary_gamma.yaml">mcm_multimlp_binary_gamma</a></td>
            <td  nowrap="nowrap">-</td>
        </tr>
    </body>
</table>

> The model trained above is based on a small dataset with 35374 records. Update the config file to use the full dataset.

### Training
```bash
# SolvGNN
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml

# SolvGNNxMLP
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_xmlp_binary_gamma.yaml

# GEGNN
python property_prediction/train.py -c property_prediction/configs/gdinn/gegnn_binary_gamma.yaml

# MCM (Matrix Completion Method)
python property_prediction/train.py -c property_prediction/configs/gdinn/mcm_multimlp_binary_gamma.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

# SolvGNN
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# SolvGNNxMLP
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_xmlp_binary_gamma.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# GEGNN
python property_prediction/train.py -c property_prediction/configs/gdinn/gegnn_binary_gamma.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# MCM
python property_prediction/train.py -c property_prediction/configs/gdinn/mcm_multimlp_binary_gamma.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# SolvGNN
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# SolvGNNxMLP
python property_prediction/train.py -c property_prediction/configs/gdinn/solvgnn_xmlp_binary_gamma.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# GEGNN
python property_prediction/train.py -c property_prediction/configs/gdinn/gegnn_binary_gamma.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'

# MCM
python property_prediction/train.py -c property_prediction/configs/gdinn/mcm_multimlp_binary_gamma.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```


## Citation
```
@article{rittig2023gibbs,
  title={Gibbs-Duhem-informed neural networks for binary activity coefficient prediction},
  author={Rittig, Jan G. and Felton, Kobi C. and Lapkin, Alexei A. and Mitsos, Alexander},
  journal={Digital Discovery},
  volume={2},
  number={6},
  pages={1752--1767},
  year={2023},
  publisher={Royal Society of Chemistry},
  doi={10.1039/D3DD00103B}
}
```
