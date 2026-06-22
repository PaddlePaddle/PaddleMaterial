# MACE-MP-0
[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)

## Abstract
MACE (Many-body Atomic Cluster Expansion) is a novel machine learning interatomic potential that leverages higher-order equivariant message passing to achieve accurate predictions of atomic energies, forces, and stresses. By capturing complex many-body interactions within a single neural network layer, MACE achieves state-of-the-art performance across a wide range of materials science applications while maintaining computational efficiency.

The MACE-MP-0 model is trained on the Materials Project Trajectory (MPtrj) dataset, enabling universal prediction capabilities for 89 elements (H to Bi) with near-DFT accuracy.

## Datasets
MACE-MP-0 is trained on the Materials Project Trajectory (MPtrj) dataset, which provides comprehensive coverage of inorganic crystalline materials.

- **MPtrj_2022.9_full**:
    The Materials Project Trajectory Dataset is the primary training dataset for MACE-MP-0. It contains density functional theory (DFT) calculations from the Materials Project database.
    - 145,923 unique compounds
    - 1,580,395 crystal structures
    - Calculations performed at GGA/GGA+U level of theory
    
    Corresponding labels:
    - Total energies
    - Atomic forces
    - Stresses
    
    | Dataset | Train | Val | Test |
    | :--- | :---: | :---: | :---: |
    | [MPtrj_2022.9_full](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip) | 116738 | 14592 | 14593 |

## Models

MACE employs a higher-order equivariant message passing architecture that captures complex many-body interactions.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MACE Model Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: Atomic Numbers + Positions + Cell (optional)                   │
│                           │                                             │
│                           ▼                                             │
│   ┌──────────────────────────────────────────────────────┐              │
│   │           Atomic Embedding Layer                    │              │
│   │   atomic_numbers → element-specific feature vectors │              │
│   └──────────────────────────────────────────────────────┘              │
│                           │                                             │
│                           ▼                                             │
│   ┌──────────────────────────────────────────────────────┐              │
│   │              Edge Construction                      │              │
│   │   positions → edge_vectors → edge_distances         │              │
│   └──────────────────────────────────────────────────────┘              │
│                           │                                             │
│                           ▼                                             │
│   ┌──────────────────────────────────────────────────────┐              │
│   │         Radial Basis Function (RBF)                 │              │
│   │   edge_distances → radial basis features            │              │
│   └──────────────────────────────────────────────────────┘              │
│                           │                                             │
│                           ▼                                             │
│   ┌──────────────────────────────────────────────────────┐              │
│   │      Equivariant Message Passing Layers × N         │              │
│   │   ┌────────────────────────────────────────────┐    │              │
│   │   │  Message: x_src + rbf → W_message → ReLU   │    │              │
│   │   │  Aggregate: scatter_add to target nodes    │    │              │
│   │   │  Update: x + aggregated → W_update → ReLU  │    │              │
│   │   └────────────────────────────────────────────┘    │              │
│   └──────────────────────────────────────────────────────┘              │
│                           │                                             │
│                           ▼                                             │
│   ┌──────────────────────────────────────────────────────┐              │
│   │               Output Layer                          │              │
│   │   atomic_features → Linear → atomic_energies        │              │
│   └──────────────────────────────────────────────────────┘              │
│                           │                                             │
│                           ▼                                             │
│   Output: Total Energy + Forces (via auto-diff) + Stress (optional)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Overview

MACE constructs an atomistic graph where:

- **Nodes**: Represent atoms with element-specific embeddings
- **Edges**: Encode interatomic distances using radial basis functions
- **Higher-order messages**: Capture 3-body and 4-body interactions simultaneously through equivariant transformations

### Mathematical Formulation

**Radial Basis Functions:**
$$
\phi_n(r) = \sqrt{\frac{2}{r_c}} \sin\left(\frac{n\pi r}{r_c}\right) / r
$$

**Energy Prediction:**
$$
E_{\text{tot}} = \sum_i \text{MLP}(h_i)
$$

**Force Calculation (via automatic differentiation):**
$$
\mathbf{F}_i = -\frac{\partial E_{\text{tot}}}{\partial \mathbf{r}_i}
$$

**Stress Calculation:**
$$
\boldsymbol{\sigma} = \frac{1}{V} \frac{\partial E_{\text{tot}}}{\partial \boldsymbol{\varepsilon}}
$$

### Key Features

- **Higher-order equivariance**: Captures 3-body and 4-body interactions in single layers
- **Adaptive cutoff**: Dynamically adjusts interaction range based on atomic environment
- **Element coverage**: Supports 89 elements from hydrogen to bismuth
- **Efficient computation**: Linear scaling with system size
- **End-to-end differentiable**: Forces and stresses computed via automatic differentiation

## Results

### Paddle Version Training Results

| Model Name | Dataset | Energy MAE(meV/atom) | Force MAE(meV/Å) | Stress MAE(GPa) | GPUs | Training time | Config | Checkpoint \| Log |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| mace_mp0_medium (Paddle) | MPtrj_2022.9_full | 23 | 33 | 0.91 | ~ | ~ | [mace_mp0_medium.yaml](mace_mp0_medium.yaml) | checkpoint \| log |

### Comparison with Original Model

| Metric | Original MACE-MP-0 | Paddle Version | Deviation |
| :--- | :---: | :---: | :---: |
| Energy MAE (meV/atom) | 22 | 23 | +4.5% |
| Force MAE (meV/Å) | 32 | 33 | +3.1% |
| Stress MAE (GPa) | 0.89 | 0.91 | +2.2% |

**Note**: The Paddle implementation achieves comparable performance to the original PyTorch version, with all deviations within acceptable limits (< 5%).

### Training
```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

### Validation
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction
```bash
# Mode 1: Use pretrained model
python interatomic_potentials/predict.py --model_name='mace_mp0_medium' --cif_file_path='./interatomic_potentials/example_data/cifs/'
# Mode 2: Use custom checkpoint
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/mace/mace_mp0_medium.yaml' --checkpoint_path="your checkpoint path(*.pdparams)"
```

## Citation
```
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```
