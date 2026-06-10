# MLIP-Machine Learning Interatomic Potential

## 1.Introduction

Machine-learning interatomic potentials (MLIP) bridge the gap between quantum-level accuracy and classical molecular-dynamics speed. Traditional force fields rely on fixed functional forms and hand-tuned parameters, limiting transferability. In contrast, MLIP learn the energy-force landscape directly from high-fidelity density-functional-theory data, capturing many-body and chemical effects without explicit equations. Modern frameworks embed rigorous physical priors—permutation, rotation and translation invariance, smoothness, locality—into expressive models such as equivariant graph neural networks, message-passing networks, Gaussian process regressors and deep neural descriptors. A typical workflow begins by sampling diverse atomic configurations, computing reference energies, forces and stresses, then training the model with loss terms that balance all three quantities. Active-learning loops iteratively enrich the dataset where prediction uncertainty is high, minimizing human intervention. Once trained, an MLIP delivers near-DFT accuracy for million-atom, nanosecond-scale simulations at a small fraction of the cost, enabling studies of crack propagation, phase transitions, ion diffusion and catalytic reactions that were previously intractable. As datasets grow and architectures mature, MLIP are poised to become standard tools for predictive, large-scale materials and molecular modeling.

## 2.Models Matrix

| **Supported Functions**             | **[CHGNet](./configs/chgnet/README.md)** | **[MatterSim](./configs/mattersim//README.md)** | **[SevenNet](./configs/sevennet/README.md)** |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| **Forward Prediction**              |                                          |                                                 |                                              |
| &emsp;Energy                        | ✅                                       | ✅                                              | ✅                                           |
| &emsp;Force                         | ✅                                       | ✅                                              | ✅                                           |
| &emsp;Stress                        | ✅                                       | ✅                                              | -                                            |
| &emsp;Magmom                        | ✅                                       | -                                               | -                                            |
| **ML Capabilities · Training**      |                                          |                                                 |                                              |
| &emsp;Single-GPU                    | ✅                                       | ✅                                              | ✅                                           |
| &emsp;Distributed Train             | ✅                                       | ✅                                              | -                                            |
| &emsp;Mixed Precision               | -                                        | -                                               | -                                            |
| &emsp;Fine-tuning                   | ✅                                       | ✅                                              | ✅                                           |
| &emsp;Uncertainty / Active-Learning | -                                        | -                                               | -                                            |
| &emsp;Dynamic→Static                | -                                        | -                                               | -                                            |
| &emsp;Compiler CINN                 | -                                        | -                                               | -                                            |
| **ML Capabilities · Predict**       |                                          |                                                 |                                              |
| &emsp;Distillation / Pruning        | -                                        | -                                               | -                                            |
| &emsp;Standard inference            | ✅                                       | ✅                                              | ✅                                           |
| &emsp;Distributed inference         | -                                        | -                                               | -                                            |
| &emsp;Compiler CINN                 | -                                        | -                                               | -                                            |
| **Molecular Dynamic Interface**     |                                          |                                                 |                                              |
| &emsp;ASE                           | ✅                                       | ✅                                              | ✅                                           |
| **Dataset**                         |                                          |                                                 |                                              |
| &emsp;MPtrj                         | ✅                                       | 🚧                                              | -                                            |
| **ML2DDB🌟**                        | ✅                                       | -                                               | -                                            | 

**Notice**:🌟 represent originate research work published from paddlematerials toolkit

## 3.SevenNet Quick Start

### 3.1 Overview

SevenNet is a message-passing graph neural network for interatomic potential prediction. This implementation provides a pure PaddlePaddle version without external dependencies like e3nn.

### 3.2 Installation

```bash
# Install dependencies
pip install paddlepaddle>=3.0.0 ase numpy tqdm pyyaml
```

### 3.3 Training

```bash
# Navigate to interatomic_potentials directory
cd interatomic_potentials

# Run training with config file
python sevennet_train.py --config configs/sevennet/sevennet_hfo2.yaml
```

### 3.4 Inference

```bash
python sevennet_train.py --config configs/sevennet/sevennet_hfo2.yaml \
    --output-dir ./output/sevennet_hfo2
```

### 3.5 Configuration

The configuration file includes:
- **Model parameters**: hidden_dim, num_message_layers, num_rbf, cutoff
- **Dataset parameters**: path, cutoff, valid_ratio
- **Training parameters**: epoch, batch_size, learning_rate

### 3.6 Test

```bash
# Run unit tests
pytest test/test_sevennet.py -v
```

### 3.7 Notes

- SevenNet uses message-passing architecture with Gaussian RBF for distance encoding
- Supports energy and force prediction
- Pure PaddlePaddle implementation, no PyTorch/e3nn dependencies
- Recommended for small to medium scale molecular systems