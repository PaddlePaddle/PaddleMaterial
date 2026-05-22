# OMatG

## Overview

A generative framework for crystal structure prediction and *de novo* generation of inorganic crystals. 
This open-source framework accompanies the [ICML 2025 paper](https://openreview.net/forum?id=gHGrzxFujU) about the 
generative OMatG model itself, the 
[NeurIPS 2025 paper](https://openreview.net/forum?id=ig9ujp50D4) about newly introduced benchmark metrics and datasets, 
and the OMatG-IRL [preprint](https://arxiv.org/abs/2602.00424) about reinforcement learning for pretrained OMatG 
models. 
These papers should be cited when using OMatG, the newly introduced benchmark metrics and datasets, 
or OMatG-IRL. 

Paper: Stochastic Interpolants for Crystal Structure Prediction (ICML 2025, NeurIPS 2025).

## Model Architecture

OMatG consists of the following core components:

- **CSPNet**: Crystal structure prediction network using message-passing GNN architecture, supporting fully-connected (fc) or k-NN (knn) edge construction
- **Stochastic Interpolants**: SI framework supporting ODE/SDE integration, with multiple interpolant types (Linear/Trigonometric/EncDec/VESBD/VPSBD)
- **IndependentSampler**: Independent distribution sampler for position/lattice/species base distributions


### Supported Datasets

| Dataset | Description | Pretrained Variants |
|---------|-------------|---------------------|
| mp_20_csp | Materials Project 20 (CSP mode) | 11 |
| mp_20_dng | Materials Project 20 (DNG mode, with masked species) | 11 |
| perov_5_csp | Perovskite 5 elements (CSP mode) | 11 |
| mpts_52_csp | MPTS 52 elements (CSP mode) | 8 |
| alex_mp_20_csp | Alex-MP20 (CSP mode) | 11 |

## Requirements

- PaddlePaddle >= 3.1
- paddle_scatter
- ase >= 3.23.0
- pymatgen >= 2024.10.29
- omegaconf >= 2.3.0

## Configuration Files

| File | Mode | Description |
|------|------|-------------|
| `omatg_mp20_csp.yaml` | CSP | MP-20 CSP main training config (fc edges, simplified MSE loss) |
| `omatg_mp20_csp_linear_ode.yaml` | CSP | Linear-ODE variant training config (with Cosine LR) |
| `omatg_mp20_dng_linear_sde.yaml` | DNG | Linear-SDE variant training config (with species prediction + WeightDecay) |
| `omatg_mp20_csp_sample.yaml` | CSP | Sampling config (by number of atoms, NumAtomsCrystalDataset) |

## Quick Start (Three Standard Paths)

### Path 1: Python API via build_model

```python
import paddle
from ppmat.models import build_model

# Minimal config to build the model (random init)
cfg = {
    "__class_name__": "OMATGCSPNetFull",
    "__init_params__": {
        "hidden_dim": 512, "num_layers": 6, "max_atoms": 100,
        "act_fn": "silu", "dis_emb": "sin", "num_freqs": 128,
        "edge_style": "fc", "cutoff": 7.0, "max_neighbors": 20,
        "ln": True, "ip": True, "time_embed_dim": 256,
    }
}
model = build_model(cfg)

# Build batch data (CSP mode, 2 structures, 3+2=5 atoms)
data = {
    "atom_types": paddle.randint(1, 10, [5], dtype="int64"),
    "frac_coords": paddle.rand([5, 3]),
    "lattices": paddle.eye(3).unsqueeze(0).expand([2, 3, 3]) * 5.0,
    "num_atoms": paddle.to_tensor([3, 2], dtype="int64"),
    "node2graph": paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64"),
}

# Forward pass and loss (simplified MSE path)
output = model(data)
print(output["loss_dict"].keys())  # keys: loss, loss_lattice, loss_coord

# Sample crystal structures
result = model.sample(data, num_inference_steps=100)
print(len(result["result"]))       # 2 structures (matching batch size)
```

### Path 2: Training Entry (train.py)

```bash
# Smoke training (1 epoch, small batch, no eval)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Trainer.max_epochs=1 \
    Trainer.do_eval=False \
    Dataset.train.sampler.__init_params__.batch_size=32 \
    Dataset.val.sampler.__init_params__.batch_size=32

# Full training (requires data/mp_20/ dataset)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp_linear_ode.yaml \
    Trainer.max_epochs=500 \
    Trainer.output_dir=./output/omatg_mp20_csp
```

### Path 3: Sampling Entry (sample.py)

```bash
# Sample by number of atoms (requires a trained checkpoint)
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp_sample.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_num_atoms \
    --num_atoms 8 \
    --save_path ./results/omatg_samples

# Sample by chemical formula
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp_sample.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_chemical_formula \
    --chemical_formula LiMnO2 \
    --save_path ./results/omatg_samples

# Batch sample by dataloader
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_dataloader \
    --save_path ./results/omatg_samples
```

## Pretrained Model Loading

OMatG ships 52 pretrained weights covering 5 datasets and 11 variants. Use the Python API:

```python
from ppmat.models.omatg import build_omatg_model

# Build model and auto-download weights
model, config = build_omatg_model("mp_20_csp", "linear_ode")

# Sample
data = {"structure_array": {"num_atoms": paddle.to_tensor([8], dtype="int64")}}
result = model.sample(data, num_inference_steps=100)

# Get weight URL
from ppmat.models.omatg import get_omatg_model_url
url = get_omatg_model_url("mp_20_csp", "trig_ode_gamma")
```

## SI Training Path (Velocity Matching Loss)

Switch from `use_si=False` (default, simplified MSE) to `use_si=True` (full SI velocity matching):

```python
import paddle
from ppmat.models.omatg import OMATGCSPNetFull
from ppmat.models.omatg.si import (
    StochasticInterpolants, SingleStochasticInterpolant,
    SingleStochasticInterpolantIdentity,
    PeriodicLinearInterpolant, LinearInterpolant,
)
from ppmat.models.omatg.sampler import (
    IndependentSampler, UniformPositionDistribution,
    InformedLatticeDistribution, MirrorSpecies,
)

# Build SI instance (Linear-ODE)
si = StochasticInterpolants(
    stochastic_interpolants=[
        SingleStochasticInterpolantIdentity(),
        SingleStochasticInterpolant(
            interpolant=PeriodicLinearInterpolant(), gamma=None,
            epsilon=None, differential_equation_type="ODE",
            velocity_annealing_factor=10.18,
            correct_center_of_mass_motion=True,
        ),
        SingleStochasticInterpolant(
            interpolant=LinearInterpolant(), gamma=None,
            epsilon=None, differential_equation_type="ODE",
            velocity_annealing_factor=1.82,
        ),
    ],
    data_fields=["species", "pos", "cell"],
    integration_time_steps=210,
)
sampler = IndependentSampler(
    position_distribution=UniformPositionDistribution(),
    cell_distribution=InformedLatticeDistribution("mp_20"),
    species_distribution=MirrorSpecies(),
)
costs = {"species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006}

model = OMATGCSPNetFull(hidden_dim=512, num_layers=6, max_atoms=100,
                        time_embed_dim=256, pred_type=False, use_si=False)
model._si = si
model._sampler = sampler
model._relative_si_costs = costs
model.use_si = True

data = {
    "atom_types": paddle.randint(1, 10, [5], dtype="int64"),
    "frac_coords": paddle.rand([5, 3]),
    "lattices": paddle.eye(3).unsqueeze(0).expand([2, 3, 3]) * 5.0,
    "num_atoms": paddle.to_tensor([3, 2], dtype="int64"),
    "node2graph": paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64"),
}
output = model(data)
print(output["loss_dict"].keys())
# CSP mode keys: species_loss, pos_loss_b, cell_loss_b, loss
# DNG SDE mode additional: pos_loss_z
```

You can also use the config-driven factory functions:

```python
from ppmat.models.omatg.si import build_si_from_cfg, build_sampler_from_cfg

si_cfg = {
    "stochastic_interpolants": [
        {"__class_name__": "SingleStochasticInterpolantIdentity"},
        {"__class_name__": "SingleStochasticInterpolant",
         "__init_params__": {
             "interpolant": {"__class_name__": "PeriodicLinearInterpolant"},
             "gamma": None, "epsilon": None,
             "differential_equation_type": "ODE",
             "velocity_annealing_factor": 10.18,
             "correct_center_of_mass_motion": True,
         }},
        {"__class_name__": "SingleStochasticInterpolant",
         "__init_params__": {
             "interpolant": {"__class_name__": "LinearInterpolant"},
             "differential_equation_type": "ODE",
             "velocity_annealing_factor": 1.82,
         }},
    ],
    "data_fields": ["species", "pos", "cell"],
    "integration_time_steps": 210,
    "relative_si_costs": {"species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006},
}
sampler_cfg = {
    "position_distribution": {"__class_name__": "UniformPositionDistribution"},
    "cell_distribution": {"__class_name__": "InformedLatticeDistribution",
                          "__init_params__": {"dataset_name": "mp_20"}},
    "species_distribution": {"__class_name__": "MirrorSpecies"},
}
model = OMATGCSPNetFull(hidden_dim=512, num_layers=6, max_atoms=100,
                        time_embed_dim=256, pred_type=False,
                        use_si=True, si_cfg=si_cfg, sampler_cfg=sampler_cfg)
```

## Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_dim` | Hidden dimension | 128 |
| `num_layers` | Number of message-passing layers | 4 |
| `max_atoms` | Maximum number of atoms | 100 |
| `edge_style` | Edge construction method (fc/knn) | fc |
| `cutoff` | Distance cutoff radius for k-NN mode | 6.0 |
| `max_neighbors` | Maximum number of neighbors | 20 |
| `time_embed_dim` | Time embedding dimension | 256 |
| `pred_type` | Whether to predict atom types (True for DNG mode) | False |
| `use_si` | Whether to use SI velocity-matching loss (default: simplified MSE) | False |

## Supported Model Variants

| Interpolant | Equation Type | ODE | SDE |
|-------------|---------------|-----|-----|
| Linear | Linear | Supported | Supported |
| Trigonometric | Trigonometric | Supported | Supported |
| EncDec | Encoder-decoder | Supported | Supported |
| VPSBD | Variance-preserving | Supported | Supported |
| VESBD | Variance-exploding | Supported | - |

Note: Variants with `_gamma` suffix use latent gamma noise (available for both ODE and SDE). `VESBD/VPSBD` variants use a pure-Paddle fixed-step Euler solver.

## Unit Tests

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ppmat
python -m pytest test/omatg/test_omatg.py -v
```
