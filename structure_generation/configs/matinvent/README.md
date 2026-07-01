# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

---

## Model Description

### Overview

The framework operates in three alternating phases:

1. **Sample** -- The diffusion backbone generates a batch of crystal structures.
2. **Score** -- Property calculators (density, HHI, band gap, synthesizability, etc.) evaluate each structure; a scalar reward is computed from the multi-property profile.
3. **Fine-tune** -- The generator is updated with a reward-weighted diffusion loss plus a KL regularizer that prevents the agent from drifting too far from the pretrained prior.

After each iteration, high-reward structures are stored in a replay buffer and a long-term memory for diversity filtering.

### Supported Backbones

| Backbone | Description | Config |
|----------|-------------|--------|
| **MatterGen** | E(3)-equivariant GNN denoiser with joint A/X/L diffusion | `matinvent_mattergen.yaml` |
| **DiffCSP** | EGNN-based CSP denoiser with lattice + coordinate diffusion | `matinvent_diffcsp.yaml` |

### Supported Property Calculators

| Calculator | Tasks | Description |
|------------|-------|-------------|
| **PyMatGen** | `density`, `hhi`, `price`, `abundance`, `log_abundance`, `mcia`, `num_atoms`, `num_elements`, `volume` | On-the-fly property computation via pymatgen |
| **SynScore** | `synthesizability` | ML-based synthesizability prediction |
| **DFTCalc** | (external VASP) | DFT property calculator interface (exports CIFs for external computation) |
| **FairChem** | `bulk_modulus`, `heat_capacity` | ML potential-based property calculator interface |

Reward configuration files are located in `configs/reward/` (see original [MatInvent](https://github.com/jwchen25/MatInvent) repository for full reward configurations).

---

## Dataset Description

### Atom count sampling

MatInvent samples random atom counts uniformly from a configurable range:

| Backbone | Min atoms | Max atoms |
|----------|-----------|-----------|
| DiffCSP  | 1         | 50        |
| MatterGen | 2        | 50        |

### Reference dataset

Novelty and stability of generated structures are assessed against a reference convex-hull dataset (`reference_MP2020correction.gz`) available from [Hugging Face (jwchen25/MatInvent)](https://huggingface.co/jwchen25/MatInvent).

### Reward computation

For property-conditioned RL, rewards are computed **on-the-fly** by property calculators. No pre-labeled training set is required.

---

## Results

Key RL metrics tracked during training include **reward mean**, **burden** (computational cost per high-reward candidate), and **diversity ratio** (unique compositions / total evaluations). Post-hoc generation quality is reported as the **SUN ratio** (Stable, Unique, Novel fraction).

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Backbone</th>
            <th nowrap="nowrap">Reward Target</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">matinvent_mattergen_mp20</td>
            <td nowrap="nowrap">MatterGen</td>
            <td nowrap="nowrap">density</td>
            <td nowrap="nowrap"><a href="matinvent_mattergen.yaml">matinvent_mattergen</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_mattergen_mp20.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">matinvent_diffcsp_mp20</td>
            <td nowrap="nowrap">DiffCSP</td>
            <td nowrap="nowrap">density</td>
            <td nowrap="nowrap"><a href="matinvent_diffcsp.yaml">matinvent_diffcsp</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_diffcsp_mp20.zip">checkpoint | log</a></td>
        </tr>
    </tbody>
</table>

Refer to the [paper](https://arxiv.org/abs/2511.03112) for full quantitative results across multiple property targets.

---

## Command

### Training (RL Fine-tuning)

```bash
# MatterGen backbone
python structure_generation/train.py \
    -c structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone
python structure_generation/train.py \
    -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml
```

RL training hyperparameters (epochs, reward targets, etc.) are configured in the `RL:` section of the YAML config. Modify the `reward_cfg.prop_cfg` list to change optimization targets.

### Sample

Generate structures using an RL-fine-tuned checkpoint.

```bash
# Mode 1: Use a pre-trained model (downloads automatically)
python structure_generation/sample.py \
    --model_name='matinvent_mattergen_mp20' \
    --weights_name='matinvent_mattergen_mp20.pdparams' \
    --save_path='result_matinvent_mattergen/' \
    --mode='by_dataloader'

# Mode 2: Use a custom configuration and checkpoint
python structure_generation/sample.py \
    --config_path='structure_generation/configs/matinvent/matinvent_mattergen.yaml' \
    --checkpoint_path='./output/matinvent_mattergen/models/final/model.pdparams' \
    --save_path='result_matinvent_mattergen/' \
    --mode='by_dataloader'

# DiffCSP backbone
python structure_generation/sample.py \
    --config_path='structure_generation/configs/matinvent/matinvent_diffcsp.yaml' \
    --checkpoint_path='./output/matinvent_diffcsp/models/final/model.pdparams' \
    --save_path='result_matinvent_diffcsp/' \
    --mode='by_dataloader'
```

---

## Citation

```bibtex
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```