# MatInvent

[MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)

## Abstract

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.

## Model Description

### Overview

The framework operates in three alternating phases:

1. **Sample** -- The diffusion backbone generates a batch of crystal structures.
2. **Score** -- Property calculators (density, HHI, band gap, synthesizability, etc.) evaluate each structure; a scalar reward is computed from the multi-property profile.
3. **Fine-tune** -- The generator is updated with a reward-weighted diffusion loss plus a KL regularizer that prevents the agent from drifting too far from the pretrained prior.

After each iteration, high-reward structures are stored in a replay buffer and a long-term memory for diversity filtering.

---

## Dataset Description

### Dataset contents

#### 1) Atom count sampling (DiffCSP backbone)
When using the **DiffCSP** backbone, MatInvent samples random atom counts uniformly between 1 and 50 atoms per unit cell.

#### 2) Atom count sampling (MatterGen backbone)
When using the **MatterGen** backbone, MatInvent samples random atom counts uniformly between 2 and 50 atoms per unit cell.

#### 3) Reference dataset for novelty and stability evaluation
Novelty and stability of generated structures are assessed against a reference convex-hull dataset (`reference_MP2020correction.gz`) downloaded automatically from [Hugging Face (jwchen25/MatInvent)](https://huggingface.co/jwchen25/MatInvent). This reference is based on the MP2020 energy correction scheme and is used during both RL training and post-hoc evaluation.

#### 4) Reward computation (no additional labeled dataset required)
For property-conditioned RL, rewards are computed **on-the-fly** by property calculators (PyMatGen). Each generated structure is scored immediately after sampling, so the RL loop is self-contained and requires no pre-labeled training set.

### Data format
Each structure sample produced by the diffusion backbone provides:
- `atom_types` / `atomic_numbers`: length-$N$ array of atomic numbers
- `frac_coords` / `pos`: $N \times 3$ fractional coordinates in $[0, 1)$
- `lengths` + `angles` or `cell`: lattice parameters / $3 \times 3$ lattice matrix

Optional fields used during RL: `reward` (scalar), `num_atoms`, `structure_id`.

---

## Results

Key RL metrics tracked during training include **reward mean**, **burden** (computational cost per high-reward candidate), and **diversity ratio** (unique compositions / total evaluations). Post-hoc generation quality is reported as the **SUN ratio** (Stable, Unique, Novel fraction). Refer to the [paper](https://arxiv.org/abs/2511.03112) for full quantitative results.

Pretrained checkpoints are available on [HuggingFace (jwchen25/MatInvent)](https://huggingface.co/jwchen25/MatInvent).

---

## Command

### Training

```bash
# MatterGen backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP backbone
python structure_generation/train.py -c structure_generation/configs/matinvent/matinvent_diffcsp.yaml
```

### Sample

```bash
# Mode 1: with RL-fine-tuned checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_mattergen.yaml' --checkpoint_path='./output/matinvent_mattergen/models/final/model.pdparams' --save_path='result_matinvent_mattergen/' --mode='by_dataloader'

# Mode 2: with DiffCSP backbone
python structure_generation/sample.py --config_path='structure_generation/configs/matinvent/matinvent_diffcsp.yaml' --checkpoint_path='./output/matinvent_diffcsp/models/final/model.pdparams' --save_path='result_matinvent_diffcsp/' --mode='by_dataloader'
```

---

## Citation

```
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```**
