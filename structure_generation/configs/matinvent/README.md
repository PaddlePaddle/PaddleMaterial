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
