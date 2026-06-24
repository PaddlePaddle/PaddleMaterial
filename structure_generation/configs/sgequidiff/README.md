# SGEquiDiff

[SPACE GROUP EQUIVARIANT CRYSTAL DIFFUSION](https://arxiv.org/abs/2505.10994)

## Abstract

We introduce SGEquiDiff, a diffusion model for crystal structure prediction that operates entirely within the asymmetric unit (ASU) of a crystallographic space group. By modeling the joint distribution over Wyckoff positions, atomic species, lattice parameters, and space group assignments, SGEquiDiff generates crystals that are consistent with the symmetries of the 230 space groups. We demonstrate the effectiveness of our model on the MP-20 and MPTS-52 crystal structure prediction benchmarks.

---

## Model Description

### Overview

A crystal is represented by its asymmetric unit:
- space group: $g \in \{1,\ldots,230\}$
- lattice parameters: $L = (a,b,c,\alpha,\beta,\gamma)$
- Wyckoff positions: $W = (w_1,\ldots,w_N)$
- atomic species: $E = (e_1,\ldots,e_N)$
- fractional coordinates: $X = (x_1,\ldots,x_N),\; x_i \in \text{ASU}_g$

SGEquiDiff generates crystals in four stages:
1. **Space group** -- a categorical distribution over 230 space groups
2. **Lattice parameters** -- telescoping discrete sampling constrained by Bravais lattice type
3. **Wyckoff positions and elements** -- autoregressive Transformer over Wyckoff sites
4. **Fractional coordinates** -- VE-SDE diffusion on the ASU-wrapped torus

### Method

#### 1) Space group and lattice sampling
The space group is sampled from a learnable categorical distribution. Lattice parameters are discretized via a telescoping binning scheme that respects the Bravais-lattice constraints of each space group.

#### 2) Autoregressive Wyckoff/element sampling
A Transformer decoder autoregressively predicts the next Wyckoff position and atomic element, conditioned on previously sampled sites and the global space-group/lattice context.

#### 3) Equivariant diffusion on the ASU torus
Fractional coordinates are corrupted with VE-SDE noise wrapped into the ASU. The denoiser predicts an equivariant score field, and sampling uses a predictor-corrector scheme with Wyckoff-subspace projection after each step.

---

## Dataset Description

- **MP-20**: 45,231 inorganic crystals (Materials Project subset) with up to 20 atoms per unit cell.
- **MPTS-52 (Materials Project Time Split)**: 40,476 crystals with up to 52 atoms per cell; chronological split for temporal generalization.

Data is stored in ASU representation as `.npz` + `.pkl` files. Each sample contains space group index, composition vector, lattice parameters, and per-atom Wyckoff indices / element indices / fractional coordinates.

---

## Results

Pretrained weights are hosted on **AiStudio (Paddle format)**. Each dataset has 4 sub-module weight files (diffusion / lattice / space_group / wyckoff). The model auto-downloads all 4 via `load_pretrained_weights()`.

| Dataset | Sub-module | Download |
| --- | --- | --- |
| mp_20 | diffusion | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_diffusion_snapshot.pdparams) |
| mp_20 | lattice | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_lattice_snapshot.pdparams) |
| mp_20 | space_group | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_space_group_snapshot.pdparams) |
| mp_20 | wyckoff | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_wyckoff-transformer_snapshot.pdparams) |
| mpts_52 | diffusion | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_diffusion_snapshot.pdparams) |
| mpts_52 | lattice | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_lattice_snapshot.pdparams) |
| mpts_52 | space_group | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_space_group_snapshot.pdparams) |
| mpts_52 | wyckoff | [download](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_wyckoff-transformer_snapshot.pdparams) |

---

## Command

### Setup
```bash
export SGEQUIFF_DATA_DIR=/path/to/data
```

### Training
```bash
python structure_generation/train.py -c structure_generation/configs/sgequidiff/sgequidiff_mp20.yaml
```

### Generation
```bash
python structure_generation/sample.py --config_path=structure_generation/configs/sgequidiff/sgequidiff_mp20_sample.yaml --checkpoint_path=/path/to/weight_dir --mode=by_num_atoms --num_atoms=8 --save_path=./sgequidiff_samples
```

---

## Citation
```
@misc{chang2025spacegroupequivariantcrystal,
  title={Space Group Equivariant Crystal Diffusion},
  author={Rees Chang and Angela Pak and Alex Guerra and Ni Zhan and Nick Richardson and Elif Ertekin and Ryan P. Adams},
  year={2025},
  eprint={2505.10994},
  archivePrefix={arXiv},
}
```
