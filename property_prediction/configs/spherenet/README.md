# SphereNet

[Spherical Message Passing for 3D Molecular Graphs](https://openreview.net/forum?id=givsRXsOt9r) (ICLR 2021)

## Abstract

We propose the spherical message passing (SMP) scheme for 3D molecular graphs,
which leverages **distance, angle, and torsion** information simultaneously to
uniquely identify the relative positions of atoms in 3D space. Previous
methods such as SchNet (distance-only) and DimeNet++ (distance + angle) suffer
from equivariance ambiguity because multiple spatial configurations can map
to the same pairwise distances or angles. By incorporating torsion angles
(dihedral angles), SphereNet resolves this ambiguity and achieves
state-of-the-art results on the QM9 and MD17 benchmarks.

## Datasets

### QM9

The QM9 dataset contains 130,831 small organic molecules (up to 9 heavy
atoms: C, O, N, F) with 12 quantum-chemical properties computed at the
B3LYP/6-31G(2df,p) level of theory.

| Split   | Size   |
|---------|--------|
| Train   | 110,831 |
| Val     | 10,000  |
| Test    | 10,000  |
| **Total** | **130,831** |

**Data format**: Each molecule contains atomic numbers (`z`), 3D positions
(`pos`), and 12 property labels. The raw dataset is available at
[figshare](https://figshare.com/ndownloader/files/3195389).

**Reference**: [Quantum-chemical insights from deep learning](https://arxiv.org/abs/1708.04444) (Gaussian, 2017)

### MD17

The MD17 dataset contains DFT molecular dynamics trajectories for 8 small
organic molecules. Each configuration includes the total energy (kcal/mol) and
atomic forces (kcal/mol/Å).

| Molecule      | Train | Val  | Test | Atoms |
|---------------|------:|-----:|-----:|------:|
| Aspirin       | 1000  | 500  | 1000 | 21    |
| Benzene       | 1000  | 500  | 1000 | 12    |
| Ethanol       | 1000  | 500  | 1000 | 9     |
| Malonaldehyde | 1000  | 500  | 1000 | 9     |
| Naphthalene   | 1000  | 500  | 1000 | 18    |
| Salicylic     | 1000  | 500  | 1000 | 16    |
| Toluene       | 1000  | 500  | 1000 | 15    |
| Uracil        | 1000  | 500  | 1000 | 12    |

**Data format**: Each molecule is stored as a single `.npz` file with keys
`E` (energies), `F` (forces), `R` (positions), and `z` (atomic numbers).

## Model

SphereNet is a spherical message passing neural network for 3D molecular
graphs. It represents each molecule as a graph where nodes correspond to
atoms, and directed edges encode interatomic interactions within a cutoff
radius. The model builds a hierarchy of geometric features and propagates
information using spherical message passing.

### Geometric embedding hierarchy

SphereNet constructs three levels of geometric embeddings to capture the
full 3D structure:

**1. Radial (distance) embeddings** — For each directed edge $j \to i$, the
interatomic distance $d_{ji}$ is expanded using a radial basis function
(RBF) composed with a smooth envelope.

**2. Angular (spherical) embeddings** — For each triplet $k \to j \to i$,
the bond angle $\theta_{kji}$ is expanded together with the distance
$d_{kj}$ using spherical Bessel functions combined with Legendre
polynomials (spherical Fourier-Bessel basis).

**3. Torsional embeddings** — For each quadruplet $l \to k \to j \to i$,
the torsion (dihedral) angle $\tau_{lkji}$ together with distances $d_{lk}$
and $d_{kj}$ is expanded using a 3D spherical Fourier-Bessel basis.

## Results

### QM9 (MAE, lower is better)

| Property | Unit | Paper MAE | Config |
|----------|------|----------:|--------|
| $\mu$ | D | 0.033 | [yaml](spherenet_qm9_mu.yaml) |
| $\alpha$ | Bohr³ | 0.235 | [yaml](spherenet_qm9_alpha.yaml) |
| $\varepsilon_{\text{HOMO}}$ | meV | 43.0 | [yaml](spherenet_qm9_homo.yaml) |
| $\varepsilon_{\text{LUMO}}$ | meV | 43.0 | [yaml](spherenet_qm9_lumo.yaml) |
| $\Delta\varepsilon$ | meV | 63.0 | [yaml](spherenet_qm9_gap.yaml) |
| $\langle R^2 \rangle$ | Bohr² | 0.295 | [yaml](spherenet_qm9_r2.yaml) |
| ZPVE | meV | 1.36 | [yaml](spherenet_qm9_zpve.yaml) |
| $U_0$ | meV | 22.0 | [yaml](spherenet_qm9_U0.yaml) |
| $U$ | meV | 22.0 | [yaml](spherenet_qm9_U.yaml) |
| $H$ | meV | 22.0 | [yaml](spherenet_qm9_H.yaml) |
| $G$ | meV | 22.0 | [yaml](spherenet_qm9_G.yaml) |
| $C_v$ | cal/(mol·K) | 0.053 | [yaml](spherenet_qm9_Cv.yaml) |

### MD17

| Molecule      | Config |
|---------------|--------|
| Aspirin       | [yaml](spherenet_md17_aspirin.yaml) |
| Benzene       | [yaml](spherenet_md17_benzene_old.yaml) |
| Ethanol       | [yaml](spherenet_md17_ethanol.yaml) |
| Malonaldehyde | [yaml](spherenet_md17_malonaldehyde.yaml) |
| Naphthalene   | [yaml](spherenet_md17_naphthalene.yaml) |
| Salicylic     | [yaml](spherenet_md17_salicylic.yaml) |
| Toluene       | [yaml](spherenet_md17_toluene.yaml) |
| Uracil        | [yaml](spherenet_md17_uracil.yaml) |

### Training

```bash
# Single-GPU training — QM9 mu property
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml

# Single-GPU training — MD17 aspirin (energy + force)
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_md17_aspirin.yaml
```

### Validation / Testing

```bash
# Validation
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path='your_model.pdparams'

# Testing
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml \
  Global.do_test=True Global.do_train=False Global.do_eval=False \
  Trainer.pretrained_model_path='your_model.pdparams'
```

## Citation

```bibtex
@inproceedings{liu2021spherenet,
  title={Spherical Message Passing for 3D Molecular Graphs},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Lin, Yuchao and Zhang, Xuan and
          Oztekin, Bora and Ji, Shuiwang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
